"""LLM bridge: Claude Code subprocess wrapper for agent voting decisions.

Calls `claude -p --model haiku` with a structured persona prompt and parses
JSON response. Includes rate limiting and retry logic to avoid 429 errors.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

from agents.voter_agent import VoterAgent, VoteAction


@dataclass
class LLMContext:
    """Context passed to the LLM for a voting decision."""

    race_description: str
    candidates: list[str]
    trusted_opinions: dict[str, str]
    delegator_count: int


DEFAULT_TIMEOUT = 30
DEFAULT_WORKERS = 10  # parallel batch calls
DEFAULT_RPM = 600  # 10 calls/sec on Claude Max

FALLBACK_RESULT = {"action": "abstain", "reason": "llm_error"}


def _make_fallback(reason: str) -> dict[str, Any]:
    """Create a fallback result with a specific error reason."""
    _log_error(reason)
    return {"action": "abstain", "reason": reason}


def _log_error(msg: str) -> None:
    """Always log errors to stderr (even without LLM_DEBUG)."""
    import sys
    print(f"[LLM ERROR] {msg}", file=sys.stderr)

# Debug mode: set via LLM_DEBUG=1 env var or set_debug()
_debug = os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")


def set_debug(enabled: bool = True) -> None:
    """Enable/disable debug logging of all Claude req/resp pairs."""
    global _debug
    _debug = enabled


def _log_debug(label: str, content: str) -> None:
    """Print debug info to stderr if debug mode is on."""
    if not _debug:
        return
    import sys
    border = "=" * 60
    print(f"\n{border}", file=sys.stderr)
    print(f"[LLM DEBUG] {label}", file=sys.stderr)
    print(border, file=sys.stderr)
    print(content, file=sys.stderr)
    print(border, file=sys.stderr)


class AdaptiveRateLimiter:
    """Adaptive token-bucket rate limiter.

    Starts at `max_rpm` and halves the rate each time a rate-limit hit
    is reported. Gradually recovers toward max_rpm when requests succeed.
    Thread-safe.
    """

    MIN_RPM = 10
    RECOVERY_FACTOR = 1.05  # 5% increase per successful request

    def __init__(self, max_rpm: int = DEFAULT_RPM):
        self.max_rpm = max_rpm
        self._current_rpm = float(max_rpm)
        self._interval = 60.0 / max_rpm
        self._tokens = 10.0  # initial burst (matches max parallel workers)
        self._burst = 10
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._consecutive_ok = 0

    @property
    def current_rpm(self) -> float:
        return self._current_rpm

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._burst, self._tokens + elapsed / self._interval)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

            time.sleep(self._interval * 0.5)

    def report_success(self) -> None:
        """Call after a successful request — gradually recovers rate."""
        with self._lock:
            self._consecutive_ok += 1
            if self._consecutive_ok >= 5 and self._current_rpm < self.max_rpm:
                self._current_rpm = min(self.max_rpm, self._current_rpm * self.RECOVERY_FACTOR)
                self._interval = 60.0 / self._current_rpm
                self._consecutive_ok = 0

    def report_rate_limit(self) -> None:
        """Call when a rate limit is hit — halves the rate immediately."""
        with self._lock:
            self._current_rpm = max(self.MIN_RPM, self._current_rpm * 0.5)
            self._interval = 60.0 / self._current_rpm
            self._tokens = 0  # drain burst
            self._consecutive_ok = 0


# Global adaptive rate limiter
_rate_limiter = AdaptiveRateLimiter(max_rpm=DEFAULT_RPM)


SCALE_LEGEND = """\
SCALE REFERENCE (all values range from -1.0 to +1.0):
  -1.0 = strongly liberal/left    0.0 = moderate/centrist    +1.0 = strongly conservative/right
  Ideology dimensions: economy (taxes/spending), social (cultural issues), \
immigration (open/restrictive), foreign_policy (dove/hawk), guns (pro-control/pro-rights), \
environment (pro-regulation/pro-industry), healthcare (universal/market), \
criminal_justice (reform/tough-on-crime), trade (free-trade/protectionist), \
government_size (big-government/small-government)
  Education: 0=no HS, 1=HS, 2=some college, 3=bachelor's, 4=master's, 5=PhD
  Knowledge: 0.0=knows nothing about this race, 1.0=deeply informed
  Trust: 0.0=no trust, 1.0=complete trust
  Party IDs: strong_D/lean_D/independent/lean_R/strong_R"""


def build_prompt(agent: VoterAgent, race_id: str, context: LLMContext) -> str:
    """Build the persona prompt for an LLM agent."""
    demo = agent.demographics
    ideo = agent.ideology
    ideology_summary = (
        f"economy={ideo[0]:+.1f}, social={ideo[1]:+.1f}, "
        f"immigration={ideo[2]:+.1f}, foreign_policy={ideo[3]:+.1f}, "
        f"guns={ideo[4]:+.1f}, environment={ideo[5]:+.1f}, "
        f"healthcare={ideo[6]:+.1f}, criminal_justice={ideo[7]:+.1f}, "
        f"trade={ideo[8]:+.1f}, government_size={ideo[9]:+.1f}"
    )
    knowledge = agent.race_states.get(race_id)
    knowledge_val = f"{knowledge.knowledge_level:.1f}" if knowledge else "0.0"

    persona = (
        f"Age {demo.age}, {demo.gender}, {demo.urban_rural}, "
        f"party: {agent.party_id.value}, "
        f"education: {demo.education}/5, "
        f"income: ${demo.income:,.0f}/yr.\n"
        f"  Ideology: {ideology_summary}\n"
        f"  {context.delegator_count} people delegated their vote to you. "
        f"Knowledge of this race: {knowledge_val}/1.0"
    )

    trusted_opinions_str = "\n".join(
        f"  - {name}: {opinion}" for name, opinion in context.trusted_opinions.items()
    )
    if not trusted_opinions_str:
        trusted_opinions_str = "  (no trusted opinions available)"

    return (
        f"You are a voter in a US congressional election.\n\n"
        f"{SCALE_LEGEND}\n\n"
        f"YOUR PROFILE:\n{persona}\n\n"
        f"RACE: {context.race_description}\n"
        f"CANDIDATES: {', '.join(context.candidates)}\n\n"
        f"YOUR TRUST NETWORK (people you know and their leanings):\n"
        f"{trusted_opinions_str}\n\n"
        f"Based on your profile, ideology, trust network, and knowledge, "
        f"decide how to vote. Respond ONLY with valid JSON, no other text:\n"
        f'{{"action": "vote"|"delegate"|"abstain", '
        f'"choice": "candidate name or delegate agent_id", '
        f'"reason": "one sentence"}}'
    )


def call_claude(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: str = "haiku",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call claude -p with rate limiting and retry on failure.

    Retries with exponential backoff on rate limit (exit code != 0) or timeout.
    """
    for attempt in range(max_retries):
        _rate_limiter.acquire()

        _log_debug(f"REQUEST (model={model}, attempt={attempt+1}/{max_retries})", prompt)

        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stderr = result.stderr.lower() if result.stderr else ""
            if result.returncode != 0:
                _log_debug(f"ERROR (exit={result.returncode})", result.stderr or "(no stderr)")
                if "rate" in stderr or "429" in stderr or "limit" in stderr or "overloaded" in stderr:
                    _rate_limiter.report_rate_limit()
                    backoff = 2 ** attempt * 3
                    _log_debug("RATE LIMITED", f"Backing off {backoff}s, RPM now {_rate_limiter.current_rpm:.0f}")
                    time.sleep(backoff)
                    continue
                return FALLBACK_RESULT

            output = result.stdout.strip()
            _log_debug("RESPONSE (raw)", output[:500])
            if "```" in output:
                lines = output.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                output = "\n".join(json_lines).strip()

            parsed = json.loads(output)

            if "action" not in parsed:
                return FALLBACK_RESULT
            if parsed["action"] not in ("vote", "delegate", "abstain"):
                return FALLBACK_RESULT

            _rate_limiter.report_success()
            _log_debug("PARSED", json.dumps(parsed, indent=2))
            return parsed

        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return FALLBACK_RESULT
        except (json.JSONDecodeError, OSError):
            return FALLBACK_RESULT

    return FALLBACK_RESULT


def llm_agent_decision(
    agent: VoterAgent, race_id: str, context: LLMContext
) -> dict[str, Any]:
    """Make a voting decision for a single LLM agent."""
    prompt = build_prompt(agent, race_id, context)
    return call_claude(prompt)


def run_llm_decisions_batched(
    llm_agents: list[VoterAgent],
    race_id: str,
    contexts: dict[int, LLMContext],
    max_workers: int = DEFAULT_WORKERS,
    batch_size: int = 100,
    batch_pause: float = 1.0,
    progress_fn: Any = None,
) -> dict[int, dict[str, Any]]:
    """Run LLM decisions concurrently with per-completion progress.

    All agents are submitted at once. The rate limiter inside call_claude()
    handles backpressure. Progress is reported after each agent completes.

    Args:
        progress_fn: optional callable(completed, total) for progress reporting
    """
    results: dict[int, dict[str, Any]] = {}
    total = len(llm_agents)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                llm_agent_decision, agent, race_id, contexts[agent.agent_id]
            ): agent
            for agent in llm_agents
        }
        for future in as_completed(futures):
            agent = futures[future]
            try:
                results[agent.agent_id] = future.result()
            except Exception:
                results[agent.agent_id] = FALLBACK_RESULT
            completed += 1
            if progress_fn:
                progress_fn(completed, total)

    return results


# ---------------------------------------------------------------------------
# Batched prompt: one Claude call for N agents
# ---------------------------------------------------------------------------

BATCH_SIZE = 20  # conservative batch size for reliability


def _compact_agent_line(agent: VoterAgent, race_id: str, context: LLMContext) -> str:
    """Ultra-compact one-line agent summary for batched prompts (~150 chars)."""
    ideo = agent.ideology
    knowledge = agent.race_states.get(race_id)
    k_val = f"{knowledge.knowledge_level:.1f}" if knowledge else "0"

    # Primary ideology (economy) + party is the most decision-relevant info
    return (
        f"{agent.agent_id}|{agent.party_id.value}|"
        f"eco={ideo[0]:+.1f} soc={ideo[1]:+.1f} imm={ideo[2]:+.1f}|"
        f"k={k_val}|d={context.delegator_count}"
    )


def build_batch_prompt(
    agents: list[VoterAgent],
    race_id: str,
    contexts: dict[int, LLMContext],
    candidates: list[str],
) -> str:
    """Build a single prompt requesting decisions for multiple agents."""
    agent_lines = []
    for agent in agents:
        ctx = contexts[agent.agent_id]
        agent_lines.append(_compact_agent_line(agent, race_id, ctx))

    agents_block = "\n".join(agent_lines)
    agent_ids = [str(a.agent_id) for a in agents]

    return (
        f"Simulate {len(agents)} voters. Candidates: {', '.join(candidates)}.\n"
        f"Format: ID|party|ideology(eco,soc,imm)|knowledge|delegators\n"
        f"Scale: -1=liberal, 0=center, +1=conservative. k=knowledge(0-1). d=delegators.\n\n"
        f"{agents_block}\n\n"
        f"Reply with ONLY a JSON object.\n"
        f'{{"ID": {{"a": "v"|"d"|"x", "c": "candidate_name", "r": "3-5 word reason"}}, ...}}\n'
        f'a=action: v=vote, d=delegate, x=abstain. c=choice. r=reason.\n'
        f"IDs: {', '.join(agent_ids)}"
    )


def _call_claude_batch(
    prompt: str,
    agent_ids: list[int],
    timeout: int = 180,
    model: str = "haiku",
    max_retries: int = 3,
) -> dict[int, dict[str, Any]]:
    """Call Claude with a batched prompt and parse the multi-agent response."""
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        _log_debug(f"BATCH REQUEST ({len(agent_ids)} agents, model={model}, attempt={attempt+1})",
                   prompt[:500] + f"\n... ({len(prompt)} chars total)")

        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stderr_raw = result.stderr or ""
            stderr = stderr_raw.lower()
            if result.returncode != 0:
                _log_debug(f"BATCH ERROR (exit={result.returncode})", stderr_raw)
                if "rate" in stderr or "429" in stderr or "limit" in stderr or "overloaded" in stderr:
                    _rate_limiter.report_rate_limit()
                    time.sleep(2 ** attempt * 3)
                    continue
                reason = f"claude exit {result.returncode}: {stderr_raw[:200]}"
                return {aid: _make_fallback(reason) for aid in agent_ids}

            output = result.stdout.strip()
            _log_debug("BATCH RESPONSE (raw)", output[:1000])

            if not output:
                reason = f"empty response from claude (stderr: {stderr_raw[:200]})"
                return {aid: _make_fallback(reason) for aid in agent_ids}

            # Strip markdown code blocks if present
            if "```" in output:
                lines = output.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                output = "\n".join(json_lines).strip()

            parsed = json.loads(output)
            _rate_limiter.report_success()

            # Parse the response map — handles both full and compact formats
            ACTION_MAP = {"v": "vote", "d": "delegate", "x": "abstain",
                          "vote": "vote", "delegate": "delegate", "abstain": "abstain"}
            results: dict[int, dict[str, Any]] = {}
            missing_ids = []
            bad_action_ids = []
            for aid in agent_ids:
                entry = parsed.get(str(aid)) or parsed.get(f"Agent_{aid}") or parsed.get(aid)
                if not entry or not isinstance(entry, dict):
                    missing_ids.append(aid)
                    results[aid] = _make_fallback(f"agent {aid} missing from batch response")
                    continue
                action_raw = entry.get("a") or entry.get("action", "")
                action = ACTION_MAP.get(action_raw, "")
                choice = entry.get("c") or entry.get("choice", "")
                if action in ("vote", "delegate", "abstain"):
                    results[aid] = {"action": action, "choice": choice,
                                    "reason": entry.get("reason", entry.get("r", ""))}
                else:
                    bad_action_ids.append((aid, action_raw))
                    results[aid] = _make_fallback(
                        f"agent {aid} bad action '{action_raw}' (entry: {entry})")

            valid = sum(1 for r in results.values() if r.get("reason", "").startswith("llm_error") is False and r.get("action") != "abstain" or r.get("reason", "") not in ("llm_error",))
            n_fallback = len(missing_ids) + len(bad_action_ids)
            _log_debug("BATCH PARSED",
                       f"{len(results)} agents, {n_fallback} fallbacks"
                       + (f" (missing: {missing_ids[:5]})" if missing_ids else "")
                       + (f" (bad action: {bad_action_ids[:5]})" if bad_action_ids else ""))
            return results

        except subprocess.TimeoutExpired:
            reason = f"batch timeout after {timeout}s (attempt {attempt+1}/{max_retries})"
            _log_debug("BATCH TIMEOUT", reason)
            _log_error(reason)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {aid: _make_fallback(reason) for aid in agent_ids}
        except json.JSONDecodeError as e:
            reason = f"JSON parse error: {e} — raw output: {output[:300]}"
            _log_debug("BATCH PARSE ERROR", reason)
            _log_error(reason)
            return {aid: _make_fallback(reason) for aid in agent_ids}
        except OSError as e:
            reason = f"OS error calling claude: {e}"
            _log_error(reason)
            return {aid: _make_fallback(reason) for aid in agent_ids}

    return {aid: _make_fallback(f"exhausted {max_retries} retries") for aid in agent_ids}


def run_llm_decisions_batch_prompt(
    llm_agents: list[VoterAgent],
    race_id: str,
    contexts: dict[int, LLMContext],
    candidates: list[str],
    batch_size: int = BATCH_SIZE,
    batch_pause: float = 1.0,
    progress_fn: Any = None,
) -> dict[int, dict[str, Any]]:
    """Run LLM decisions using batched prompts with parallel execution.

    Sends batches of `batch_size` agents per Claude call, running up to
    `max_parallel` batches concurrently. 10K agents at 500/batch = 20 calls,
    10 in parallel = 2 rounds = ~30 seconds.
    """
    results: dict[int, dict[str, Any]] = {}
    total = len(llm_agents)
    max_parallel = DEFAULT_WORKERS  # 10 concurrent batch calls

    # Build all batches upfront
    batches: list[tuple[list[VoterAgent], list[int], str]] = []
    for batch_start in range(0, total, batch_size):
        batch = llm_agents[batch_start : batch_start + batch_size]
        batch_ids = [a.agent_id for a in batch]
        prompt = build_batch_prompt(batch, race_id, contexts, candidates)
        batches.append((batch, batch_ids, prompt))

    _log_debug("BATCH PLAN", f"{len(batches)} batches of up to {batch_size}, "
               f"{max_parallel} parallel workers, {total} agents total")

    # Execute batches in parallel
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(_call_claude_batch, prompt, batch_ids): (batch, batch_ids)
            for batch, batch_ids, prompt in batches
        }
        for future in as_completed(futures):
            batch, batch_ids = futures[future]
            try:
                batch_results = future.result()
            except Exception:
                batch_results = {aid: FALLBACK_RESULT for aid in batch_ids}
            results.update(batch_results)

            completed_count += len(batch)
            if progress_fn:
                progress_fn(min(completed_count, total), total)

    return results


def parse_llm_result(
    result: dict[str, Any], race_id: str
) -> Optional[tuple[VoteAction, Optional[str]]]:
    """Convert LLM JSON response to (VoteAction, choice)."""
    action_str = result.get("action", "abstain")
    choice = result.get("choice")

    action_map = {
        "vote": VoteAction.CAST_VOTE,
        "delegate": VoteAction.DELEGATE,
        "abstain": VoteAction.ABSTAIN,
    }
    action = action_map.get(action_str, VoteAction.ABSTAIN)
    return action, choice
