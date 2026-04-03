"""LLM-powered analysis of simulation results via Claude Code.

Generates a natural-language summary of the simulation run.
The prompt is intentionally compact — only parameters + summary stats,
no raw data arrays or per-agent details.

The prompt sent to Claude is visible in the dashboard via the
"Show prompt" expander under the AI Analysis section.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

TIMEOUT_SECONDS = 120


def build_analysis_prompt(results: dict, config: dict) -> str:
    """Build a prompt with full per-system results for each race.

    Sends the complete tally output from each system so Claude can
    reference specific vote counts, RCV rounds, TRS thresholds, and
    delegation chain metrics in its analysis.
    """
    fptp = results.get("fptp_results", {})
    rcv = results.get("rcv_results", {})
    trs = results.get("trs_results", {})
    deleg = results.get("delegation_results", {})
    stats = results.get("delegation_stats", {})

    # Build detailed per-race, per-system blocks
    race_blocks = []
    for race_id in fptp:
        fptp_r = fptp[race_id]
        rcv_r = rcv.get(race_id, {})
        trs_r = trs.get(race_id, {})
        deleg_r = deleg.get(race_id, {})

        # Round floats for readability
        deleg_wc = deleg_r.get("weighted_counts", {})
        deleg_rounded = {k: round(v, 1) for k, v in deleg_wc.items()} if deleg_wc else {}

        gini = stats.get("gini_per_race", {}).get(race_id, 0)
        max_w = stats.get("max_weight_per_race", {}).get(race_id, 0)
        chains = stats.get("chain_count_per_race", {}).get(race_id, 0)
        avg_len = stats.get("avg_chain_length_per_race", {}).get(race_id, 0)

        # RCV: show round-by-round counts
        rcv_rounds = rcv_r.get("rounds", [])
        rcv_eliminated = rcv_r.get("eliminated", [])
        rcv_detail = f"winner={rcv_r.get('winner','?')}"
        if rcv_rounds:
            rcv_detail += f", rounds={rcv_rounds}"
        if rcv_eliminated:
            rcv_detail += f", eliminated={rcv_eliminated}"

        # TRS: show both rounds
        trs_detail = f"winner={trs_r.get('winner','?')}, decided_in_round={trs_r.get('decided_in','?')}"
        r1 = trs_r.get("round1_counts")
        r2 = trs_r.get("round2_counts")
        if r1:
            trs_detail += f", round1={r1}"
        if r2:
            trs_detail += f", round2={r2}"

        block = (
            f"--- {race_id.upper()} ---\n"
            f"  US System (FPTP): winner={fptp_r.get('winner','?')}, "
            f"counts={fptp_r.get('counts',{})}, total={fptp_r.get('total_votes','?')}\n"
            f"  Ranked Choice (RCV): {rcv_detail}\n"
            f"  Two-Round (TRS): {trs_detail}\n"
            f"  Liquid Delegation: winner={deleg_r.get('winner','?')}, "
            f"weighted_counts={deleg_rounded}, "
            f"effective_votes={round(deleg_r.get('total_effective_votes', 0), 1)}\n"
            f"  Delegation metrics: Gini={gini:.3f}, max_weight={max_w:.1f}, "
            f"chains={chains}, avg_chain_length={avg_len:.1f}"
        )
        race_blocks.append(block)

    total_d = stats.get("total_delegators", 0)
    total_v = stats.get("total_direct_voters", 0)
    total_a = stats.get("total_abstentions", 0)

    pvi = config.get("pvi_lean", 0)
    pvi_label = f"D+{abs(pvi):.0f}" if pvi < 0 else f"R+{pvi:.0f}" if pvi > 0 else "EVEN"
    district = config.get("district_id") or "synthetic"
    data_mode = "CES survey" if config.get("use_ces") else ("district profile" if config.get("district_id") else "synthetic")

    results_block = "\n".join(race_blocks)

    use_llm = config.get("use_llm", False)
    llm_frac = config.get("llm_agent_fraction", 0.05)
    n_agents = config.get("n_agents", "?")
    bc_eps = config.get("bounded_confidence_epsilon")
    bandwagon = config.get("bandwagon_coefficient", 0.25)

    races = config.get("races", {})
    race_summary = ", ".join(
        f"{rid}: {len(cands)} candidates ({', '.join(cands)})"
        for rid, cands in races.items()
    )

    return f"""Analyze this liquid democracy simulation (4 voting systems on same electorate).

SIMULATION PARAMETERS (with explanations):
- Races: {race_summary}
- Agents: {n_agents} ({data_mode} seeding from {district})
- PVI (Partisan Voter Index): {pvi_label} — how much this district leans D or R vs national average
- Delegation probability: {config.get('delegation_probability_base',0.10):.0%} — fraction of agents who consider delegating each campaign tick
- Viscous decay (alpha): {config.get('viscous_decay_alpha',0.85)} — weight retained per delegation hop (0.85^3 = 61% at 3 hops)
- Weight cap: {config.get('weight_cap','none')} — max votes any single delegate can carry (none = unlimited)
- Delegation options (k): {config.get('delegation_options_k',1)} — how many delegates each voter can split across (k=2 splits vote)
- Preferential attachment (gamma): {config.get('preferential_attachment_gamma',1.5)} — how strongly popular delegates attract more (rich-get-richer exponent)
- Bandwagon coefficient: {bandwagon} — herding effect strength (0.25 = 25% of agents follow the crowd)
- Bounded confidence (epsilon): {bc_eps if bc_eps else 'off'} — agents only listen to peers within this ideological distance (lower = echo chambers)
- Political homophily: {config.get('homophily', 0.65)} — fraction of network ties rewired to same-party (0.65 = most neighbors share your party; lower = more cross-party ties and delegation)
- LLM agents: {'enabled (' + str(round(llm_frac*100)) + '% of agents use Claude for decisions)' if use_llm else 'disabled (all agents use rule-based decisions)'}

FULL RESULTS BY RACE AND SYSTEM:
{results_block}

PARTICIPATION: {total_v} direct voters, {total_d} delegated, {total_a} abstained

Write the analysis using these markdown section headers (##) so they render properly:

#### System Agreement
State how many candidates ran (e.g. "In this 2-candidate race..." or "With 4 candidates on the ballot..."). \
Which systems agreed/disagreed on winners and why. Reference specific vote counts. \
Note: with only 2 candidates, RCV and TRS should always match FPTP — if they diverge, explain why.

#### Power Concentration
Interpret the Gini coefficient and max delegate weight. Is power dangerously concentrated? \
Compare to real-world benchmarks (LiquidFeedback Gini was 0.64-0.99).

#### Delegation Impact
Did delegation change the outcome vs the US system (FPTP)? What does that mean — did the \
"wisdom of delegation" improve representation, or did concentrated power distort it?

#### Electoral Reform Takeaway
One key insight for someone thinking about electoral system design.

#### Try Next
Suggest 2-3 specific parameter changes to try that would likely change the results. \
Be concrete with values (e.g., "increase delegation probability to 60%").

IMPORTANT: After each suggestion, on its own line, emit a machine-readable params line \
in exactly this format (JSON dict of parameter overrides to apply):
<!-- PARAMS: {{"key": value, ...}} -->

The valid parameter keys and their types are:
- "n_agents": int (100-10000) — number of voters in the simulation
- "pvi_lean": float (-30 to 30) — partisan lean (negative=D, positive=R)
- "delegation_probability_base": float (0.0-1.0) — chance each agent considers delegating per tick
- "viscous_decay_alpha": float (0.0-1.0) — weight retained per delegation hop
- "weight_cap": float or null — max votes any single delegate can carry
- "delegation_options_k": int (1-5) — how many delegates each voter can split across
- "preferential_attachment_gamma": float (0.0-3.0) — rich-get-richer exponent for popular delegates
- "bandwagon_coefficient": float (0.0-1.0) — herding effect strength
- "bounded_confidence_epsilon": float or null — agents only listen to peers within this ideology distance
- "homophily": float (0.0-1.0) — fraction of network ties rewired to same-party
- "llm_agent_fraction": float (0.01-0.50) — fraction of agents that use Claude for decisions
- "use_media": bool — enable/disable media influence during campaign
- "media_bias_factor": float (0.5-2.0) — amplifies or dampens media influence strength
- "races": dict — candidate list, e.g. {{"race": ["Democrat", "Republican", "Libertarian"]}}

Example:
1. **Raise delegation probability to 40%** — more delegation means higher Gini.
<!-- PARAMS: {{"delegation_probability_base": 0.40}} -->

2. **Add a third-party candidate** — spoiler effects make RCV and TRS diverge more from FPTP.
<!-- PARAMS: {{"races": {{"race": ["Democrat", "Republican", "Libertarian"]}}}} -->

Always expand acronyms on first use: FPTP (First Past The Post), RCV (Ranked Choice Voting), \
TRS (Two-Round System), PVI (Partisan Voter Index), CES (Cooperative Election Study).

Use specific numbers from the results. Keep each section to 2-4 sentences."""


def analyze_results(
    results: dict,
    config: dict,
    timeout: int = TIMEOUT_SECONDS,
    status_callback: Any | None = None,
) -> str | None:
    """Call Claude Code to generate an analysis summary.

    *status_callback*, when provided, is called with a short string whenever
    stderr output is available (e.g. rate-limit notices, connection info).
    This lets the dashboard display live progress.

    Returns the analysis text, or an error message on failure.
    """
    import select
    import time

    prompt = build_analysis_prompt(results, config)

    debug = os.environ.get("LLM_DEBUG", "").lower() in ("1", "true", "yes")
    if debug:
        import sys
        print("=" * 60, file=sys.stderr)
        print("[AI SUMMARY DEBUG] PROMPT:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    try:
        proc = subprocess.Popen(
            ["claude", "-p", "--model", "sonnet"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Send prompt and close stdin
        proc.stdin.write(prompt)
        proc.stdin.close()

        t0 = time.monotonic()
        stderr_lines: list[str] = []
        stdout_chunks: list[str] = []

        while proc.poll() is None:
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                proc.kill()
                return f"_Claude timed out after {timeout}s. This can happen on first run (cold start). Try again._"

            # Non-blocking read of stderr for status updates
            ready, _, _ = select.select([proc.stderr], [], [], 0.5)
            if ready:
                line = proc.stderr.readline()
                if line:
                    stripped = line.strip()
                    stderr_lines.append(stripped)
                    if status_callback and stripped:
                        # Surface rate-limit or connection info
                        status_callback(stripped, elapsed)

            if status_callback:
                # Periodic heartbeat even without stderr
                status_callback(None, elapsed)

        # Read remaining output (communicate() drains and closes both pipes)
        remaining_out, remaining_err = proc.communicate(timeout=5)
        if remaining_out:
            stdout_chunks.append(remaining_out)
        if remaining_err:
            for line in remaining_err.strip().splitlines():
                stderr_lines.append(line.strip())

        stdout = "".join(stdout_chunks).strip()
        stderr_full = "\n".join(stderr_lines)

        if proc.returncode != 0:
            msg = stderr_full[:300] if stderr_full else "(no stderr)"
            return f"_Claude exited with code {proc.returncode}: {msg}_"

        if not stdout or len(stdout) < 50:
            msg = stderr_full[:200] if stderr_full else ""
            return f"_Claude returned an empty/short response. {msg}_"

        if debug:
            print("=" * 60, file=sys.stderr)
            print("[AI SUMMARY DEBUG] RESPONSE:", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print(stdout[:2000], file=sys.stderr)
            print("=" * 60, file=sys.stderr)

        return stdout

    except FileNotFoundError:
        return "_`claude` CLI not found. Install with `npm install -g @anthropic-ai/claude-code` to enable AI summaries._"
    except OSError as e:
        return f"_Failed to run claude: {e}_"


# ---------------------------------------------------------------------------
# Per-graph AI summaries
# ---------------------------------------------------------------------------


def build_graph_summaries_prompt(graph_data: dict[str, str]) -> str:
    """Build a prompt asking Claude for one-sentence summaries per graph.

    *graph_data* maps a chart key (e.g. ``"election_outcomes"``) to a compact
    description of the chart type **plus** the key data points it displays.
    """
    sections = "\n\n".join(f"[{key}]\n{desc}" for key, desc in graph_data.items())

    return (
        "You are analyzing charts from a liquid democracy simulation dashboard. "
        "For each chart below, write exactly ONE sentence summarizing the most "
        "notable pattern or finding in this specific data. Be concrete — cite "
        "numbers from the data. Write in present tense. Do not repeat the chart "
        "title.\n\n"
        "Format: one line per chart, starting with the label in square brackets.\n"
        "Example:\n"
        "[example] The Democrat wins in all four systems with 58% of the FPTP "
        "vote, suggesting strong partisan lean overrides system differences.\n\n"
        f"{sections}"
    )


def generate_graph_summaries(
    graph_data: dict[str, str], timeout: int = 60,
) -> dict[str, str]:
    """Call Claude to generate per-graph one-sentence summaries.

    Returns a dict mapping chart key -> summary string.  On any failure
    returns an empty dict so the dashboard degrades gracefully.
    """
    prompt = build_graph_summaries_prompt(graph_data)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        return _parse_graph_summaries(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}


def _parse_graph_summaries(text: str) -> dict[str, str]:
    """Parse ``[key] summary`` lines from Claude output."""
    import re

    summaries: dict[str, str] = {}
    for match in re.finditer(r"\[(\w+)\]\s*(.+?)(?=\n\[|\Z)", text, re.DOTALL):
        key = match.group(1)
        summary = " ".join(match.group(2).split())  # collapse whitespace
        if summary:
            summaries[key] = summary
    return summaries
