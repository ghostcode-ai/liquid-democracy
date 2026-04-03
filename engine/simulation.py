"""Mesa-style simulation controller for the liquid democracy simulation.

Orchestrates three temporal phases across ~91 ticks:
  Phase A: Campaign (ticks 1-80)  -- opinion formation, trust, delegation discovery
  Phase B: Voting  (ticks 81-90)  -- agents commit decisions, delegations registered
  Phase C: Tally   (tick 91)      -- all four tally engines run on the same data

Uses plain Python with tick-based stepping (no mesa.Model inheritance).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from agents.voter_agent import ActionResult, RaceState, VoteAction, VoterAgent
from engine.delegation_graph import DelegationGraph
from engine.opinion_dynamics import (
    build_influence_matrix,
    build_stubbornness_matrix,
    extract_opinions,
    friedkin_johnsen_step,
)
from agents.llm_bridge import FALLBACK_RESULT, LLMContext, parse_llm_result, run_llm_decisions_batch_prompt
from agents.media_agent import MediaAgent, apply_media_cycle, create_default_media_environment
from engine.seeding import build_social_network, seed_agents
from engine.trust import detect_betrayals, update_all_trust
from tally.fptp import FPTPTally
from tally.rcv import RCVTally
from tally.trs import TRSTally


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """All parameters for a single simulation run."""
    n_agents: int = 10_000
    pvi_lean: float = 0.0
    ideology_std: float = 0.3
    seed: int = 42
    # Real-data mode: set district_id to seed from an actual 2024 district profile
    district_id: Optional[str] = None
    # CES mode: set use_ces=True + district_id to seed from actual survey respondents
    use_ces: bool = False
    races: dict[str, list[str]] = field(default_factory=lambda: {
        "race": ["Democrat", "Republican"],
    })
    campaign_ticks: int = 80
    voting_ticks: int = 1
    bounded_confidence_epsilon: Optional[float] = None
    # Delegation
    delegation_probability_base: float = 0.10
    knowledge_threshold: float = 0.6
    delegation_threshold: float = 0.5
    viscous_decay_alpha: float = 0.85
    weight_cap: Optional[float] = None
    delegation_options_k: int = 1
    # Trust
    trust_alpha: float = 0.7
    trust_beta: float = 0.15
    trust_gamma: float = 0.05
    trust_delta: float = 0.3
    # Bandwagon / preferential attachment
    bandwagon_coefficient: float = 0.25
    preferential_attachment_gamma: float = 1.5
    # Social network
    homophily: float = 0.65
    # LLM
    llm_agent_fraction: float = 0.05
    llm_max_workers: int = 100
    use_llm: bool = False
    # Media
    use_media: bool = False
    media_bias_factor: float = 1.0  # 1.0=fair, 1.5=slanted, 2.0=radical
    # TRS
    trs_withdrawal_prob: float = 0.6
    # Roll-off rates
    roll_off_rates: dict[str, float] = field(default_factory=lambda: {
        "senate": 0.01, "house": 0.03, "state_legislature": 0.15,
        "judicial": 0.23, "proposition": 0.20, "local": 0.31,
    })

    @property
    def total_ticks(self) -> int:
        return self.campaign_ticks + self.voting_ticks + 1  # +1 for tally


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SimulationResults:
    """Complete output of one simulation run."""
    fptp_results: dict[str, dict]
    rcv_results: dict[str, dict]
    trs_results: dict[str, dict]
    delegation_results: dict[str, dict]
    delegation_stats: dict
    opinion_history: dict[int, list]
    config: SimulationConfig


# Candidate ideological positions for RCV ballot generation
_CANDIDATE_IDEOLOGIES: dict[str, float] = {
    "Democrat": -0.6, "Republican": 0.6, "Libertarian": 0.4,
    "Green": -0.8, "Independent": 0.0,
}


def _candidate_ideology(name: str) -> float:
    return _CANDIDATE_IDEOLOGIES.get(name, 0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LiquidDemocracyModel:
    """Tick-based simulation model for liquid democracy experiments."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.tick = 0
        self.rng = np.random.default_rng(config.seed)
        random.seed(config.seed)
        # Optional callback: progress_callback(message: str) for live status updates
        self.progress_callback: Any = None

        # Agents: three seeding modes
        if config.use_ces and config.district_id:
            # CES mode: seed from actual 2024 survey respondents
            from data.ces_loader import seed_agents_from_ces
            self.agents: dict[int, VoterAgent] = seed_agents_from_ces(
                district_id=config.district_id,
                n_agents=config.n_agents,
                seed=config.seed,
            )
        elif config.district_id:
            # District profile mode: seed from curated 2024 district data
            from engine.seeding import seed_from_district
            self.agents: dict[int, VoterAgent] = seed_from_district(
                district_id=config.district_id,
                n_agents=config.n_agents,
                ideology_std=config.ideology_std,
                seed=config.seed,
            )
        else:
            # Synthetic mode: parametric distributions
            self.agents: dict[int, VoterAgent] = seed_agents(
                n_agents=config.n_agents, pvi_lean=config.pvi_lean,
                ideology_std=config.ideology_std, seed=config.seed,
            )
        if config.use_llm:
            n_llm = max(1, int(config.n_agents * config.llm_agent_fraction))
            # Prefer moderates: agents closest to ideology 0 are most likely
            # to be LLM agents — partisans vote predictably either way,
            # but swing voters benefit most from genuine reasoning.
            agent_ids = list(self.agents.keys())
            centrist_scores = np.array([
                1.0 / (1.0 + abs(self.agents[aid].ideology[0])) for aid in agent_ids
            ])
            centrist_probs = centrist_scores / centrist_scores.sum()
            llm_ids = self.rng.choice(
                agent_ids, size=min(n_llm, len(agent_ids)),
                replace=False, p=centrist_probs,
            )
            for aid in llm_ids:
                self.agents[aid].is_llm_agent = True
        self.social_graph: nx.Graph = build_social_network(
            self.agents, homophily=config.homophily, rng=self.rng,
        )
        self._init_neighbor_trust()
        self.delegation_graph = DelegationGraph()
        self._init_race_states()

        # Media agents
        self._media_agents: list[MediaAgent] = []
        if config.use_media:
            self._media_agents = create_default_media_environment(seed=config.seed)
            if config.media_bias_factor != 1.0:
                for ma in self._media_agents:
                    ma.ideology = np.clip(ma.ideology * config.media_bias_factor, -1.0, 1.0)

        # Opinion dynamics matrices (dimension 0 = economy)
        self._agent_ids_sorted = sorted(self.agents.keys())
        self._x_initial = extract_opinions(self.agents, dimension=0)
        self._x_current = self._x_initial.copy()
        self._Lambda = build_stubbornness_matrix(self.agents)
        self._W = build_influence_matrix(
            self.agents, self.social_graph, dimension=0,
            epsilon=config.bounded_confidence_epsilon,
        )

        # Data collection
        self._opinion_snapshots: list[np.ndarray] = [self._x_current.copy()]
        self._opinion_deltas: list[float] = []  # mean absolute change per tick
        self._convergence_tick: int | None = None  # first tick where delta < threshold
        self._gini_history: list[dict[str, float]] = []
        self._trust_snapshots: list[float] = []
        self._betrayal_events: dict[tuple[int, int], float] = {}
        self._influence_attribution: list[dict[str, float]] = []  # per-campaign-tick: peer vs media shift

        # Track initial party/ideology for cross-party shift detection
        self._initial_parties: dict[int, str] = {
            aid: a.party_id.value for aid, a in self.agents.items()
        }
        self._initial_ideologies: dict[int, float] = {
            aid: float(a.ideology[0]) for aid, a in self.agents.items()
        }

        # Per-race voting records (populated during voting phase)
        race_ids = list(config.races.keys())
        self._direct_votes: dict[str, dict[int, str]] = {r: {} for r in race_ids}
        self._ranked_ballots: dict[str, list[list[str]]] = {r: [] for r in race_ids}
        self._abstentions: dict[str, list[tuple[int, str]]] = {r: [] for r in race_ids}
        self._delegation_records: dict[str, list[tuple[int, int]]] = {r: [] for r in race_ids}
        self._has_acted: dict[str, set[int]] = {r: set() for r in race_ids}
        self._delegation_cooldown: dict[int, int] = {}

        # LLM vs rule-based decision tracking
        self._llm_decisions: dict[str, list[tuple[int, str, str, str]]] = {r: [] for r in race_ids}  # (aid, action, choice, reason)
        self._rule_decisions: dict[str, list[tuple[int, str, str, str]]] = {r: [] for r in race_ids}

        self.results: Optional[SimulationResults] = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_neighbor_trust(self) -> None:
        """Seed initial trust between neighbors based on ideological agreement.

        Same-party neighbors start with trust ~0.5, cross-party ~0.2.
        This gives delegation discovery a realistic starting point.
        """
        from engine.trust import compute_agreement

        for aid, agent in self.agents.items():
            for nid in agent.neighbors:
                neighbor = self.agents.get(nid)
                if neighbor is None:
                    continue
                agreement = compute_agreement(agent, neighbor)
                # Initial trust: base 0.3, boosted by agreement
                initial_trust = 0.3 + 0.4 * agreement
                agent.trust_scores[nid] = initial_trust

    def rebuild_opinion_matrices(self) -> None:
        """Rebuild opinion dynamics matrices after agents are added/removed.

        Call this after injecting new agents (e.g. celebrity scenario).
        """
        self._agent_ids_sorted = sorted(self.agents.keys())
        self._id_to_idx = {aid: i for i, aid in enumerate(self._agent_ids_sorted)}
        self._x_initial = extract_opinions(self.agents, dimension=0)
        self._x_current = self._x_initial.copy()
        self._Lambda = build_stubbornness_matrix(self.agents)
        self._W = build_influence_matrix(
            self.agents, self.social_graph, dimension=0,
            epsilon=self.config.bounded_confidence_epsilon,
        )

    def _init_race_states(self) -> None:
        """Create per-race state for every agent."""
        for agent in self.agents.values():
            for race_id in self.config.races:
                knowledge = agent.political_knowledge
                knowledge = float(np.clip(knowledge + self.rng.normal(0, 0.05), 0.0, 1.0))
                agent.race_states[race_id] = RaceState(
                    knowledge_level=knowledge, preference=float(agent.ideology[0]),
                )

    # ------------------------------------------------------------------
    # Phase routing
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one tick."""
        self.tick += 1
        if self.tick <= self.config.campaign_ticks:
            self._campaign_step()
        elif self.tick <= self.config.campaign_ticks + self.config.voting_ticks:
            self._voting_step()
        elif self.tick == self.config.total_ticks:
            self._tally()

    def run(self, n_ticks: Optional[int] = None) -> SimulationResults:
        """Run the full simulation and return results."""
        for _ in range(n_ticks or self.config.total_ticks):
            self.step()
        return self.get_results()

    # ------------------------------------------------------------------
    # Phase A: Campaign
    # ------------------------------------------------------------------

    def _campaign_step(self) -> None:
        """One campaign tick: opinion dynamics, trust, delegation discovery."""
        # Measure influence sources: snapshot before each step
        pre_fj = self._x_current.copy()

        # Friedkin-Johnsen on dimension 0
        self._x_current = np.clip(
            friedkin_johnsen_step(self._x_current, self._x_initial, self._Lambda, self._W),
            -1.0, 1.0,
        )
        for i, aid in enumerate(self._agent_ids_sorted):
            self.agents[aid].ideology[0] = self._x_current[i]
            for state in self.agents[aid].race_states.values():
                state.preference = float(self._x_current[i])

        peer_shift = float(np.mean(np.abs(self._x_current - pre_fj)))

        # Media influence (shifts ideology vectors, then sync _x_current)
        media_shift = 0.0
        if self._media_agents:
            pre_media = self._x_current.copy()
            apply_media_cycle(
                self._media_agents, self.agents, ad_campaigns=[],
                current_tick=self.tick, rng=self.rng,
            )
            # Sync _x_current from agent ideology after media shifts
            for i, aid in enumerate(self._agent_ids_sorted):
                self._x_current[i] = self.agents[aid].ideology[0]
            media_shift = float(np.mean(np.abs(self._x_current - pre_media)))

        self._influence_attribution.append({
            "peer": round(peer_shift, 6),
            "media": round(media_shift, 6),
            "total": round(peer_shift + media_shift, 6),
        })

        self._opinion_snapshots.append(self._x_current.copy())

        # Track convergence: mean absolute opinion change this tick
        if len(self._opinion_snapshots) >= 2:
            prev = self._opinion_snapshots[-2]
            delta = float(np.mean(np.abs(self._x_current - prev)))
            self._opinion_deltas.append(delta)
            if self._convergence_tick is None and delta < 0.001:
                self._convergence_tick = self.tick

        # Trust update every 10 ticks
        if self.tick % 10 == 0:
            update_all_trust(
                self.agents, self.social_graph, betrayal_events=self._betrayal_events,
                alpha=self.config.trust_alpha, beta=self.config.trust_beta,
                gamma=self.config.trust_gamma, delta=self.config.trust_delta,
            )
            self._W = build_influence_matrix(
                self.agents, self.social_graph, dimension=0,
                epsilon=self.config.bounded_confidence_epsilon,
            )
            all_t = [t for a in self.agents.values() for t in a.trust_scores.values()]
            self._trust_snapshots.append(float(np.mean(all_t)) if all_t else 0.0)

        self._delegation_discovery()
        self._knowledge_growth()
        self._record_gini_snapshot()

    def _record_gini_snapshot(self) -> None:
        """Record Gini coefficient for each race at the current tick."""
        self._gini_history.append(
            {r: self.delegation_graph.get_gini(r) for r in self.config.races}
        )

    def _delegation_discovery(self) -> None:
        """Agents consider delegating based on trust and preferential attachment."""
        cfg = self.config
        for aid, agent in self.agents.items():
            if self._delegation_cooldown.get(aid, 0) > self.tick:
                continue
            if self.rng.random() > cfg.delegation_probability_base:
                continue
            if not any(s.knowledge_level < cfg.knowledge_threshold for s in agent.race_states.values()):
                continue
            # Neighbors with sufficient trust
            trusted = [
                (nid, agent.trust_scores.get(nid, 0.0))
                for nid in agent.neighbors
                if agent.trust_scores.get(nid, 0.0) > cfg.delegation_threshold
            ]
            if not trusted:
                continue
            # Preferential attachment + bandwagon:
            # P(j) ~ trust * (1 + delegators(j))^gamma * (1 + bandwagon * delegators(j))
            race_ids_needing = [
                r for r, s in agent.race_states.items()
                if s.knowledge_level < cfg.knowledge_threshold
            ]
            if not race_ids_needing:
                continue
            sample_race = race_ids_needing[0]
            weights = []
            for nid, tv in trusted:
                n_del = len(self.delegation_graph.get_delegators(str(nid), sample_race))
                pa = (1 + n_del) ** cfg.preferential_attachment_gamma
                bw = 1.0 + cfg.bandwagon_coefficient * n_del
                weights.append(tv * pa * bw)
            total_w = sum(weights)
            if total_w < 1e-12:
                continue
            k = min(cfg.delegation_options_k, len(trusted))
            probs = [w / total_w for w in weights]
            chosen_indices = self.rng.choice(len(trusted), size=k, replace=False, p=probs)
            chosen_nids = [trusted[i][0] for i in chosen_indices]
            for race_id in race_ids_needing:
                agent.race_states[race_id].delegation_targets = chosen_nids
                # Register provisional delegation edges so Gini tracks during campaign
                for nid in chosen_nids:
                    self.delegation_graph.add_delegation(str(aid), str(nid), race_id)
            self._delegation_cooldown[aid] = self.tick + int(self.rng.integers(5, 16))

    def _knowledge_growth(self) -> None:
        """Agents slowly learn about races during the campaign."""
        for agent in self.agents.values():
            for state in agent.race_states.values():
                growth = 0.005 * agent.engagement_level * (1.0 - state.knowledge_level)
                state.knowledge_level = float(
                    np.clip(state.knowledge_level + growth + self.rng.normal(0, 0.002), 0.0, 1.0)
                )

    # ------------------------------------------------------------------
    # Phase B: Voting
    # ------------------------------------------------------------------

    def _voting_step(self) -> None:
        """One voting tick: agents commit decisions across all races."""
        cfg = self.config
        fraction = (self.tick - cfg.campaign_ticks) / cfg.voting_ticks  # ramp 0->1

        # Determine which agents are eligible to act this tick
        eligible: list[tuple[int, VoterAgent]] = []
        for aid, agent in self.agents.items():
            # Skip if already acted in all races
            if all(aid in self._has_acted[r] for r in cfg.races):
                continue
            # Engagement ramp: high-engagement agents act earlier
            if self.rng.random() < (1.0 - agent.engagement_level) * (1.0 - fraction):
                continue
            eligible.append((aid, agent))

        # Separate LLM agents from rule-based agents
        llm_eligible = [(aid, agent) for aid, agent in eligible if cfg.use_llm and agent.is_llm_agent]
        rule_eligible = [(aid, agent) for aid, agent in eligible if not (cfg.use_llm and agent.is_llm_agent)]

        # Rule-based agents: decide inline
        cb = self.progress_callback
        vote_tick = self.tick - cfg.campaign_ticks
        if cb and rule_eligible:
            cb(f"Rule-based voters deciding ({len(rule_eligible):,})...")
        for aid, agent in rule_eligible:
            for race_id, candidates in cfg.races.items():
                if aid in self._has_acted[race_id]:
                    continue
                result = agent.decide_action(
                    race_id=race_id, race_type=race_id,
                    knowledge_threshold=cfg.knowledge_threshold,
                    delegation_threshold=cfg.delegation_threshold,
                    candidates=candidates, pvi_lean=cfg.pvi_lean,
                    rng=self.rng,
                )
                self._record_action(aid, agent, race_id, candidates, result)
                self._has_acted[race_id].add(aid)

        # LLM agents: batch per race, run in parallel via thread pool
        if llm_eligible:
            batch_size = 100
            for race_id, candidates in cfg.races.items():
                batch_agents = [
                    agent for aid, agent in llm_eligible
                    if aid not in self._has_acted[race_id]
                ]
                if not batch_agents:
                    continue
                if cb:
                    cb(f"Free-thinking agents deciding ({len(batch_agents)} voters)...")
                contexts = {
                    agent.agent_id: self._build_llm_context(agent, race_id, candidates)
                    for agent in batch_agents
                }

                def _llm_progress(completed: int, total: int) -> None:
                    if cb:
                        cb(f"Free-thinking agents deciding ({completed}/{total})...")

                raw_results = run_llm_decisions_batch_prompt(
                    batch_agents, race_id, contexts,
                    candidates=candidates,
                    batch_size=20,
                    batch_pause=1.0,
                    progress_fn=_llm_progress,
                )

                for aid, agent in llm_eligible:
                    if aid not in raw_results:
                        continue
                    raw = raw_results[aid]
                    action, choice = parse_llm_result(raw, race_id)
                    result = ActionResult(
                        action=action,
                        race_id=race_id,
                        choice=choice,
                        reason=raw.get("reason", "llm_decision"),
                    )
                    self._record_action(aid, agent, race_id, candidates, result)
                    self._has_acted[race_id].add(aid)

        self._record_gini_snapshot()

    def _build_llm_context(
        self, agent: VoterAgent, race_id: str, candidates: list[str],
    ) -> LLMContext:
        """Build the context an LLM agent needs to make a voting decision."""
        trusted_opinions: dict[str, str] = {}
        for nid in agent.neighbors:
            neighbor = self.agents.get(nid)
            if neighbor is None:
                continue
            trust = agent.trust_scores.get(nid, 0.0)
            if trust <= 0.0:
                continue
            ideo = neighbor.ideology[0]
            party = neighbor.party_id.value
            trusted_opinions[f"Agent_{nid}"] = (
                f"{party}, ideology {ideo:+.2f}, trust {trust:.2f}"
            )
        delegator_count = len(
            self.delegation_graph.get_delegators(str(agent.agent_id), race_id)
        )
        return LLMContext(
            race_description=f"US congressional {race_id} election",
            candidates=candidates,
            trusted_opinions=trusted_opinions,
            delegator_count=delegator_count,
        )

    def _record_action(
        self, aid: int, agent: VoterAgent, race_id: str,
        candidates: list[str], result: ActionResult,
    ) -> None:
        """Record a single agent's voting decision."""
        # Track LLM vs rule-based (4-tuple: aid, action, choice, reason)
        tracker = self._llm_decisions if agent.is_llm_agent else self._rule_decisions
        tracker[race_id].append((aid, result.action.value, result.choice or "", result.reason))

        rs = agent.race_states[race_id]
        if result.action == VoteAction.CAST_VOTE and result.choice:
            self._direct_votes[race_id][aid] = result.choice
            rs.voted, rs.vote_choice = True, result.choice
            ranked = self._generate_ranked_ballot(agent, candidates, first_choice=result.choice)
            self._ranked_ballots[race_id].append(ranked)
            rs.ranked_choices = ranked
        elif result.action == VoteAction.DELEGATE:
            k = self.config.delegation_options_k
            delegates = rs.delegation_targets or agent.find_best_k_delegates(race_id, k=k)
            if delegates:
                fraction = 1.0 / len(delegates)
                for delegate_id in delegates:
                    self.delegation_graph.add_delegation(
                        str(aid), str(delegate_id), race_id, fraction=fraction,
                    )
                    self._delegation_records[race_id].append((aid, delegate_id))
                rs.delegation_targets = list(delegates)
            else:
                self._abstentions[race_id].append((aid, "no_delegate_found"))
        elif result.action == VoteAction.ABSTAIN:
            self._abstentions[race_id].append((aid, result.reason))

    def _generate_ranked_ballot(
        self, agent: VoterAgent, candidates: list[str], first_choice: str | None = None,
    ) -> list[str]:
        """Rank candidates by ideology distance with knowledge-based noise.

        *first_choice*, when provided, is pinned at position 1 so the ranked
        ballot is consistent with the agent's direct vote.  Remaining
        candidates are ranked by ideology distance + noise.
        """
        ideo = float(agent.ideology[0])
        ns = 0.3 * (1.0 - agent.political_knowledge)
        remaining = [c for c in candidates if c != first_choice]
        scored = sorted(
            ((abs(ideo - _candidate_ideology(c)) + float(self.rng.normal(0, ns)), c) for c in remaining),
            key=lambda p: p[0],
        )
        ranked = [name for _, name in scored]
        if first_choice:
            ranked.insert(0, first_choice)
        return ranked

    # ------------------------------------------------------------------
    # Phase C: Tally
    # ------------------------------------------------------------------

    def _tally(self) -> None:
        """Run all four tally engines and collect results."""
        cfg = self.config
        for race_id in cfg.races:
            betrayals = detect_betrayals(self.agents, race_id, self._delegation_records[race_id])
            self._betrayal_events.update(betrayals)

        fptp_eng, rcv_eng, trs_eng = FPTPTally(), RCVTally(), TRSTally()
        fptp_r: dict[str, dict] = {}
        rcv_r: dict[str, dict] = {}
        trs_r: dict[str, dict] = {}
        del_r: dict[str, dict] = {}
        for race_id, candidates in cfg.races.items():
            fptp_votes = self._build_fptp_votes(race_id, candidates)
            fptp_r[race_id] = fptp_eng.tally(fptp_votes)
            rcv_r[race_id] = rcv_eng.tally(self._ranked_ballots[race_id])
            trs_r[race_id] = self._run_trs(trs_eng, race_id, candidates)
            del_r[race_id] = self._run_delegation_tally(race_id, candidates)

        self.results = SimulationResults(
            fptp_results=fptp_r, rcv_results=rcv_r,
            trs_results=trs_r, delegation_results=del_r,
            delegation_stats=self._compute_delegation_stats(),
            opinion_history={0: [s.tolist() for s in self._opinion_snapshots]},
            config=cfg,
        )

    def _build_fptp_votes(self, race_id: str, candidates: list[str]) -> dict[str, list[int]]:
        """Build {candidate: [voter_ids]} from direct votes."""
        votes: dict[str, list[int]] = {c: [] for c in candidates}
        for vid, choice in self._direct_votes[race_id].items():
            if choice in votes:
                votes[choice].append(vid)
        return votes

    def _run_trs(self, engine: TRSTally, race_id: str, candidates: list[str]) -> dict:
        """Run a full TRS election, generating round 2 votes if needed."""
        r1_votes = self._build_fptp_votes(race_id, candidates)
        def round2_vote_fn(r2_cands: list[str]) -> dict[str, list[int]]:
            r2: dict[str, list[int]] = {c: [] for c in r2_cands}
            for aid, agent in self.agents.items():
                st = agent.race_states.get(race_id)
                if st and st.voted:
                    r2[self._closest_candidate(agent, r2_cands)].append(aid)
            return r2
        return engine.full_election(
            round1_votes=r1_votes, registered_voters=self.config.n_agents,
            round2_vote_fn=round2_vote_fn, withdrawal_prob=self.config.trs_withdrawal_prob,
        )

    def _closest_candidate(self, agent: VoterAgent, candidates: list[str]) -> str:
        """Return the candidate closest to the agent's ideology with partisan anchoring."""
        choice = agent._preference_to_candidate(float(agent.ideology[0]), candidates)
        return choice or candidates[0]

    def _run_delegation_tally(self, race_id: str, candidates: list[str]) -> dict:
        """Resolve delegation chains and tally weighted votes.

        Uses DelegationGraph.resolve_all directly (the DelegationTallyEngine
        expects methods not yet present on the graph).
        """
        weights = self.delegation_graph.resolve_all(race_id, weight_cap=self.config.weight_cap)
        w_counts: dict[str, float] = {c: 0.0 for c in candidates}
        v_weights: list[float] = []
        # Accumulate weights for voters in delegation chains
        for voter_str, w in weights.items():
            if w <= 0:
                continue
            vid = int(voter_str)
            choice = self._direct_votes[race_id].get(vid)
            if choice is None:
                agent = self.agents.get(vid)
                if agent is None:
                    continue
                choice = self._closest_candidate(agent, candidates)
            w_counts[choice] = w_counts.get(choice, 0.0) + w
            v_weights.append(w)
        # Direct voters not in the delegation graph get weight 1.0
        for vid, choice in self._direct_votes[race_id].items():
            if str(vid) not in weights:
                w_counts[choice] = w_counts.get(choice, 0.0) + 1.0
                v_weights.append(1.0)

        total_eff = sum(w_counts.values())
        max_w = max(v_weights) if v_weights else 0.0
        winner = None
        if w_counts and total_eff > 0:
            mx = max(w_counts.values())
            winner = sorted(c for c, v in w_counts.items() if v == mx)[0]
        return {
            "winner": winner, "weighted_counts": w_counts, "gini": _gini(v_weights),
            "max_weight": max_w, "delegation_chains": len(self._delegation_records[race_id]),
            "total_effective_votes": total_eff,
        }

    # ------------------------------------------------------------------
    # Delegation stats
    # ------------------------------------------------------------------

    def _compute_delegation_stats(self) -> dict:
        """Aggregate delegation statistics across all races."""
        gini_pr: dict[str, float] = {}
        maxw_pr: dict[str, float] = {}
        cnt_pr: dict[str, int] = {}
        avglen_pr: dict[str, float] = {}
        for race_id in self.config.races:
            weights = self.delegation_graph.resolve_all(
                race_id, weight_cap=self.config.weight_cap,
            )
            gini_pr[race_id] = self.delegation_graph.get_gini(race_id)
            maxw_pr[race_id] = max(weights.values()) if weights else 0.0
            chains = self._delegation_records[race_id]
            cnt_pr[race_id] = len(chains)
            if chains:
                lengths = [self.delegation_graph.get_chain_length(str(d), race_id) for d, _ in chains]
                avglen_pr[race_id] = float(np.mean(lengths))
            else:
                avglen_pr[race_id] = 0.0
        # Abstention reason breakdown
        from collections import Counter
        reason_counts: dict[str, int] = Counter()
        for race_abstentions in self._abstentions.values():
            for _, reason in race_abstentions:
                reason_counts[reason] += 1

        return {
            "gini_per_race": gini_pr, "max_weight_per_race": maxw_pr,
            "chain_count_per_race": cnt_pr, "avg_chain_length_per_race": avglen_pr,
            "gini_history": self._gini_history,
            "total_delegators": sum(cnt_pr.values()),
            "total_direct_voters": sum(len(v) for v in self._direct_votes.values()),
            "total_abstentions": sum(len(v) for v in self._abstentions.values()),
            "abstention_reasons": dict(reason_counts),
        }

    # ------------------------------------------------------------------
    # Public data accessors
    # ------------------------------------------------------------------

    def get_results(self) -> SimulationResults:
        """Return final results. Runs tally if not yet done."""
        if self.results is None:
            self._tally()
        return self.results  # type: ignore[return-value]

    def get_opinion_history(self) -> dict[int, list]:
        """Opinion snapshots keyed by ideology dimension."""
        return {0: [s.tolist() for s in self._opinion_snapshots]}

    def get_delegation_stats(self) -> dict:
        """Current delegation statistics."""
        return self._compute_delegation_stats()

    def get_agent_summary(self) -> dict:
        """Aggregate agent statistics."""
        n = len(self.agents)
        ideologies = np.array([a.ideology[0] for a in self.agents.values()])
        parties = [a.party_id.value for a in self.agents.values()]
        return {
            "n_agents": n,
            "ideology_mean": float(np.mean(ideologies)),
            "ideology_std": float(np.std(ideologies)),
            "ideology_median": float(np.median(ideologies)),
            "party_distribution": {p: parties.count(p) / n for p in set(parties)},
            "mean_knowledge": float(np.mean([
                s.knowledge_level for a in self.agents.values() for s in a.race_states.values()
            ])),
            "mean_engagement": float(np.mean([a.engagement_level for a in self.agents.values()])),
            "n_llm_agents": sum(1 for a in self.agents.values() if a.is_llm_agent),
        }

    def get_dynamics_report(self) -> dict:
        """Report on opinion convergence, stance changes, and cross-party shifts.

        Returns:
            convergence_tick: when mean opinion delta dropped below 0.001
            opinion_deltas: per-tick mean absolute ideology change
            total_stance_changes: agents whose vote choice differs from their initial party lean
            cross_party_shifts: agents who moved from D-leaning to R-leaning or vice versa
            shift_by_group: {party: {stayed, shifted_right, shifted_left, shifted_party}}
            delegation_turnout_boost: extra effective votes from delegation vs FPTP
        """
        n = len(self.agents)

        # Convergence
        convergence = self._convergence_tick or len(self._opinion_deltas)

        # Cross-party and stance shift analysis
        shift_by_group: dict[str, dict[str, int]] = {}
        total_cross_party = 0
        total_stance_changed = 0

        for aid, agent in self.agents.items():
            initial_ideo = self._initial_ideologies.get(aid, 0.0)
            initial_party = self._initial_parties.get(aid, "independent")
            current_ideo = float(agent.ideology[0])

            # Classify initial and current party lean
            def _lean(ideo: float) -> str:
                if ideo < -0.15:
                    return "D"
                elif ideo > 0.15:
                    return "R"
                return "I"

            initial_lean = _lean(initial_ideo)
            current_lean = _lean(current_ideo)
            shift = current_ideo - initial_ideo

            group = initial_party
            if group not in shift_by_group:
                shift_by_group[group] = {
                    "total": 0, "stayed": 0,
                    "shifted_right": 0, "shifted_left": 0,
                    "crossed_party": 0, "mean_shift": 0.0,
                }
            shift_by_group[group]["total"] += 1

            if initial_lean == current_lean:
                shift_by_group[group]["stayed"] += 1
            else:
                total_cross_party += 1
                shift_by_group[group]["crossed_party"] += 1

            if shift > 0.05:
                shift_by_group[group]["shifted_right"] += 1
            elif shift < -0.05:
                shift_by_group[group]["shifted_left"] += 1
            else:
                shift_by_group[group]["stayed"] += 1

            # Did their actual vote differ from initial ideology's nearest candidate?
            for race_id, state in agent.race_states.items():
                if state.vote_choice:
                    race_candidates = self.config.races.get(race_id, [])
                    if race_candidates:
                        expected = min(race_candidates, key=lambda c: abs(initial_ideo - _candidate_ideology(c)))
                    else:
                        expected = "Democrat" if initial_ideo < 0 else "Republican"
                    if state.vote_choice != expected:
                        total_stance_changed += 1

        # Compute mean shift per group
        for group, data in shift_by_group.items():
            agents_in_group = [
                float(self.agents[aid].ideology[0]) - self._initial_ideologies[aid]
                for aid in self.agents
                if self._initial_parties.get(aid) == group
            ]
            if agents_in_group:
                data["mean_shift"] = round(float(np.mean(agents_in_group)), 4)

        # Delegation turnout boost
        stats = self._compute_delegation_stats()
        deleg_effective = 0
        fptp_effective = 0
        for race_id in self.config.races:
            dr = getattr(self.results, "delegation_results", {}).get(race_id, {}) if self.results else {}
            fr = getattr(self.results, "fptp_results", {}).get(race_id, {}) if self.results else {}
            deleg_effective += dr.get("total_effective_votes", 0)
            fptp_effective += fr.get("total_votes", 0)

        return {
            "convergence_tick": convergence,
            "converged": self._convergence_tick is not None,
            "opinion_deltas": [round(d, 6) for d in self._opinion_deltas],
            "total_cross_party_shifts": total_cross_party,
            "cross_party_pct": round(total_cross_party / n * 100, 1) if n else 0,
            "total_stance_changes": total_stance_changed,
            "shift_by_group": shift_by_group,
            "delegation_turnout_boost": round(deleg_effective - fptp_effective, 1),
            "fptp_effective_votes": fptp_effective,
            "delegation_effective_votes": round(deleg_effective, 1),
        }

    def get_llm_vs_rule_stats(self) -> dict:
        """Compare LLM agent decisions against rule-based agent decisions.

        Returns per-race breakdown of action distributions (vote/delegate/abstain)
        and candidate choice distributions for both groups.
        """
        stats: dict = {"llm_enabled": self.config.use_llm}
        if not self.config.use_llm:
            return stats

        n_llm = sum(1 for a in self.agents.values() if a.is_llm_agent)
        n_rule = len(self.agents) - n_llm
        stats["n_llm"] = n_llm
        stats["n_rule"] = n_rule

        for race_id in self.config.races:
            llm_decs = self._llm_decisions[race_id]
            rule_decs = self._rule_decisions[race_id]

            def _summarize(decs: list) -> dict:
                if not decs:
                    return {"total": 0, "actions": {}, "choices": {}, "top_reasons": []}
                actions: dict[str, int] = {}
                choices: dict[str, int] = {}
                reasons: dict[str, int] = {}
                _skip = {"", "llm_decision", "llm_error"}
                for _, action, choice, reason in decs:
                    actions[action] = actions.get(action, 0) + 1
                    if action == "vote" and choice:
                        choices[choice] = choices.get(choice, 0) + 1
                    if reason and reason not in _skip:
                        # Normalize: lowercase, strip quotes/punctuation
                        key = reason.strip().strip('"').strip("'").lower().rstrip(".")
                        reasons[key] = reasons.get(key, 0) + 1
                total = len(decs)
                top_reasons = sorted(reasons.items(), key=lambda x: -x[1])[:5]
                return {
                    "total": total,
                    "actions": {a: round(c / total * 100, 1) for a, c in actions.items()},
                    "choices": {c: round(n / max(sum(choices.values()), 1) * 100, 1) for c, n in choices.items()},
                    "top_reasons": [(r, cnt) for r, cnt in top_reasons],
                }

            stats[race_id] = {
                "llm": _summarize(llm_decs),
                "rule": _summarize(rule_decs),
            }

        return stats

    def get_influence_summary(self) -> dict:
        """Summarize influence attribution across the campaign phase."""
        if not self._influence_attribution:
            return {"peer_total": 0, "media_total": 0, "peer_pct": 0, "media_pct": 0, "per_tick": []}

        peer_total = sum(d["peer"] for d in self._influence_attribution)
        media_total = sum(d["media"] for d in self._influence_attribution)
        grand = peer_total + media_total
        return {
            "peer_total": round(peer_total, 4),
            "media_total": round(media_total, 4),
            "peer_pct": round(peer_total / grand * 100, 1) if grand > 0 else 0,
            "media_pct": round(media_total / grand * 100, 1) if grand > 0 else 0,
            "per_tick": self._influence_attribution,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values."""
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    weighted_sum = sum(
        (2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals)
    )
    return weighted_sum / (n * total)
