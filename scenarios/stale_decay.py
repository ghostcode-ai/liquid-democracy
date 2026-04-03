"""Stale delegation decay scenario.

Tests what happens when delegations persist without review.
In LiquidFeedback, delegations had no expiry — this was a known
failure mode where delegates accumulated power without ongoing
consent from delegators. This scenario models delegation persistence
and its effect on outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.voter_agent import VoterAgent


@dataclass
class StaleDecayConfig:
    """Configuration for the stale delegation scenario."""

    n_agents: int = 10_000
    pvi_lean: float = 0.0
    seed: int = 42
    races: dict[str, list[str]] = field(
        default_factory=lambda: {
            "race": ["Democrat", "Republican"],
        }
    )

    # Stale delegation parameters
    delegation_persistence_ticks: int = 80  # how many ticks a delegation survives without review
    review_probability: float = 0.05  # prob of reviewing delegation per tick
    ideology_drift_rate: float = 0.02  # how fast delegates drift per tick

    # Simulation overrides forwarded to SimulationConfig
    sim_overrides: dict[str, Any] = field(default_factory=dict)


def apply_delegate_drift(
    agents: dict[int, VoterAgent],
    delegation_graph,
    topic: str,
    drift_rate: float,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Apply ideology drift to delegates (simulating changing positions).

    High-delegation delegates drift more (power corrupts / pressure to moderate).
    Returns {agent_id: drift_amount}.
    """
    weights = delegation_graph.resolve_all(topic)
    drifts = {}

    for aid, weight in weights.items():
        if weight <= 1.0:
            continue  # only delegates drift
        agent_key = int(aid) if str(aid).isdigit() else None
        agent = agents.get(agent_key) if agent_key is not None else None
        if agent is None:
            continue

        # Drift proportional to weight (more power -> more drift)
        drift = rng.normal(0, drift_rate * np.log1p(weight))
        agent.ideology[0] = np.clip(agent.ideology[0] + drift, -1.0, 1.0)
        drifts[aid] = drift

    return drifts


def apply_stale_review(
    agents: dict[int, VoterAgent],
    delegation_graph,
    topic: str,
    review_probability: float,
    delegation_ages: dict[str, int],
    persistence_ticks: int,
    current_tick: int,
    rng: np.random.Generator,
) -> int:
    """Agents review their delegation and revoke if delegate drifted or delegation expired.

    delegation_ages tracks {voter_id_str: tick_when_delegated}.
    Delegations older than persistence_ticks are automatically reviewed.

    Returns count of revocations.
    """
    revocations = 0

    for voter_id in list(delegation_graph.graph.nodes()):
        delegate_id = delegation_graph.get_delegate(voter_id, topic)
        if delegate_id is None:
            continue

        # Forced review if delegation is stale (exceeded persistence)
        age = current_tick - delegation_ages.get(voter_id, 0)
        forced_review = age > persistence_ticks

        # Random review
        voluntary_review = rng.random() < review_probability

        if not forced_review and not voluntary_review:
            continue

        voter = agents.get(int(voter_id)) if str(voter_id).isdigit() else None
        delegate = agents.get(int(delegate_id)) if delegate_id and str(delegate_id).isdigit() else None
        if voter is None or delegate is None:
            continue

        # Revoke if delegate's ideology drifted too far from voter's
        ideology_distance = abs(voter.ideology[0] - delegate.ideology[0])
        revoke_threshold = 0.3 if forced_review else 0.5
        if ideology_distance > revoke_threshold:
            delegation_graph.revoke(voter_id, topic)
            revocations += 1

    return revocations


def run_stale_decay(config: StaleDecayConfig | None = None) -> dict[str, Any]:
    """Run the stale delegation decay scenario."""
    if config is None:
        config = StaleDecayConfig()

    from engine.simulation import LiquidDemocracyModel, SimulationConfig

    sim_config = SimulationConfig(
        n_agents=config.n_agents,
        pvi_lean=config.pvi_lean,
        seed=config.seed,
        races=config.races,
        **config.sim_overrides,
    )

    model = LiquidDemocracyModel(sim_config)
    rng = np.random.default_rng(config.seed + 3000)

    gini_over_time = []
    drift_history = []
    revocation_history = []
    # Track when each delegation was created
    delegation_ages: dict[str, int] = {}

    # Run campaign with drift and stale review
    for tick in range(sim_config.campaign_ticks):
        model.step()

        # Record new delegations this tick
        for race_id in config.races:
            for rec in model._delegation_records.get(race_id, []):
                voter_str = str(rec[0])
                if voter_str not in delegation_ages:
                    delegation_ages[voter_str] = tick

        # Apply drift to delegates
        drifts = apply_delegate_drift(
            model.agents,
            model.delegation_graph,
            "race",
            config.ideology_drift_rate,
            rng,
        )
        drift_history.append(drifts)

        # Apply stale review with persistence enforcement
        revocations = apply_stale_review(
            model.agents,
            model.delegation_graph,
            "race",
            config.review_probability,
            delegation_ages,
            config.delegation_persistence_ticks,
            tick,
            rng,
        )
        revocation_history.append(revocations)

        # Track Gini
        gini = model.delegation_graph.get_gini("race")
        gini_over_time.append(gini)

    # Continue with voting + tally
    for _ in range(sim_config.voting_ticks + 1):
        model.step()

    results = model.get_results()

    return {
        "results": results,
        "gini_over_time": gini_over_time,
        "total_revocations": sum(revocation_history),
        "max_gini": max(gini_over_time) if gini_over_time else 0.0,
        "final_gini": gini_over_time[-1] if gini_over_time else 0.0,
        "config": config,
    }
