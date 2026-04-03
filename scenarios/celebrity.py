"""Celebrity mega-delegate scenario.

A high-profile figure enters the delegation network mid-campaign,
attracting a disproportionate number of delegations through fame
rather than expertise. Tests how celebrity status distorts outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.voter_agent import Demographics, PartyID, VoterAgent


@dataclass
class CelebrityConfig:
    """Configuration for the celebrity scenario."""

    n_agents: int = 10_000
    pvi_lean: float = 0.0
    seed: int = 42
    races: dict[str, list[str]] = field(
        default_factory=lambda: {
            "race": ["Democrat", "Republican"],
        }
    )

    # Celebrity parameters
    celebrity_ideology: float = -0.3  # slightly left-leaning
    celebrity_party: PartyID = PartyID.LEAN_D
    celebrity_trust_boost: float = 0.8  # starts with high trust
    celebrity_entry_tick: int = 40  # enters mid-campaign
    celebrity_reach_fraction: float = 0.3  # 30% of agents become aware

    # Simulation overrides forwarded to SimulationConfig
    sim_overrides: dict[str, Any] = field(default_factory=dict)


def inject_celebrity(
    agents: dict[int, VoterAgent],
    config: CelebrityConfig,
    rng: np.random.Generator,
) -> int:
    """Inject a celebrity agent into the population.

    The celebrity gets artificially high trust from a large fraction of agents.
    Returns the celebrity's agent_id.
    """
    celebrity_id = max(agents.keys()) + 1

    ideology = np.full(10, config.celebrity_ideology)
    demographics = Demographics(
        age=45,
        income=5_000_000.0,
        education=3,
        race="white",
        gender="male",
        urban_rural="urban",
    )

    celebrity = VoterAgent.from_profile(
        agent_id=celebrity_id,
        demographics=demographics,
        ideology=ideology,
        party_id=config.celebrity_party,
    )
    celebrity.political_knowledge = 0.9
    celebrity.engagement_level = 1.0

    agents[celebrity_id] = celebrity

    # Boost trust from a fraction of the population
    n_aware = int(len(agents) * config.celebrity_reach_fraction)
    aware_ids = rng.choice(
        [aid for aid in agents if aid != celebrity_id],
        size=min(n_aware, len(agents) - 1),
        replace=False,
    )

    for aid in aware_ids:
        agents[aid].trust_scores[celebrity_id] = config.celebrity_trust_boost
        agents[aid].neighbors.append(celebrity_id)

    celebrity.neighbors = list(aware_ids)

    return celebrity_id


def run_celebrity(config: CelebrityConfig | None = None) -> dict[str, Any]:
    """Run the celebrity mega-delegate scenario."""
    if config is None:
        config = CelebrityConfig()

    from engine.simulation import LiquidDemocracyModel, SimulationConfig

    sim_config = SimulationConfig(
        n_agents=config.n_agents,
        pvi_lean=config.pvi_lean,
        seed=config.seed,
        races=config.races,
        **config.sim_overrides,
    )

    model = LiquidDemocracyModel(sim_config)

    # Run campaign until celebrity entry point
    for _ in range(config.celebrity_entry_tick):
        model.step()

    # Inject celebrity
    rng = np.random.default_rng(config.seed + 1000)
    celebrity_id = inject_celebrity(model.agents, config, rng)

    # Add celebrity to social graph and rebuild matrices
    model.social_graph.add_node(celebrity_id)
    for nid in model.agents[celebrity_id].neighbors:
        model.social_graph.add_edge(celebrity_id, nid)
    # Init race state for the new agent
    from agents.voter_agent import RaceState
    for race_id in config.races:
        model.agents[celebrity_id].race_states[race_id] = RaceState(
            knowledge_level=0.9,
            preference=float(model.agents[celebrity_id].ideology[0]),
        )
    model.rebuild_opinion_matrices()

    # Continue campaign + voting + tally
    remaining = sim_config.campaign_ticks + sim_config.voting_ticks + 1 - config.celebrity_entry_tick
    for _ in range(remaining):
        model.step()

    results = model.get_results()

    return {
        "results": results,
        "celebrity_id": celebrity_id,
        "celebrity_delegation_count": len(
            model.delegation_graph.get_delegators(str(celebrity_id), "race")
        ),
        "config": config,
    }
