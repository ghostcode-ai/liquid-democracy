"""Hub-targeting attack scenario.

An adversary identifies high-delegation hubs and attempts to
compromise or influence them, demonstrating the vulnerability
of concentrated delegation to targeted attacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.voter_agent import VoterAgent


@dataclass
class HubAttackConfig:
    """Configuration for the hub-targeting attack scenario."""

    n_agents: int = 10_000
    pvi_lean: float = 0.0
    seed: int = 42
    races: dict[str, list[str]] = field(
        default_factory=lambda: {
            "race": ["Democrat", "Republican"],
        }
    )

    # Attack parameters
    attack_tick: int = 70  # late campaign, after delegations form
    n_hubs_targeted: int = 5  # how many top hubs to compromise
    ideology_shift: float = 0.6  # how much to shift compromised hubs
    trust_boost_to_compromised: float = 0.3  # extra trust agents give to compromised hubs

    # Simulation overrides forwarded to SimulationConfig
    sim_overrides: dict[str, Any] = field(default_factory=dict)


def identify_hubs(
    agents: dict[int, VoterAgent],
    delegation_graph,
    topic: str,
    n_hubs: int,
) -> list[int]:
    """Identify the top-N delegation hubs by incoming delegation count."""
    weights = delegation_graph.resolve_all(topic)
    sorted_by_weight = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    # Convert string graph IDs back to int
    return [int(aid) for aid, _ in sorted_by_weight[:n_hubs]]


def compromise_hubs(
    agents: dict[int, VoterAgent],
    hub_ids: list[int],
    ideology_shift: float,
    trust_boost: float,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Shift hub agents' ideology and boost trust toward them.

    Simulates an adversary who compromises delegates and manipulates
    the trust network to retain their delegations after the shift.

    Returns {hub_id: original_primary_ideology} for tracking.
    """
    original = {}
    for hid in hub_ids:
        if hid not in agents:
            continue
        agent = agents[hid]
        original[hid] = float(agent.ideology[0])

        # Shift primary ideology dimension
        agent.ideology[0] = np.clip(agent.ideology[0] + ideology_shift, -1.0, 1.0)

        # Also shift their race preference
        for state in agent.race_states.values():
            state.preference = np.clip(state.preference + ideology_shift, -1.0, 1.0)

        # Boost trust toward compromised hub (adversary astroturfs credibility)
        if trust_boost > 0:
            for neighbor_id in agent.neighbors:
                neighbor = agents.get(neighbor_id)
                if neighbor is None:
                    continue
                current = neighbor.trust_scores.get(hid, 0.3)
                neighbor.trust_scores[hid] = min(1.0, current + trust_boost)

    return original


def run_hub_attack(config: HubAttackConfig | None = None) -> dict[str, Any]:
    """Run the hub-targeting attack scenario."""
    if config is None:
        config = HubAttackConfig()

    from engine.simulation import LiquidDemocracyModel, SimulationConfig

    sim_config = SimulationConfig(
        n_agents=config.n_agents,
        pvi_lean=config.pvi_lean,
        seed=config.seed,
        races=config.races,
        **config.sim_overrides,
    )

    model = LiquidDemocracyModel(sim_config)

    # Run campaign until attack point
    for _ in range(config.attack_tick):
        model.step()

    # Identify and compromise hubs
    hub_ids = identify_hubs(
        model.agents, model.delegation_graph, "race", config.n_hubs_targeted
    )
    rng = np.random.default_rng(config.seed + 2000)
    original_ideologies = compromise_hubs(
        model.agents, hub_ids, config.ideology_shift,
        config.trust_boost_to_compromised, rng,
    )

    # Continue through voting + tally
    remaining = sim_config.campaign_ticks + sim_config.voting_ticks + 1 - config.attack_tick
    for _ in range(remaining):
        model.step()

    results = model.get_results()

    return {
        "results": results,
        "compromised_hubs": hub_ids,
        "original_ideologies": original_ideologies,
        "total_delegations_affected": sum(
            len(model.delegation_graph.get_delegators(str(hid), "race"))
            for hid in hub_ids
        ),
        "config": config,
    }
