"""k=2 multiple delegation options mitigation scenario.

Tests the hypothesis from the spec: with k>=2 delegation options,
max vote weight drops from Theta(sqrt(n)) to Theta(log(n)),
dramatically reducing power concentration.

Each voter can delegate to up to k different delegates, with their
vote weight split among them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.voter_agent import VoterAgent
from engine.delegation_graph import DelegationGraph


@dataclass
class K2Config:
    """Configuration for the k=2 mitigation scenario."""

    n_agents: int = 10_000
    pvi_lean: float = 0.0
    seed: int = 42
    races: dict[str, list[str]] = field(
        default_factory=lambda: {
            "race": ["Democrat", "Republican"],
        }
    )

    # k-delegation parameters
    delegation_options_k: int = 2
    compare_k1: bool = True

    # Simulation overrides forwarded to SimulationConfig
    sim_overrides: dict[str, Any] = field(default_factory=dict)


def run_k2_mitigation(config: K2Config | None = None) -> dict[str, Any]:
    """Run the k=2 mitigation scenario and optionally compare with k=1."""
    if config is None:
        config = K2Config()

    from engine.simulation import LiquidDemocracyModel, SimulationConfig

    # Remove delegation_options_k from overrides — we set it explicitly
    overrides = {k: v for k, v in config.sim_overrides.items() if k != "delegation_options_k"}

    # Run with k=2
    sim_config_k2 = SimulationConfig(
        n_agents=config.n_agents,
        pvi_lean=config.pvi_lean,
        seed=config.seed,
        races=config.races,
        delegation_options_k=config.delegation_options_k,
        **overrides,
    )

    model_k2 = LiquidDemocracyModel(sim_config_k2)
    model_k2.run()
    results_k2 = model_k2.get_results()
    stats_k2 = model_k2.get_delegation_stats()

    # Canonical "results" key points to the k=2 run
    output: dict[str, Any] = {
        "results": results_k2,
        "k2_results": results_k2,
        "k2_stats": stats_k2,
        "config": config,
    }

    # Optionally compare with k=1
    if config.compare_k1:
        sim_config_k1 = SimulationConfig(
            n_agents=config.n_agents,
            pvi_lean=config.pvi_lean,
            seed=config.seed,
            races=config.races,
            delegation_options_k=1,
            **overrides,
        )

        model_k1 = LiquidDemocracyModel(sim_config_k1)
        model_k1.run()
        results_k1 = model_k1.get_results()
        stats_k1 = model_k1.get_delegation_stats()

        k1_gini = stats_k1.get("gini_per_race", {}).get("race", 0)
        k2_gini = stats_k2.get("gini_per_race", {}).get("race", 0)
        k1_max = stats_k1.get("max_weight_per_race", {}).get("race", 0)
        k2_max = stats_k2.get("max_weight_per_race", {}).get("race", 0)

        output["k1_results"] = results_k1
        output["k1_stats"] = stats_k1
        output["comparison"] = {
            "k1_gini": k1_gini,
            "k2_gini": k2_gini,
            "k1_max_weight": k1_max,
            "k2_max_weight": k2_max,
            "gini_reduction": k1_gini - k2_gini,
        }

    return output
