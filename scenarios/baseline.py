"""Baseline scenario: organic delegation with default parameters.

No external shocks, no celebrity entry, no adversarial attacks.
Tests the natural equilibrium of liquid democracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaselineConfig:
    """Configuration for the baseline scenario."""

    n_agents: int = 10_000
    pvi_lean: float = 0.0
    seed: int = 42
    races: dict[str, list[str]] = field(
        default_factory=lambda: {
            "race": ["Democrat", "Republican"],
        }
    )
    # Simulation overrides forwarded to SimulationConfig
    sim_overrides: dict[str, Any] = field(default_factory=dict)


def run_baseline(config: BaselineConfig | None = None) -> dict[str, Any]:
    """Run the baseline scenario."""
    if config is None:
        config = BaselineConfig()

    from engine.simulation import LiquidDemocracyModel, SimulationConfig

    sim_config = SimulationConfig(
        n_agents=config.n_agents,
        pvi_lean=config.pvi_lean,
        seed=config.seed,
        races=config.races,
        **config.sim_overrides,
    )

    model = LiquidDemocracyModel(sim_config)
    model.run()

    return {
        "results": model.get_results(),
        "opinion_history": model.get_opinion_history(),
        "delegation_stats": model.get_delegation_stats(),
        "config": config,
    }
