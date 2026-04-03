#!/usr/bin/env python3
"""Parameter sensitivity sweep for the liquid democracy simulation.

Varies the top 7 high-sensitivity parameters from the spec one at a time,
holding others at defaults, to produce a tornado chart of outcome sensitivity.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SweepParameter:
    """A parameter to sweep."""

    name: str
    default: float
    values: list[float]
    config_field: str  # field name in SimulationConfig


# Top 7 high-sensitivity parameters from spec section 8
DEFAULT_SWEEP_PARAMS = [
    SweepParameter(
        name="Preferential Attachment (gamma)",
        default=1.5,
        values=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        config_field="preferential_attachment_gamma",
    ),
    SweepParameter(
        name="Delegation Options (k)",
        default=1,
        values=[1, 2, 3, 5],
        config_field="delegation_options_k",
    ),
    SweepParameter(
        name="Viscous Decay (alpha)",
        default=0.85,
        values=[0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0],
        config_field="viscous_decay_alpha",
    ),
    SweepParameter(
        name="Delegation Probability",
        default=0.10,
        values=[0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
        config_field="delegation_probability_base",
    ),
    SweepParameter(
        name="Bandwagon Coefficient",
        default=0.25,
        values=[0.0, 0.10, 0.25, 0.50, 0.75, 1.0],
        config_field="bandwagon_coefficient",
    ),
    SweepParameter(
        name="Weight Cap",
        default=0,  # 0 means no cap
        values=[0, 10, 50, 100, 500, 1000],
        config_field="weight_cap",
    ),
    SweepParameter(
        name="Bounded Confidence (epsilon)",
        default=0,  # 0 means no filter
        values=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        config_field="bounded_confidence_epsilon",
    ),
]


def run_single(
    param_name: str,
    param_field: str,
    param_value: float,
    n_agents: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single simulation with one parameter modified."""
    from engine.simulation import LiquidDemocracyModel, SimulationConfig

    kwargs = {"n_agents": n_agents, "seed": seed}

    # Handle special cases
    if param_field == "weight_cap":
        kwargs[param_field] = None if param_value == 0 else param_value
    elif param_field == "bounded_confidence_epsilon":
        kwargs[param_field] = None if param_value == 0 else param_value
    else:
        kwargs[param_field] = param_value

    config = SimulationConfig(**kwargs)
    model = LiquidDemocracyModel(config)
    model.run()

    results = model.get_results()
    stats = model.get_delegation_stats()

    # Use the first race key from the config (default is "race")
    race_id = next(iter(config.races))

    return {
        "param_name": param_name,
        "param_field": param_field,
        "param_value": param_value,
        "gini": stats.get("gini_per_race", {}).get(race_id, 0),
        "max_weight": stats.get("max_weight_per_race", {}).get(race_id, 0),
        "fptp_winner": results.fptp_results.get(race_id, {}).get("winner", ""),
        "delegation_winner": results.delegation_results.get(race_id, {}).get(
            "winner", ""
        ),
        "outcome_divergence": (
            results.fptp_results.get(race_id, {}).get("winner", "")
            != results.delegation_results.get(race_id, {}).get("winner", "")
        ),
    }


def run_sweep(
    params: list[SweepParameter] | None = None,
    n_agents: int = 1000,
    seed: int = 42,
    output_path: str | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run a full parameter sensitivity sweep.

    Returns list of result dicts, one per (parameter, value) combination.
    """
    if params is None:
        params = DEFAULT_SWEEP_PARAMS

    all_results = []
    total_runs = sum(len(p.values) for p in params)
    run_idx = 0

    for param in params:
        if verbose:
            print(f"\nSweeping: {param.name}")

        for value in param.values:
            run_idx += 1
            if verbose:
                print(f"  [{run_idx}/{total_runs}] {param.name} = {value}...", end=" ")

            start = time.time()
            result = run_single(
                param.name, param.config_field, value, n_agents, seed
            )
            elapsed = time.time() - start

            if verbose:
                print(f"Gini={result['gini']:.3f}, "
                      f"MaxW={result['max_weight']:.1f} "
                      f"({elapsed:.1f}s)")

            all_results.append(result)

    # Save results
    if output_path:
        path = Path(output_path)
        if path.suffix == ".csv":
            _write_csv(all_results, path)
        else:
            path.write_text(json.dumps(all_results, indent=2, default=str))

        if verbose:
            print(f"\nResults saved to {path}")

    return all_results


def _write_csv(results: list[dict], path: Path) -> None:
    """Write results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def compute_sensitivity(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute sensitivity scores for each parameter.

    Sensitivity = range of Gini values across parameter sweep / max Gini range.
    Higher = more sensitive.
    """
    param_ranges = {}
    for result in results:
        name = result["param_name"]
        gini = result["gini"]
        if name not in param_ranges:
            param_ranges[name] = {"min": gini, "max": gini}
        param_ranges[name]["min"] = min(param_ranges[name]["min"], gini)
        param_ranges[name]["max"] = max(param_ranges[name]["max"], gini)

    return {
        name: ranges["max"] - ranges["min"]
        for name, ranges in param_ranges.items()
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parameter sensitivity sweep")
    parser.add_argument(
        "--agents", type=int, default=1000, help="Agents per run (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default="sweep_results.csv", help="Output file"
    )
    args = parser.parse_args()

    print(f"Running parameter sweep with {args.agents} agents...")
    results = run_sweep(
        n_agents=args.agents, seed=args.seed, output_path=args.output
    )

    print("\n=== Sensitivity Ranking ===")
    sensitivity = compute_sensitivity(results)
    for name, score in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
