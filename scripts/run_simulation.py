#!/usr/bin/env python3
"""CLI entry point for running liquid democracy simulations.

Supports both flag-based and interactive modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# -- ANSI helpers --
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

SCENARIOS = {
    "baseline": "Organic delegation, default parameters",
    "celebrity": "Celebrity mega-delegate enters mid-campaign",
    "hub_attack": "Adversary compromises top delegation hubs",
    "stale_decay": "Delegations persist without review, delegates drift",
    "k2": "Voters delegate to 2 people (splits concentration)",
}

SCENARIO_KEYS = list(SCENARIOS.keys())


def prompt_choice(prompt: str, options: list[str], default: str) -> str:
    """Prompt the user to pick from a numbered list."""
    print(f"\n{BOLD}{prompt}{RESET}")
    for i, opt in enumerate(options, 1):
        marker = f" {DIM}(default){RESET}" if opt == default else ""
        desc = SCENARIOS.get(opt, "")
        if desc:
            print(f"  {CYAN}{i}{RESET}) {opt:<14} {DIM}{desc}{RESET}{marker}")
        else:
            print(f"  {CYAN}{i}{RESET}) {opt}{marker}")
    while True:
        raw = input(f"{DIM}  Enter number or name [{default}]: {RESET}").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        print(f"  {RED}Invalid choice. Try again.{RESET}")


def prompt_int(prompt: str, default: int, lo: int = 1, hi: int = 100_000) -> int:
    """Prompt for an integer with bounds."""
    while True:
        raw = input(f"  {prompt} {DIM}[{default}]{RESET}: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  {RED}Must be between {lo} and {hi}.{RESET}")
        except ValueError:
            print(f"  {RED}Enter a whole number.{RESET}")


def prompt_float(prompt: str, default: float, lo: float = -100, hi: float = 100) -> float:
    """Prompt for a float with bounds."""
    while True:
        raw = input(f"  {prompt} {DIM}[{default}]{RESET}: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  {RED}Must be between {lo} and {hi}.{RESET}")
        except ValueError:
            print(f"  {RED}Enter a number.{RESET}")


def prompt_optional_float(prompt: str, default: str = "none") -> float | None:
    """Prompt for a float or 'none'."""
    raw = input(f"  {prompt} {DIM}[{default}]{RESET}: ").strip()
    if not raw or raw.lower() == "none":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def interactive_config() -> dict:
    """Walk the user through configuration interactively."""
    print(f"\n{BOLD}{'=' * 56}{RESET}")
    print(f"{BOLD}  Liquid Democracy Simulation Engine{RESET}")
    print(f"{BOLD}{'=' * 56}{RESET}")

    scenario = prompt_choice("Select scenario:", SCENARIO_KEYS, "baseline")

    print(f"\n{BOLD}Population:{RESET}")
    n_agents = prompt_int("Number of agents", 1000, 50, 100_000)
    pvi = prompt_float("Cook PVI lean (negative=D, positive=R)", 0.0, -30, 30)
    seed = prompt_int("Random seed", 42, 0, 999_999)

    print(f"\n{BOLD}Delegation parameters:{RESET}")
    deleg_prob = prompt_float("Delegation probability per tick", 0.10, 0.0, 1.0)
    alpha = prompt_float("Viscous decay alpha (0=no delegation, 1=full)", 0.85, 0.0, 1.0)
    weight_cap = prompt_optional_float("Weight cap per delegate (none=unlimited)", "none")
    k = prompt_int("Delegation options k (1=single, 2+=split)", 1, 1, 10)

    print(f"\n{BOLD}Advanced (press Enter for defaults):{RESET}")
    gamma = prompt_float("Preferential attachment gamma", 1.5, 0.0, 5.0)
    bandwagon = prompt_float("Bandwagon coefficient", 0.25, 0.0, 1.0)
    epsilon = prompt_optional_float("Bounded confidence epsilon (none=off)", "none")

    return {
        "scenario": scenario,
        "agents": n_agents,
        "pvi": pvi,
        "seed": seed,
        "delegation_probability_base": deleg_prob,
        "viscous_decay_alpha": alpha,
        "weight_cap": weight_cap,
        "delegation_options_k": k,
        "preferential_attachment_gamma": gamma,
        "bandwagon_coefficient": bandwagon,
        "bounded_confidence_epsilon": epsilon,
    }


# -- Result display --

def display_results(output: dict) -> None:
    """Pretty-print simulation results."""
    results = output.get("results")
    if results is None:
        print(f"{RED}No results returned.{RESET}")
        return

    races = list(results.fptp_results.keys())

    # -- Election outcomes table --
    print(f"\n{BOLD}{'=' * 56}{RESET}")
    print(f"{BOLD}  Election Results{RESET}")
    print(f"{BOLD}{'=' * 56}{RESET}")

    systems = [
        ("FPTP", results.fptp_results),
        ("RCV", results.rcv_results),
        ("TRS", results.trs_results),
        ("Delegation", results.delegation_results),
    ]

    for race_id in races:
        print(f"\n  {BOLD}{race_id.upper()}{RESET}")
        winners = []
        for label, data in systems:
            result = data.get(race_id, {})
            winner = result.get("winner", "?")
            winners.append(winner)

            counts = result.get("counts") or result.get("weighted_counts") or {}
            count_str = ", ".join(f"{c}: {v}" for c, v in sorted(counts.items())) if counts else ""

            color = BLUE if "Democrat" in str(winner) else RED if "Republican" in str(winner) else ""
            print(f"    {label:<12} {color}{winner:<14}{RESET} {DIM}{count_str}{RESET}")

        # Check for divergence
        unique_winners = set(winners)
        if len(unique_winners) > 1:
            print(f"    {YELLOW}^ Systems disagree on winner{RESET}")

    # -- Delegation stats --
    d_stats = results.delegation_stats
    if d_stats:
        print(f"\n{BOLD}{'=' * 56}{RESET}")
        print(f"{BOLD}  Delegation Statistics{RESET}")
        print(f"{BOLD}{'=' * 56}{RESET}")

        gini_per_race = d_stats.get("gini_per_race", {})
        max_w_per_race = d_stats.get("max_weight_per_race", {})
        chain_count = d_stats.get("chain_count_per_race", {})
        avg_chain = d_stats.get("avg_chain_length_per_race", {})

        for race_id in races:
            gini = gini_per_race.get(race_id, 0)
            max_w = max_w_per_race.get(race_id, 0)
            chains = chain_count.get(race_id, 0)
            avg_len = avg_chain.get(race_id, 0)

            gini_color = GREEN if gini < 0.3 else YELLOW if gini < 0.6 else RED
            print(f"\n  {BOLD}{race_id.upper()}{RESET}")
            print(f"    Gini coefficient:   {gini_color}{gini:.3f}{RESET}")
            print(f"    Max delegate weight: {max_w:.1f}")
            print(f"    Delegation chains:   {chains}")
            print(f"    Avg chain length:    {avg_len:.1f}")

        total_d = d_stats.get("total_delegators", 0)
        total_v = d_stats.get("total_direct_voters", 0)
        total_a = d_stats.get("total_abstentions", 0)
        total = total_d + total_v + total_a

        print(f"\n  {BOLD}Participation{RESET}")
        print(f"    Direct voters:  {total_v:>6}  ({_pct(total_v, total)})")
        print(f"    Delegators:     {total_d:>6}  ({_pct(total_d, total)})")
        print(f"    Abstentions:    {total_a:>6}  ({_pct(total_a, total)})")

    # -- Scenario-specific extras --
    if "celebrity_id" in output:
        print(f"\n  {BOLD}Celebrity Impact{RESET}")
        print(f"    Celebrity agent ID:      {output['celebrity_id']}")
        print(f"    Delegations received:    {output.get('celebrity_delegation_count', '?')}")

    if "compromised_hubs" in output:
        print(f"\n  {BOLD}Hub Attack Impact{RESET}")
        print(f"    Hubs compromised:        {len(output['compromised_hubs'])}")
        print(f"    Delegations affected:    {output.get('total_delegations_affected', '?')}")

    if "comparison" in output:
        comp = output["comparison"]
        print(f"\n  {BOLD}k=1 vs k=2 Comparison{RESET}")
        print(f"    k=1 Gini:      {comp.get('k1_gini', 0):.3f}")
        print(f"    k=2 Gini:      {comp.get('k2_gini', 0):.3f}")
        print(f"    Gini reduction: {comp.get('gini_reduction', 0):.3f}")

    if "gini_over_time" in output:
        gini_hist = output["gini_over_time"]
        if gini_hist:
            print(f"\n  {BOLD}Stale Decay Trajectory{RESET}")
            print(f"    Peak Gini:        {output.get('max_gini', 0):.3f}")
            print(f"    Final Gini:       {output.get('final_gini', 0):.3f}")
            print(f"    Total revocations: {output.get('total_revocations', 0)}")


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{n / total * 100:.1f}%"


# -- Scenario runners --

def run_scenario(cfg: dict) -> dict:
    """Dispatch to the appropriate scenario runner.

    For baseline, passes all delegation parameters through to SimulationConfig.
    All delegation tuning knobs are forwarded to every scenario via sim_overrides.
    """
    scenario = cfg["scenario"]
    n = cfg["agents"]
    pvi = cfg["pvi"]
    seed = cfg["seed"]

    # Build sim_overrides dict from CLI config
    sim_overrides: dict = {}
    for key in [
        "delegation_probability_base",
        "viscous_decay_alpha",
        "weight_cap",
        "delegation_options_k",
        "preferential_attachment_gamma",
        "bandwagon_coefficient",
        "bounded_confidence_epsilon",
    ]:
        if key in cfg and cfg[key] is not None:
            sim_overrides[key] = cfg[key]

    common = dict(n_agents=n, pvi_lean=pvi, seed=seed, sim_overrides=sim_overrides)

    if scenario == "baseline":
        from scenarios.baseline import BaselineConfig, run_baseline
        return run_baseline(BaselineConfig(**common))

    elif scenario == "celebrity":
        from scenarios.celebrity import CelebrityConfig, run_celebrity
        return run_celebrity(CelebrityConfig(**common))

    elif scenario == "hub_attack":
        from scenarios.hub_attack import HubAttackConfig, run_hub_attack
        return run_hub_attack(HubAttackConfig(**common))

    elif scenario == "stale_decay":
        from scenarios.stale_decay import StaleDecayConfig, run_stale_decay
        return run_stale_decay(StaleDecayConfig(**common))

    elif scenario == "k2":
        from scenarios.k2_mitigation import K2Config, run_k2_mitigation
        return run_k2_mitigation(K2Config(**common))

    else:
        print(f"{RED}Unknown scenario: {scenario}{RESET}")
        sys.exit(1)


# -- Entry point --

def main():
    parser = argparse.ArgumentParser(
        description="Liquid Democracy Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
{BOLD}scenarios:{RESET}
  baseline      Organic delegation, default parameters
  celebrity     Celebrity mega-delegate enters mid-campaign
  hub_attack    Adversary compromises top delegation hubs
  stale_decay   Delegations persist without review
  k2            Multiple delegation options (reduces concentration)

{BOLD}examples:{RESET}
  %(prog)s --interactive                     Interactive mode
  %(prog)s --agents 5000 --pvi -15           D+15 district, 5000 agents
  %(prog)s --scenario celebrity --agents 2000
  %(prog)s --agents 1000 --weight-cap 50 --output results.json
""",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Interactive mode: prompts for all options",
    )
    parser.add_argument(
        "-s", "--scenario",
        choices=SCENARIO_KEYS, default="baseline",
        help="Scenario to run (default: baseline)",
    )
    parser.add_argument("-n", "--agents", type=int, default=1000, help="Number of agents (default: 1000)")
    parser.add_argument("--pvi", type=float, default=0.0, help="Cook PVI lean: negative=D, positive=R (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--weight-cap", type=float, default=None, help="Max delegation weight per delegate")
    parser.add_argument("--decay", type=float, default=0.85, help="Viscous decay alpha (default: 0.85)")
    parser.add_argument("--deleg-prob", type=float, default=0.10, help="Delegation probability per tick (default: 0.10)")
    parser.add_argument("-k", type=int, default=1, help="Delegation options k (default: 1)")
    parser.add_argument("--gamma", type=float, default=1.5, help="Preferential attachment gamma (default: 1.5)")
    parser.add_argument("--bandwagon", type=float, default=0.25, help="Bandwagon coefficient (default: 0.25)")
    parser.add_argument("--epsilon", type=float, default=None, help="Bounded confidence epsilon (default: off)")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM agents for top delegates (slow)")

    args = parser.parse_args()

    if args.interactive:
        cfg = interactive_config()
    else:
        cfg = {
            "scenario": args.scenario,
            "agents": args.agents,
            "pvi": args.pvi,
            "seed": args.seed,
            "delegation_probability_base": args.deleg_prob,
            "viscous_decay_alpha": args.decay,
            "weight_cap": args.weight_cap,
            "delegation_options_k": args.k,
            "preferential_attachment_gamma": args.gamma,
            "bandwagon_coefficient": args.bandwagon,
            "bounded_confidence_epsilon": args.epsilon,
        }

    # -- Run --
    pvi_label = f"D+{abs(cfg['pvi']):.0f}" if cfg["pvi"] < 0 else f"R+{cfg['pvi']:.0f}" if cfg["pvi"] > 0 else "EVEN"
    print(f"\n{CYAN}Running {cfg['scenario']} scenario{RESET}")
    print(f"  {cfg['agents']} agents | PVI {pvi_label} | seed {cfg['seed']}")

    start = time.time()
    output = run_scenario(cfg)
    elapsed = time.time() - start

    print(f"  {GREEN}Completed in {elapsed:.1f}s{RESET}")

    display_results(output)

    # -- Save --
    if args.output if not args.interactive else False:
        out_path = Path(args.output)
        serializable = _make_serializable(output)
        out_path.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\n  {DIM}Results saved to {out_path}{RESET}")
    elif args.interactive:
        save = input(f"\n  {DIM}Save results to JSON? (path or Enter to skip): {RESET}").strip()
        if save:
            out_path = Path(save)
            serializable = _make_serializable(output)
            out_path.write_text(json.dumps(serializable, indent=2, default=str))
            print(f"  {DIM}Saved to {out_path}{RESET}")


def _make_serializable(obj):
    """Recursively convert an object to JSON-serializable form."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
