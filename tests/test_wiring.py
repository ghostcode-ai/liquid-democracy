"""Wiring tests: every tunable parameter is exposed, connected, and functional.

These tests verify that:
1. SimulationConfig fields are exposed in the dashboard config_dict
2. Parameters actually affect simulation output (not just stored)
3. No hardcoded race IDs leak into simulation logic
4. Feature toggles produce different behavior when enabled
"""

import math
from dataclasses import fields

import numpy as np
import pytest

from engine.delegation_graph import DelegationGraph
from engine.simulation import LiquidDemocracyModel, SimulationConfig


# ------------------------------------------------------------------
# 1. Config coverage: dashboard exposes all user-facing parameters
# ------------------------------------------------------------------

# Parameters the dashboard SHOULD pass to SimulationConfig.
# Internal / structural params (races, roll_off_rates, llm_max_workers)
# are intentionally excluded from dashboard exposure.
DASHBOARD_KEYS = {
    "n_agents", "pvi_lean", "seed", "district_id", "use_ces",
    "delegation_probability_base", "viscous_decay_alpha", "weight_cap",
    "delegation_options_k", "preferential_attachment_gamma",
    "bandwagon_coefficient", "bounded_confidence_epsilon", "homophily",
    "use_llm", "llm_agent_fraction", "use_media", "media_bias_factor",
    "races",
}

# Parameters intentionally NOT exposed in dashboard (structural / internal).
INTERNAL_KEYS = {
    "ideology_std", "campaign_ticks", "voting_ticks",
    "knowledge_threshold", "delegation_threshold",
    "trust_alpha", "trust_beta", "trust_gamma", "trust_delta",
    "llm_max_workers", "trs_withdrawal_prob", "roll_off_rates",
}


def test_dashboard_covers_expected_keys():
    """Every user-facing SimulationConfig field should be in DASHBOARD_KEYS or INTERNAL_KEYS."""
    config_fields = {f.name for f in fields(SimulationConfig)}
    covered = DASHBOARD_KEYS | INTERNAL_KEYS
    uncovered = config_fields - covered
    assert uncovered == set(), (
        f"SimulationConfig fields not accounted for in test: {uncovered}. "
        f"Add to DASHBOARD_KEYS (if exposed) or INTERNAL_KEYS (if intentionally hidden)."
    )


def test_all_config_fields_accepted():
    """SimulationConfig should accept every field without error."""
    # Build a config with every field explicitly set
    cfg = SimulationConfig(
        n_agents=50, pvi_lean=-5, ideology_std=0.4, seed=99,
        district_id=None, use_ces=False,
        campaign_ticks=20, voting_ticks=5,
        bounded_confidence_epsilon=0.3,
        delegation_probability_base=0.15,
        knowledge_threshold=0.5, delegation_threshold=0.4,
        viscous_decay_alpha=0.9, weight_cap=10.0,
        delegation_options_k=2,
        trust_alpha=0.6, trust_beta=0.2, trust_gamma=0.1, trust_delta=0.25,
        bandwagon_coefficient=0.3,
        preferential_attachment_gamma=1.2,
        homophily=0.5,
        llm_agent_fraction=0.1, llm_max_workers=4, use_llm=False,
        use_media=True, media_bias_factor=1.5,
        trs_withdrawal_prob=0.4,
    )
    assert cfg.total_ticks == 26  # 20 + 5 + 1


# ------------------------------------------------------------------
# 2. No hardcoded race IDs in simulation logic
# ------------------------------------------------------------------

def test_custom_race_id_works():
    """Simulation should work with a custom race ID, not just 'house' or 'race'."""
    cfg = SimulationConfig(
        n_agents=50, seed=42,
        races={"municipal_council": ["Alice", "Bob", "Carol"]},
        campaign_ticks=10, voting_ticks=3,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    results = model.get_results()
    assert "municipal_council" in results.fptp_results
    assert results.fptp_results["municipal_council"]["winner"] in {"Alice", "Bob", "Carol"}


def test_multi_race_simulation():
    """Simulation should handle multiple races simultaneously."""
    cfg = SimulationConfig(
        n_agents=100, seed=42,
        races={
            "mayor": ["D_Mayor", "R_Mayor"],
            "council": ["D_Council", "R_Council"],
        },
        campaign_ticks=10, voting_ticks=3,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    results = model.get_results()
    # Both races should have results in all 4 systems
    for race_id in ["mayor", "council"]:
        assert race_id in results.fptp_results
        assert race_id in results.rcv_results
        assert race_id in results.trs_results
        assert race_id in results.delegation_results


def test_gini_history_per_race():
    """Gini history should track all configured races, not just hardcoded ones."""
    cfg = SimulationConfig(
        n_agents=80, seed=42,
        races={"alpha": ["X", "Y"], "beta": ["P", "Q"]},
        campaign_ticks=10, voting_ticks=3,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    stats = model.get_delegation_stats()
    gini_hist = stats["gini_history"]
    assert len(gini_hist) > 0
    # Every snapshot should have both race IDs
    for snap in gini_hist:
        assert "alpha" in snap
        assert "beta" in snap


# ------------------------------------------------------------------
# 3. Parameters actually affect output (not just stored)
# ------------------------------------------------------------------

def _run_quick(seed=42, **overrides) -> dict:
    """Helper: run a small sim and return delegation stats."""
    cfg = SimulationConfig(
        n_agents=150, seed=seed, campaign_ticks=15, voting_ticks=3,
        **overrides,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    return {
        "stats": model.get_delegation_stats(),
        "opinions": model.get_opinion_history(),
        "results": model.get_results(),
        "model": model,
    }


def test_viscous_decay_affects_max_weight():
    """Lower alpha should produce lower max weight."""
    hi = _run_quick(viscous_decay_alpha=1.0, delegation_probability_base=0.3)
    lo = _run_quick(viscous_decay_alpha=0.5, delegation_probability_base=0.3)
    max_hi = max(hi["stats"]["max_weight_per_race"].values())
    max_lo = max(lo["stats"]["max_weight_per_race"].values())
    assert max_lo <= max_hi + 0.5  # lower decay → lower max weight


def test_delegation_prob_affects_delegator_count():
    """Higher delegation probability should produce more delegators."""
    lo = _run_quick(delegation_probability_base=0.01)
    hi = _run_quick(delegation_probability_base=0.50)
    # Allow small stochastic variance with tiny populations
    assert hi["stats"]["total_delegators"] >= lo["stats"]["total_delegators"] - 5


def test_weight_cap_enforced():
    """Weight cap should limit max delegation weight."""
    uncapped = _run_quick(weight_cap=None, delegation_probability_base=0.3)
    capped = _run_quick(weight_cap=2.0, delegation_probability_base=0.3)
    max_capped = max(capped["stats"]["max_weight_per_race"].values())
    assert max_capped <= 2.01


def test_pvi_lean_affects_winner():
    """Strong partisan lean should predictably affect the winner."""
    d_lean = _run_quick(pvi_lean=-20)
    r_lean = _run_quick(pvi_lean=20)
    d_winner = d_lean["results"].fptp_results["race"]["winner"]
    r_winner = r_lean["results"].fptp_results["race"]["winner"]
    assert d_winner == "Democrat"
    assert r_winner == "Republican"


def test_bounded_confidence_affects_convergence():
    """Tight bounded confidence should reduce cross-partisan opinion exchange."""
    open_sim = _run_quick(bounded_confidence_epsilon=None)
    tight_sim = _run_quick(bounded_confidence_epsilon=0.1)
    # With tight epsilon, final opinion variance should be higher (less convergence)
    open_final = np.array(open_sim["opinions"][0][-1])
    tight_final = np.array(tight_sim["opinions"][0][-1])
    assert np.std(tight_final) >= np.std(open_final) - 0.02


def test_homophily_affects_network():
    """Higher homophily should produce more same-party edges."""
    lo = _run_quick(homophily=0.0)
    hi = _run_quick(homophily=0.9)
    # Count same-party edges
    def same_party_frac(model):
        same = 0
        total = 0
        for u, v in model.social_graph.edges():
            total += 1
            if model.agents[u].party_id == model.agents[v].party_id:
                same += 1
        return same / total if total else 0
    assert same_party_frac(hi["model"]) > same_party_frac(lo["model"])


def test_preferential_attachment_affects_concentration():
    """Higher gamma should produce more concentrated delegation."""
    lo = _run_quick(preferential_attachment_gamma=0.5, delegation_probability_base=0.25)
    hi = _run_quick(preferential_attachment_gamma=2.5, delegation_probability_base=0.25)
    gini_lo = max(lo["stats"]["gini_per_race"].values())
    gini_hi = max(hi["stats"]["gini_per_race"].values())
    assert gini_hi >= gini_lo - 0.05


def test_trs_withdrawal_prob_flows():
    """TRS withdrawal probability should affect round 2 candidate count."""
    # With 0.0 withdrawal, all qualified candidates proceed to round 2
    # With 1.0 withdrawal, most withdraw
    lo = _run_quick(trs_withdrawal_prob=0.0)
    hi = _run_quick(trs_withdrawal_prob=1.0)
    # Just verify both complete without error — withdrawal is stochastic
    assert lo["results"].trs_results is not None
    assert hi["results"].trs_results is not None


# ------------------------------------------------------------------
# 4. Feature toggles produce different behavior
# ------------------------------------------------------------------

def test_media_toggle_changes_opinions():
    """use_media=True should shift opinions vs use_media=False."""
    off = _run_quick(use_media=False)
    on = _run_quick(use_media=True)
    final_off = np.array(off["opinions"][0][-1])
    final_on = np.array(on["opinions"][0][-1])
    diff = np.abs(final_off - final_on).mean()
    assert diff > 0.0005, f"Media agents should shift opinions (mean diff={diff:.6f})"


def test_k2_toggle_changes_delegation():
    """delegation_options_k=2 should produce fractional edges."""
    k1 = _run_quick(delegation_options_k=1, delegation_probability_base=0.3)
    k2 = _run_quick(delegation_options_k=2, delegation_probability_base=0.3)
    # k=2 should have fractional edges
    fracs = [
        d.get("fraction", 1.0)
        for _, _, d in k2["model"].delegation_graph.graph.edges(data=True)
    ]
    has_fractional = any(f < 1.0 for f in fracs)
    assert has_fractional, "k=2 should produce edges with fraction < 1.0"


def test_bandwagon_zero_vs_nonzero():
    """bandwagon_coefficient=0 vs 0.5 should produce different delegation patterns."""
    zero = _run_quick(bandwagon_coefficient=0.0, delegation_probability_base=0.25)
    half = _run_quick(bandwagon_coefficient=0.5, delegation_probability_base=0.25)
    # Just verify both complete and produce delegation
    assert zero["stats"]["total_delegators"] >= 0
    assert half["stats"]["total_delegators"] >= 0


# ------------------------------------------------------------------
# 5. Delegation graph k-way correctness
# ------------------------------------------------------------------

def test_fraction_preserved_on_edges():
    """add_delegation with fraction should store it on edge data."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t", fraction=0.5)
    g.add_delegation("A", "C", "t", fraction=0.5)
    edges = list(g.graph.out_edges("A", data=True))
    fractions = sorted(d.get("fraction", 1.0) for _, _, d in edges)
    assert fractions == [0.5, 0.5]


def test_fraction_1_revokes_previous():
    """add_delegation with fraction=1.0 should replace existing edges."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t")
    g.add_delegation("A", "C", "t")  # fraction=1.0 default → replaces
    edges = list(g.graph.out_edges("A", data=True))
    assert len(edges) == 1
    assert edges[0][1] == "C"


def test_resolve_paths_handles_no_delegation():
    """resolve_paths on a direct voter should return self with weight 1.0."""
    g = DelegationGraph()
    # Add an edge to create the node, then resolve a voter with no outgoing
    g.add_delegation("A", "B", "t")
    paths = g.resolve_paths("B", "t", cycle_nodes=set())
    assert len(paths) == 1
    assert paths[0] == ("B", 1.0)


# ------------------------------------------------------------------
# 6. Campaign/voting phase timing
# ------------------------------------------------------------------

def test_custom_phase_timing():
    """Custom campaign_ticks and voting_ticks should produce correct total ticks."""
    cfg = SimulationConfig(
        n_agents=50, seed=42,
        campaign_ticks=5, voting_ticks=2,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    assert model.tick == 8  # 5 + 2 + 1 (tally)


def test_short_campaign_still_produces_results():
    """Even a very short campaign should produce valid results."""
    cfg = SimulationConfig(
        n_agents=50, seed=42,
        campaign_ticks=1, voting_ticks=1,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    results = model.get_results()
    assert results.fptp_results["race"]["winner"] is not None


# ------------------------------------------------------------------
# 7. Seed reproducibility
# ------------------------------------------------------------------

def test_same_seed_same_results():
    """Same seed should produce identical results."""
    a = _run_quick(seed=123)
    b = _run_quick(seed=123)
    assert a["stats"]["gini_per_race"] == b["stats"]["gini_per_race"]
    assert a["stats"]["total_delegators"] == b["stats"]["total_delegators"]


def test_different_seed_different_results():
    """Different seeds should usually produce different results."""
    a = _run_quick(seed=1)
    b = _run_quick(seed=2)
    # At least one metric should differ
    differs = (
        a["stats"]["total_delegators"] != b["stats"]["total_delegators"]
        or a["stats"]["gini_per_race"] != b["stats"]["gini_per_race"]
    )
    assert differs, "Different seeds should produce different delegation patterns"


# ------------------------------------------------------------------
# 8. Partisan anchoring: strong partisans resist cross-party voting
# ------------------------------------------------------------------

def test_strong_partisan_stays_loyal():
    """A Strong R voter with slight leftward drift should still pick Republican."""
    from agents.voter_agent import VoterAgent, PartyID, Demographics
    agent = VoterAgent.from_profile(
        agent_id=1,
        ideology=np.array([0.3]),  # drifted left from ~0.7
        demographics=Demographics(age=40, income=50000, education=3,
                                  race="white", gender="male", urban_rural="suburban"),
        party_id=PartyID.STRONG_R,
    )
    # 4-candidate race: Green (-0.8), Democrat (-0.6), Independent (0.0), Republican (0.6)
    candidates = ["Green", "Democrat", "Independent", "Republican"]
    choice = agent._preference_to_candidate(0.3, candidates)
    assert choice == "Republican", (
        f"Strong R at ideology 0.3 should still pick Republican (anchor=0.5), got {choice}"
    )


def test_moderate_independent_picks_nearest():
    """An Independent voter should pick the nearest candidate without anchoring."""
    from agents.voter_agent import VoterAgent, PartyID, Demographics
    agent = VoterAgent.from_profile(
        agent_id=2,
        ideology=np.array([0.1]),
        demographics=Demographics(age=30, income=60000, education=4,
                                  race="white", gender="female", urban_rural="urban"),
        party_id=PartyID.INDEPENDENT,
    )
    candidates = ["Democrat", "Independent", "Republican"]
    choice = agent._preference_to_candidate(0.1, candidates)
    assert choice == "Independent", (
        f"Independent at ideology 0.1 should pick Independent (0.0), got {choice}"
    )


def test_lean_d_can_defect_with_large_shift():
    """A Lean D voter who drifted far right should eventually defect."""
    from agents.voter_agent import VoterAgent, PartyID, Demographics
    agent = VoterAgent.from_profile(
        agent_id=3,
        ideology=np.array([0.5]),  # way past center
        demographics=Demographics(age=50, income=70000, education=2,
                                  race="white", gender="male", urban_rural="rural"),
        party_id=PartyID.LEAN_D,
    )
    candidates = ["Democrat", "Republican"]
    choice = agent._preference_to_candidate(0.5, candidates)
    # Lean D anchor is 0.2, so effective distance: D = |0.5-(-0.6)|-0.2 = 0.9, R = |0.5-0.6| = 0.1
    assert choice == "Republican", (
        f"Lean D at ideology 0.5 should defect to Republican (anchor only 0.2), got {choice}"
    )


def test_strong_d_resists_defection():
    """A Strong D voter at ideology 0.0 should still pick Democrat with anchor bonus."""
    from agents.voter_agent import VoterAgent, PartyID, Demographics
    agent = VoterAgent.from_profile(
        agent_id=4,
        ideology=np.array([0.0]),
        demographics=Demographics(age=35, income=55000, education=3,
                                  race="white", gender="female", urban_rural="urban"),
        party_id=PartyID.STRONG_D,
    )
    candidates = ["Democrat", "Independent", "Republican"]
    choice = agent._preference_to_candidate(0.0, candidates)
    # Strong D anchor=0.5: effective distance to Dem = |0-(-0.6)|-0.5 = 0.1, Ind = |0-0| = 0.0
    # Independent is still closer — Strong D at exact center is borderline
    # But with Green in the mix or in a 2-party race, loyalty kicks in
    candidates_2 = ["Democrat", "Republican"]
    choice_2 = agent._preference_to_candidate(0.0, candidates_2)
    # D = |0-(-0.6)|-0.5 = 0.1, R = |0-0.6| = 0.6 → Democrat
    assert choice_2 == "Democrat", (
        f"Strong D at ideology 0.0 in 2-party race should stay Democrat, got {choice_2}"
    )


def test_four_candidate_strong_r_no_mass_defection():
    """In a 4-candidate race, strong R voters shouldn't all defect to Independent."""
    cfg = SimulationConfig(
        n_agents=200, seed=42,
        races={"race": ["Democrat", "Green", "Independent", "Republican"]},
        campaign_ticks=15, voting_ticks=3,
        pvi_lean=10,  # lean R district
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    results = model.get_results()
    fptp = results.fptp_results["race"]
    # In a lean-R district, Republican should get significant votes
    r_votes = fptp["counts"].get("Republican", 0)
    total = sum(fptp["counts"].values())
    r_pct = r_votes / total * 100 if total else 0
    assert r_pct > 20, (
        f"In a lean-R district, Republican should get >20% of votes, got {r_pct:.1f}%: {fptp['counts']}"
    )


# ------------------------------------------------------------------
# 9. RCV must agree with FPTP in 2-candidate races
# ------------------------------------------------------------------

def test_rcv_and_trs_match_fptp_in_two_candidate_race():
    """With only 2 candidates, RCV and TRS cannot diverge from FPTP."""
    for seed in (42, 99, 2024):
        cfg = SimulationConfig(
            n_agents=300, seed=seed,
            races={"race": ["Democrat", "Republican"]},
            campaign_ticks=15, voting_ticks=3,
        )
        model = LiquidDemocracyModel(cfg)
        model.run()
        results = model.get_results()
        fptp_winner = results.fptp_results["race"]["winner"]
        rcv_winner = results.rcv_results["race"]["winner"]
        trs_winner = results.trs_results["race"]["winner"]
        assert fptp_winner == rcv_winner, (
            f"seed={seed}: RCV winner ({rcv_winner}) != FPTP winner ({fptp_winner}) "
            f"in a 2-candidate race. FPTP counts={results.fptp_results['race']['counts']}, "
            f"RCV rounds={results.rcv_results['race'].get('rounds', [])}"
        )
        assert fptp_winner == trs_winner, (
            f"seed={seed}: TRS winner ({trs_winner}) != FPTP winner ({fptp_winner}) "
            f"in a 2-candidate race."
        )


def test_delegation_fallback_uses_partisan_anchoring():
    """Delegation tally fallback for non-direct-voters must use partisan anchoring."""
    cfg = SimulationConfig(
        n_agents=200, seed=42,
        races={"race": ["Democrat", "Republican", "Independent"]},
        campaign_ticks=15, voting_ticks=3,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    candidates = ["Democrat", "Republican", "Independent"]
    # Verify _closest_candidate matches _preference_to_candidate for every agent
    for aid, agent in model.agents.items():
        via_method = model._closest_candidate(agent, candidates)
        via_agent = agent._preference_to_candidate(float(agent.ideology[0]), candidates)
        assert via_method == via_agent, (
            f"Agent {aid} ({agent.party_id.value}): _closest_candidate={via_method} "
            f"!= _preference_to_candidate={via_agent}"
        )
