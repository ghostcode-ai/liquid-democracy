"""Dashboard contract tests: verify every model attribute and dict key the dashboard touches.

These tests catch the class of bugs where dashboard code accesses an attribute
that doesn't exist (typo, renamed, moved) or expects a dict key the engine
doesn't return. A small simulation is run once; then each test checks a specific
attribute path or dict shape that dashboard/app.py relies on.
"""

from __future__ import annotations

import pytest

from engine.simulation import LiquidDemocracyModel, SimulationConfig, SimulationResults
from agents.voter_agent import RaceState, VoterAgent


# ---------------------------------------------------------------------------
# Shared fixture: run one small sim that all tests read from
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim():
    """Run a minimal simulation and return model + results."""
    cfg = SimulationConfig(
        n_agents=100,
        seed=42,
        campaign_ticks=15,
        voting_ticks=3,
        delegation_probability_base=0.25,
    )
    model = LiquidDemocracyModel(cfg)
    model.run()
    results = model.get_results()
    return {"model": model, "results": results, "cfg": cfg}


# ---------------------------------------------------------------------------
# 1. Model public methods exist and return without error
# ---------------------------------------------------------------------------

class TestModelMethods:

    def test_get_results(self, sim):
        r = sim["model"].get_results()
        assert isinstance(r, SimulationResults)

    def test_get_delegation_stats(self, sim):
        s = sim["model"].get_delegation_stats()
        assert isinstance(s, dict)

    def test_get_opinion_history(self, sim):
        h = sim["model"].get_opinion_history()
        assert isinstance(h, dict)
        assert 0 in h  # dimension 0

    def test_get_dynamics_report(self, sim):
        d = sim["model"].get_dynamics_report()
        assert isinstance(d, dict)

    def test_get_agent_summary(self, sim):
        s = sim["model"].get_agent_summary()
        assert isinstance(s, dict)

    def test_get_llm_vs_rule_stats(self, sim):
        s = sim["model"].get_llm_vs_rule_stats()
        assert isinstance(s, dict)

    def test_get_influence_summary(self, sim):
        s = sim["model"].get_influence_summary()
        assert isinstance(s, dict)
        assert "peer_total" in s
        assert "media_total" in s
        assert "per_tick" in s


# ---------------------------------------------------------------------------
# 2. Model internal attributes the dashboard reads directly
# ---------------------------------------------------------------------------

class TestModelAttributes:

    def test_agents_dict(self, sim):
        m = sim["model"]
        assert isinstance(m.agents, dict)
        assert len(m.agents) > 0

    def test_config(self, sim):
        m = sim["model"]
        assert hasattr(m.config, "races")
        assert isinstance(m.config.races, dict)

    def test_delegation_graph(self, sim):
        m = sim["model"]
        assert hasattr(m, "delegation_graph")

    def test_initial_ideologies(self, sim):
        m = sim["model"]
        assert isinstance(m._initial_ideologies, dict)
        assert len(m._initial_ideologies) == len(m.agents)

    def test_initial_parties(self, sim):
        m = sim["model"]
        assert isinstance(m._initial_parties, dict)
        assert len(m._initial_parties) == len(m.agents)

    def test_direct_votes(self, sim):
        m = sim["model"]
        assert isinstance(m._direct_votes, dict)
        for race_id in m.config.races:
            assert race_id in m._direct_votes

    def test_delegation_records(self, sim):
        m = sim["model"]
        assert isinstance(m._delegation_records, dict)
        for race_id in m.config.races:
            assert race_id in m._delegation_records

    def test_opinion_deltas(self, sim):
        m = sim["model"]
        assert isinstance(m._opinion_deltas, list)

    def test_social_graph(self, sim):
        m = sim["model"]
        assert hasattr(m, "social_graph")

    def test_influence_attribution(self, sim):
        m = sim["model"]
        assert isinstance(m._influence_attribution, list)
        assert len(m._influence_attribution) == m.config.campaign_ticks
        if m._influence_attribution:
            entry = m._influence_attribution[0]
            assert "peer" in entry
            assert "media" in entry
            assert "total" in entry

    def test_decision_tuple_has_reason(self, sim):
        m = sim["model"]
        for race_id in m.config.races:
            all_decs = m._llm_decisions[race_id] + m._rule_decisions[race_id]
            if all_decs:
                sample = all_decs[0]
                assert len(sample) == 4, (
                    f"Decision tuple should be (aid, action, choice, reason), got {len(sample)} elements"
                )
                assert isinstance(sample[3], str), "Fourth element (reason) should be a string"

    def test_decision_tuples_accessible_by_index(self, sim):
        """Dashboard accesses decision tuples via dec[0], dec[1], dec[2] to
        tolerate both 3-element (stale) and 4-element (current) tuples.
        Verify the first 3 elements are always (int, str, str)."""
        m = sim["model"]
        for race_id in m.config.races:
            for dec in m._rule_decisions[race_id]:
                assert isinstance(dec[0], int), "dec[0] (agent_id) must be int"
                assert isinstance(dec[1], str), "dec[1] (action) must be str"
                assert isinstance(dec[2], str), "dec[2] (choice) must be str"


# ---------------------------------------------------------------------------
# 3. Agent and RaceState attributes the dashboard accesses
# ---------------------------------------------------------------------------

class TestAgentAttributes:

    def test_agent_has_expected_fields(self, sim):
        agent = next(iter(sim["model"].agents.values()))
        # Fields the dashboard reads directly
        assert hasattr(agent, "ideology")
        assert hasattr(agent, "party_id")
        assert hasattr(agent, "race_states")
        assert hasattr(agent, "is_llm_agent")
        assert hasattr(agent, "neighbors")
        assert hasattr(agent, "trust_scores")
        assert hasattr(agent, "engagement_level")
        assert hasattr(agent, "agent_id")

    def test_race_state_has_expected_fields(self, sim):
        agent = next(iter(sim["model"].agents.values()))
        rs = next(iter(agent.race_states.values()))
        # Fields the dashboard reads — note: delegation_targets (plural, list)
        assert hasattr(rs, "delegation_targets")
        assert isinstance(rs.delegation_targets, list)
        assert hasattr(rs, "vote_choice")
        assert hasattr(rs, "voted")
        assert hasattr(rs, "knowledge_level")
        assert hasattr(rs, "preference")
        assert hasattr(rs, "ranked_choices")
        assert isinstance(rs.ranked_choices, list)

    def test_no_singular_delegation_target(self, sim):
        """Dashboard previously used rs.delegation_target (singular) — must not exist."""
        agent = next(iter(sim["model"].agents.values()))
        rs = next(iter(agent.race_states.values()))
        assert not hasattr(rs, "delegation_target"), (
            "RaceState should have 'delegation_targets' (plural list), not 'delegation_target'"
        )


# ---------------------------------------------------------------------------
# 4. SimulationResults shape: keys the dashboard destructures
# ---------------------------------------------------------------------------

class TestResultsShape:

    def test_result_fields(self, sim):
        r = sim["results"]
        assert hasattr(r, "fptp_results")
        assert hasattr(r, "rcv_results")
        assert hasattr(r, "trs_results")
        assert hasattr(r, "delegation_results")
        assert hasattr(r, "delegation_stats")
        assert hasattr(r, "opinion_history")

    def test_per_race_results_present(self, sim):
        r = sim["results"]
        for race_id in sim["cfg"].races:
            assert race_id in r.fptp_results
            assert race_id in r.rcv_results
            assert race_id in r.trs_results
            assert race_id in r.delegation_results

    def test_fptp_result_keys(self, sim):
        r = sim["results"]
        race_id = next(iter(sim["cfg"].races))
        fptp = r.fptp_results[race_id]
        assert "winner" in fptp
        assert "total_votes" in fptp

    def test_delegation_result_keys(self, sim):
        r = sim["results"]
        race_id = next(iter(sim["cfg"].races))
        dr = r.delegation_results[race_id]
        assert "winner" in dr
        assert "total_effective_votes" in dr


# ---------------------------------------------------------------------------
# 4b. _get_counts returns non-empty counts for every voting system
# ---------------------------------------------------------------------------

class TestGetCountsAllSystems:
    """Verify _get_counts extracts vote counts from every system's result format.

    This catches the bug where RCV results (keyed by 'rounds') or TRS round-2
    results (keyed by 'round2_counts') were invisible to the dashboard pie charts.
    """

    def test_fptp_counts(self, sim):
        """FPTP stores counts under 'counts'."""
        race_id = next(iter(sim["cfg"].races))
        fptp = sim["results"].fptp_results[race_id]
        assert "counts" in fptp
        assert fptp["counts"], "FPTP counts should not be empty"

    def test_rcv_has_rounds(self, sim):
        """RCV stores per-round counts under 'rounds'."""
        race_id = next(iter(sim["cfg"].races))
        rcv = sim["results"].rcv_results[race_id]
        assert "rounds" in rcv
        assert isinstance(rcv["rounds"], list)
        if rcv["rounds"]:
            final_round = rcv["rounds"][-1]
            assert isinstance(final_round, dict)
            assert final_round, "RCV final round counts should not be empty"

    def test_trs_has_round1_counts(self, sim):
        """TRS always has round1_counts."""
        race_id = next(iter(sim["cfg"].races))
        trs = sim["results"].trs_results[race_id]
        assert "round1_counts" in trs
        assert trs["round1_counts"], "TRS round1_counts should not be empty"

    def test_trs_round2_when_runoff(self, sim):
        """If TRS decided in round 2, round2_counts must be present."""
        race_id = next(iter(sim["cfg"].races))
        trs = sim["results"].trs_results[race_id]
        if trs.get("decided_in") == 2:
            assert trs.get("round2_counts"), "TRS round 2 decided but round2_counts empty"

    def test_delegation_weighted_counts(self, sim):
        """Delegation stores counts under 'weighted_counts'."""
        race_id = next(iter(sim["cfg"].races))
        dr = sim["results"].delegation_results[race_id]
        assert "weighted_counts" in dr
        assert dr["weighted_counts"], "Delegation weighted_counts should not be empty"

    def test_get_counts_returns_nonempty_for_all_systems(self, sim):
        """_get_counts must return a non-empty dict for every system."""
        # Import the function from the dashboard module — since it's in a
        # Streamlit app we replicate the logic here to avoid top-level imports.
        def _get_counts(result: dict) -> dict:
            if result.get("round2_counts"):
                return result["round2_counts"]
            rounds = result.get("rounds")
            if rounds:
                return rounds[-1]
            for key in ("counts", "weighted_counts", "round1_counts"):
                if key in result:
                    return result[key]
            return {}

        race_id = next(iter(sim["cfg"].races))
        r = sim["results"]
        for label, result in [
            ("FPTP", r.fptp_results[race_id]),
            ("RCV", r.rcv_results[race_id]),
            ("TRS", r.trs_results[race_id]),
            ("Delegation", r.delegation_results[race_id]),
        ]:
            counts = _get_counts(result)
            assert counts, f"_get_counts returned empty for {label}: keys={list(result.keys())}"
            assert sum(counts.values()) > 0, f"_get_counts for {label} has zero total votes"


# ---------------------------------------------------------------------------
# 5. Delegation stats dict shape
# ---------------------------------------------------------------------------

class TestDelegationStatsShape:

    def test_required_keys(self, sim):
        s = sim["model"].get_delegation_stats()
        required = {
            "gini_per_race", "max_weight_per_race", "total_delegators",
            "gini_history", "chain_count_per_race", "avg_chain_length_per_race",
        }
        missing = required - set(s.keys())
        assert missing == set(), f"Delegation stats missing keys: {missing}"

    def test_gini_history_structure(self, sim):
        s = sim["model"].get_delegation_stats()
        gh = s["gini_history"]
        assert isinstance(gh, list)
        if gh:
            # Each snapshot should be a dict keyed by race_id
            for race_id in sim["cfg"].races:
                assert race_id in gh[0]


# ---------------------------------------------------------------------------
# 6. Dynamics report dict shape
# ---------------------------------------------------------------------------

class TestDynamicsReportShape:

    def test_required_keys(self, sim):
        d = sim["model"].get_dynamics_report()
        required = {
            "convergence_tick", "converged", "opinion_deltas",
            "total_cross_party_shifts", "cross_party_pct",
            "total_stance_changes", "shift_by_group",
            "delegation_turnout_boost",
        }
        missing = required - set(d.keys())
        assert missing == set(), f"Dynamics report missing keys: {missing}"

    def test_shift_by_group_structure(self, sim):
        d = sim["model"].get_dynamics_report()
        for group, data in d["shift_by_group"].items():
            assert "total" in data
            assert "stayed" in data
            assert "shifted_right" in data
            assert "shifted_left" in data
            assert "crossed_party" in data
            assert "mean_shift" in data


# ---------------------------------------------------------------------------
# 7. Delegation graph methods the dashboard calls
# ---------------------------------------------------------------------------

class TestDelegationGraphMethods:

    def test_resolve_all(self, sim):
        m = sim["model"]
        race_id = next(iter(m.config.races))
        weights = m.delegation_graph.resolve_all(race_id)
        assert isinstance(weights, dict)

    def test_get_delegators(self, sim):
        m = sim["model"]
        race_id = next(iter(m.config.races))
        agent_id = str(next(iter(m.agents)))
        delegators = m.delegation_graph.get_delegators(agent_id, race_id)
        assert isinstance(delegators, (list, set))

    def test_get_chain_length(self, sim):
        m = sim["model"]
        race_id = next(iter(m.config.races))
        agent_id = str(next(iter(m.agents)))
        length = m.delegation_graph.get_chain_length(agent_id, race_id)
        assert isinstance(length, (int, float))


# ---------------------------------------------------------------------------
# 8. Campaign-phase progress: delegation_targets populated during campaign
# ---------------------------------------------------------------------------

class TestCampaignPhaseState:

    def test_delegation_targets_populated_during_campaign(self):
        """After campaign ticks, some agents should have delegation_targets set."""
        cfg = SimulationConfig(
            n_agents=100, seed=42,
            campaign_ticks=15, voting_ticks=0,
            delegation_probability_base=0.3,
        )
        model = LiquidDemocracyModel(cfg)
        for _ in range(cfg.campaign_ticks):
            model.step()

        n_intending = sum(
            1 for a in model.agents.values()
            if any(s.delegation_targets for s in a.race_states.values())
        )
        assert n_intending > 0, (
            "After campaign phase, some agents should have delegation_targets set"
        )


# ---------------------------------------------------------------------------
# 9. Campaign trail summary: model attributes that build_campaign_summary reads
# ---------------------------------------------------------------------------

class TestCampaignSummaryContract:
    """Verify all model internals that build_campaign_summary() accesses."""

    def test_influence_summary_guarded_by_hasattr(self, sim):
        """Dashboard guards get_influence_summary with hasattr for stale models."""
        m = sim["model"]
        assert hasattr(m, "get_influence_summary"), (
            "Fresh model must have get_influence_summary — "
            "dashboard uses hasattr guard for stale session state"
        )

    def test_opinion_snapshots(self, sim):
        m = sim["model"]
        assert isinstance(m._opinion_snapshots, list)
        assert len(m._opinion_snapshots) > 0
        # Each snapshot is an ndarray indexed by agent position
        assert m._opinion_snapshots[0].shape[0] == len(m.agents)

    def test_initial_parties(self, sim):
        m = sim["model"]
        assert isinstance(m._initial_parties, dict)
        assert len(m._initial_parties) == len(m.agents)
        # Values should be party ID strings
        sample = next(iter(m._initial_parties.values()))
        assert isinstance(sample, str)

    def test_convergence_tick(self, sim):
        m = sim["model"]
        assert hasattr(m, "_convergence_tick")
        assert m._convergence_tick is None or isinstance(m._convergence_tick, int)

    def test_initial_ideologies(self, sim):
        m = sim["model"]
        assert isinstance(m._initial_ideologies, dict)
        assert len(m._initial_ideologies) == len(m.agents)
        sample = next(iter(m._initial_ideologies.values()))
        assert isinstance(sample, float)

    def test_progress_callback(self, sim):
        m = sim["model"]
        assert hasattr(m, "progress_callback")

    def test_trust_scores_accessible(self, sim):
        """build_campaign_summary aggregates agent.trust_scores."""
        agent = next(iter(sim["model"].agents.values()))
        assert hasattr(agent, "trust_scores")
        assert isinstance(agent.trust_scores, dict)

    def test_opinion_snapshots_indexable_by_party(self, sim):
        """Convergence chart indexes snapshots by party group indices."""
        m = sim["model"]
        agent_ids = sorted(m.agents.keys())
        # Build party indices the same way the chart does
        party_indices: dict[str, list[int]] = {}
        for idx, aid in enumerate(agent_ids):
            party = m._initial_parties.get(aid, "independent")
            party_indices.setdefault(party, []).append(idx)
        # Each snapshot should be indexable by these indices
        snap = m._opinion_snapshots[0]
        for party, indices in party_indices.items():
            subset = snap[indices]
            assert len(subset) == len(indices)


# ---------------------------------------------------------------------------
# 10. Influence summary shape
# ---------------------------------------------------------------------------

class TestInfluenceSummaryShape:

    def test_required_keys(self, sim):
        s = sim["model"].get_influence_summary()
        required = {"peer_total", "media_total", "peer_pct", "media_pct", "per_tick"}
        missing = required - set(s.keys())
        assert missing == set(), f"Influence summary missing keys: {missing}"

    def test_per_tick_entries_shape(self, sim):
        s = sim["model"].get_influence_summary()
        assert len(s["per_tick"]) == sim["cfg"].campaign_ticks
        for entry in s["per_tick"]:
            assert "peer" in entry
            assert "media" in entry
            assert "total" in entry

    def test_percentages_sum_to_100(self, sim):
        s = sim["model"].get_influence_summary()
        if s["peer_total"] > 0 or s["media_total"] > 0:
            assert abs(s["peer_pct"] + s["media_pct"] - 100.0) < 0.2


# ---------------------------------------------------------------------------
# 11. LLM stats top_reasons key
# ---------------------------------------------------------------------------

class TestLLMStatsShape:

    def test_top_reasons_in_summarize(self, sim):
        """get_llm_vs_rule_stats summaries should include top_reasons."""
        s = sim["model"].get_llm_vs_rule_stats()
        # Even without LLM enabled, per-race summaries should have the key
        for race_id in sim["cfg"].races:
            if race_id in s:
                for group in ("llm", "rule"):
                    if group in s[race_id]:
                        assert "top_reasons" in s[race_id][group]
