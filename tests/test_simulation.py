"""Integration tests for the full simulation pipeline."""

import pytest
import numpy as np

from engine.simulation import LiquidDemocracyModel, SimulationConfig


class TestSimulationE2E:
    """End-to-end simulation tests."""

    def test_basic_run_completes(self):
        """A basic simulation should complete without errors."""
        config = SimulationConfig(n_agents=100, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        results = model.get_results()
        assert results is not None
        assert results.fptp_results is not None
        assert results.rcv_results is not None
        assert results.trs_results is not None
        assert results.delegation_results is not None

    def test_all_systems_produce_winners(self):
        """Each voting system should produce a winner for each race."""
        config = SimulationConfig(n_agents=200, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        results = model.get_results()

        for race_id in config.races:
            assert results.fptp_results[race_id]["winner"] is not None
            assert results.rcv_results[race_id]["winner"] is not None
            assert results.trs_results[race_id]["winner"] is not None
            assert results.delegation_results[race_id]["winner"] is not None

    def test_delegation_produces_nonzero_gini(self):
        """With sufficient agents and delegation probability, Gini should be > 0."""
        config = SimulationConfig(
            n_agents=300,
            seed=42,
            delegation_probability_base=0.20,
        )
        model = LiquidDemocracyModel(config)
        model.run()
        stats = model.get_delegation_stats()

        # At least one race should have non-zero Gini
        ginis = list(stats["gini_per_race"].values())
        assert any(g > 0 for g in ginis), f"Expected non-zero Gini, got {ginis}"

    def test_reproducible_seeding(self):
        """Same seed should produce identical agent populations."""
        config = SimulationConfig(n_agents=200, seed=123)

        model_a = LiquidDemocracyModel(config)
        model_b = LiquidDemocracyModel(config)

        # Agent ideologies should be identical
        for aid in model_a.agents:
            np.testing.assert_array_equal(
                model_a.agents[aid].ideology,
                model_b.agents[aid].ideology,
            )
            assert model_a.agents[aid].party_id == model_b.agents[aid].party_id

    def test_d_leaning_district_favors_democrat(self):
        """A D+20 district should elect a Democrat in most systems."""
        config = SimulationConfig(n_agents=500, pvi_lean=-20.0, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        results = model.get_results()

        # FPTP should reliably reflect the lean
        fptp_winner = results.fptp_results["race"]["winner"]
        assert fptp_winner == "Democrat", f"D+20 FPTP winner was {fptp_winner}"

    def test_r_leaning_district_favors_republican(self):
        """An R+20 district should elect a Republican."""
        config = SimulationConfig(n_agents=500, pvi_lean=20.0, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        results = model.get_results()

        fptp_winner = results.fptp_results["race"]["winner"]
        assert fptp_winner == "Republican", f"R+20 FPTP winner was {fptp_winner}"

    def test_opinion_history_captured(self):
        """Opinion snapshots should be collected during campaign."""
        config = SimulationConfig(n_agents=100, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        history = model.get_opinion_history()

        # History keyed by dimension; dimension 0 should have snapshots
        assert 0 in history
        # Should have campaign_ticks + 1 snapshots (initial + one per tick)
        assert len(history[0]) > config.campaign_ticks

    def test_weight_cap_limits_delegation_power(self):
        """Weight cap should limit maximum delegation weight."""
        config_uncapped = SimulationConfig(
            n_agents=300, seed=42,
            delegation_probability_base=0.25,
            weight_cap=None,
        )
        config_capped = SimulationConfig(
            n_agents=300, seed=42,
            delegation_probability_base=0.25,
            weight_cap=2.0,
        )

        model_uncapped = LiquidDemocracyModel(config_uncapped)
        model_uncapped.run()
        stats_uncapped = model_uncapped.get_delegation_stats()

        model_capped = LiquidDemocracyModel(config_capped)
        model_capped.run()
        stats_capped = model_capped.get_delegation_stats()

        # Capped max weight should be <= 2.0
        for race_id in config_capped.races:
            capped_max = stats_capped["max_weight_per_race"][race_id]
            assert capped_max <= 2.01, f"Capped max weight {capped_max} exceeds cap"

    def test_step_count_matches_config(self):
        """Simulation should run exactly campaign + voting + 1 tally ticks."""
        config = SimulationConfig(n_agents=50, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()

        expected_ticks = config.campaign_ticks + config.voting_ticks + 1
        assert model.tick == expected_ticks

    def test_delegation_stats_structure(self):
        """Delegation stats should have expected keys."""
        config = SimulationConfig(n_agents=100, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        stats = model.get_delegation_stats()

        assert "gini_per_race" in stats
        assert "max_weight_per_race" in stats
        assert "total_delegators" in stats
        assert "total_direct_voters" in stats
        assert "total_abstentions" in stats

    def test_voter_counts_add_up(self):
        """Delegators + direct voters + abstentions should <= n_agents * n_races."""
        config = SimulationConfig(n_agents=200, seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        stats = model.get_delegation_stats()

        total_actions = (
            stats["total_delegators"]
            + stats["total_direct_voters"]
            + stats["total_abstentions"]
        )
        max_possible = config.n_agents * len(config.races)
        assert total_actions <= max_possible

    def test_k2_reduces_max_weight(self):
        """k=2 delegation should produce lower max weight than k=1."""
        base = dict(n_agents=300, seed=42, delegation_probability_base=0.25)

        model_k1 = LiquidDemocracyModel(SimulationConfig(**base, delegation_options_k=1))
        model_k1.run()
        max_k1 = model_k1.get_delegation_stats()["max_weight_per_race"]

        model_k2 = LiquidDemocracyModel(SimulationConfig(**base, delegation_options_k=2))
        model_k2.run()
        max_k2 = model_k2.get_delegation_stats()["max_weight_per_race"]

        # k=2 should have lower or equal max weight in every race
        for race_id in max_k1:
            assert max_k2[race_id] <= max_k1[race_id] + 0.01, (
                f"k=2 max weight ({max_k2[race_id]:.2f}) should be <= "
                f"k=1 ({max_k1[race_id]:.2f}) for {race_id}"
            )

    def test_k2_produces_fractional_delegations(self):
        """k=2 should produce delegation edges with fraction=0.5."""
        config = SimulationConfig(
            n_agents=200, seed=42,
            delegation_probability_base=0.30,
            delegation_options_k=2,
        )
        model = LiquidDemocracyModel(config)
        model.run()

        # Check that at least some edges have fraction < 1.0
        fractions = []
        for _, _, data in model.delegation_graph.graph.edges(data=True):
            fractions.append(data.get("fraction", 1.0))
        fractional = [f for f in fractions if f < 1.0]
        assert len(fractional) > 0, "k=2 should produce fractional delegation edges"

    def test_bandwagon_increases_concentration(self):
        """Higher bandwagon coefficient should increase power concentration."""
        base = dict(n_agents=300, seed=42, delegation_probability_base=0.20)

        model_lo = LiquidDemocracyModel(SimulationConfig(**base, bandwagon_coefficient=0.0))
        model_lo.run()
        gini_lo = model_lo.get_delegation_stats()["gini_per_race"]

        model_hi = LiquidDemocracyModel(SimulationConfig(**base, bandwagon_coefficient=1.0))
        model_hi.run()
        gini_hi = model_hi.get_delegation_stats()["gini_per_race"]

        # With strong bandwagon, Gini should be >= no-bandwagon case
        for race_id in gini_lo:
            assert gini_hi[race_id] >= gini_lo[race_id] - 0.05, (
                f"bandwagon=1.0 Gini ({gini_hi[race_id]:.3f}) should be >= "
                f"bandwagon=0 ({gini_lo[race_id]:.3f}) for {race_id}"
            )

    def test_media_agents_shift_opinions(self):
        """Media agents should cause opinion trajectories to differ from baseline."""
        base = dict(n_agents=200, seed=42)

        model_no = LiquidDemocracyModel(SimulationConfig(**base, use_media=False))
        model_no.run()
        hist_no = model_no.get_opinion_history()

        model_yes = LiquidDemocracyModel(SimulationConfig(**base, use_media=True))
        model_yes.run()
        hist_yes = model_yes.get_opinion_history()

        # Opinion histories should differ when media is enabled
        final_no = np.array(hist_no[0][-1])
        final_yes = np.array(hist_yes[0][-1])
        diff = np.abs(final_no - final_yes).mean()
        assert diff > 0.001, (
            f"Media agents should shift opinions (mean diff={diff:.6f})"
        )

    def test_media_agents_complete_successfully(self):
        """Simulation with media agents should complete without errors."""
        config = SimulationConfig(n_agents=100, seed=42, use_media=True)
        model = LiquidDemocracyModel(config)
        model.run()
        results = model.get_results()
        assert results is not None
        assert results.fptp_results is not None
