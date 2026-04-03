"""Tests for the synthetic agent seeding engine."""

import networkx as nx
import numpy as np
import pytest

from agents.voter_agent import PartyID
from engine.seeding import (
    build_social_network,
    pvi_to_party_probs,
    seed_agents,
)


class TestPVIConversion:
    def test_even_district(self):
        """PVI=0 should give roughly balanced party distribution."""
        probs = pvi_to_party_probs(0.0)
        d_total = probs[PartyID.STRONG_D] + probs[PartyID.LEAN_D]
        r_total = probs[PartyID.STRONG_R] + probs[PartyID.LEAN_R]
        assert abs(d_total - r_total) < 0.05

    def test_strong_d_district(self):
        """D+20 district should have more Democrats."""
        probs = pvi_to_party_probs(-20.0)
        d_total = probs[PartyID.STRONG_D] + probs[PartyID.LEAN_D]
        r_total = probs[PartyID.STRONG_R] + probs[PartyID.LEAN_R]
        assert d_total > r_total

    def test_strong_r_district(self):
        """R+20 district should have more Republicans."""
        probs = pvi_to_party_probs(20.0)
        d_total = probs[PartyID.STRONG_D] + probs[PartyID.LEAN_D]
        r_total = probs[PartyID.STRONG_R] + probs[PartyID.LEAN_R]
        assert r_total > d_total

    def test_probs_sum_to_one(self):
        """Party probabilities must sum to 1."""
        for pvi in [-30, -10, 0, 10, 30]:
            probs = pvi_to_party_probs(pvi)
            assert abs(sum(probs.values()) - 1.0) < 1e-10


class TestSeedAgents:
    def test_correct_count(self):
        """Should generate exactly n_agents."""
        agents = seed_agents(n_agents=100, seed=42)
        assert len(agents) == 100

    def test_reproducible_with_seed(self):
        """Same seed should produce same agents."""
        agents_a = seed_agents(n_agents=50, seed=123)
        agents_b = seed_agents(n_agents=50, seed=123)

        for aid in agents_a:
            np.testing.assert_array_equal(
                agents_a[aid].ideology, agents_b[aid].ideology
            )
            assert agents_a[aid].party_id == agents_b[aid].party_id

    def test_d_district_skews_democratic(self):
        """D+15 district should have more Democrats than Republicans."""
        agents = seed_agents(n_agents=1000, pvi_lean=-15.0, seed=42)
        d_count = sum(
            1 for a in agents.values()
            if a.party_id in (PartyID.STRONG_D, PartyID.LEAN_D)
        )
        r_count = sum(
            1 for a in agents.values()
            if a.party_id in (PartyID.STRONG_R, PartyID.LEAN_R)
        )
        assert d_count > r_count

    def test_ideology_correlated_with_party(self):
        """Strong D agents should have negative ideology mean; strong R positive."""
        agents = seed_agents(n_agents=2000, seed=42)

        d_ideologies = [
            a.ideology[0] for a in agents.values() if a.party_id == PartyID.STRONG_D
        ]
        r_ideologies = [
            a.ideology[0] for a in agents.values() if a.party_id == PartyID.STRONG_R
        ]

        if d_ideologies and r_ideologies:
            assert np.mean(d_ideologies) < 0
            assert np.mean(r_ideologies) > 0

    def test_demographics_plausible(self):
        """All agents should have plausible demographic values."""
        agents = seed_agents(n_agents=500, seed=42)
        for agent in agents.values():
            assert 18 <= agent.demographics.age <= 95
            assert agent.demographics.income > 0
            assert 0 <= agent.demographics.education <= 5
            assert agent.demographics.gender in ("male", "female")
            assert agent.demographics.urban_rural in ("urban", "suburban", "rural")

    def test_behavioral_params_derived(self):
        """Behavioral params should be derived from demographics."""
        agents = seed_agents(n_agents=100, seed=42)
        for agent in agents.values():
            assert 0 <= agent.political_knowledge <= 1
            assert 0 <= agent.engagement_level <= 1
            assert 0 <= agent.stubbornness <= 1


class TestSocialNetwork:
    def test_network_has_all_agents(self):
        """Social network should include all agents as nodes."""
        agents = seed_agents(n_agents=100, seed=42)
        G = build_social_network(agents, rng=np.random.default_rng(42))
        assert set(G.nodes()) == set(agents.keys())

    def test_network_connected(self):
        """Social network should be mostly connected."""
        agents = seed_agents(n_agents=200, seed=42)
        G = build_social_network(agents, rng=np.random.default_rng(42))
        # Watts-Strogatz with k=8 should be connected for n=200
        largest_cc = max(nx.connected_components(G), key=len)
        assert len(largest_cc) > 0.9 * len(agents)

    def test_homophily_increases_same_party_edges(self):
        """Homophily rewiring should increase same-party connections."""
        agents = seed_agents(n_agents=200, seed=42)

        # Build with and without homophily
        G_no_homophily = build_social_network(agents, homophily=0.0, rng=np.random.default_rng(42))
        G_homophily = build_social_network(agents, homophily=0.65, rng=np.random.default_rng(42))

        def same_party_fraction(G):
            same = 0
            total = 0
            for u, v in G.edges():
                total += 1
                u_lean = agents[u].party_id.value[:4]  # "stro" or "lean" or "inde"
                v_lean = agents[v].party_id.value[:4]
                # Simplified: same D/R bucket
                u_d = agents[u].party_id in (PartyID.STRONG_D, PartyID.LEAN_D)
                v_d = agents[v].party_id in (PartyID.STRONG_D, PartyID.LEAN_D)
                if u_d == v_d:
                    same += 1
            return same / total if total > 0 else 0

        assert same_party_fraction(G_homophily) > same_party_fraction(G_no_homophily)

    def test_neighbors_assigned_to_agents(self):
        """After building network, agents should have neighbor lists."""
        agents = seed_agents(n_agents=50, seed=42)
        build_social_network(agents, rng=np.random.default_rng(42))
        agents_with_neighbors = sum(1 for a in agents.values() if len(a.neighbors) > 0)
        assert agents_with_neighbors > 0.9 * len(agents)


class TestRealDataSeeding:
    """Tests for seeding from real 2024 district profiles."""

    def test_seed_from_known_district(self):
        """Seeding from a real district should produce agents."""
        from engine.seeding import seed_from_district
        agents = seed_from_district("CA-12", n_agents=100, seed=42)
        assert len(agents) == 100

    def test_safe_d_district_skews_democrat(self):
        """CA-12 (D+28) should produce mostly Democrats."""
        from engine.seeding import seed_from_district
        agents = seed_from_district("CA-12", n_agents=500, seed=42)
        d_count = sum(
            1 for a in agents.values()
            if a.party_id in (PartyID.STRONG_D, PartyID.LEAN_D)
        )
        assert d_count > len(agents) * 0.5

    def test_safe_r_district_skews_republican(self):
        """TX-13 (R+28) should produce mostly Republicans."""
        from engine.seeding import seed_from_district
        agents = seed_from_district("TX-13", n_agents=500, seed=42)
        r_count = sum(
            1 for a in agents.values()
            if a.party_id in (PartyID.STRONG_R, PartyID.LEAN_R)
        )
        assert r_count > len(agents) * 0.5

    def test_demographics_reflect_district(self):
        """NY-14 (Bronx/Queens) should produce high Hispanic fraction."""
        from engine.seeding import seed_from_district
        agents = seed_from_district("NY-14", n_agents=1000, seed=42)
        hispanic_count = sum(
            1 for a in agents.values()
            if a.demographics.race == "hispanic"
        )
        # NY-14 is 50% Hispanic
        assert hispanic_count > len(agents) * 0.3

    def test_unknown_district_raises(self):
        """Unknown district ID should raise ValueError."""
        from engine.seeding import seed_from_district
        with pytest.raises(ValueError, match="Unknown district"):
            seed_from_district("XX-99", n_agents=100)

    def test_simulation_with_real_district(self):
        """Full simulation with real district data should complete."""
        from engine.simulation import LiquidDemocracyModel, SimulationConfig
        config = SimulationConfig(n_agents=100, district_id="PA-07", seed=42)
        model = LiquidDemocracyModel(config)
        model.run()
        results = model.get_results()
        assert results.fptp_results["race"]["winner"] is not None
