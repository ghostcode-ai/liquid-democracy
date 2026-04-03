"""Tests for Friedkin-Johnsen opinion dynamics and trust formation."""

import networkx as nx
import numpy as np
import pytest

from agents.voter_agent import Demographics, PartyID, VoterAgent
from engine.opinion_dynamics import (
    build_influence_matrix,
    build_stubbornness_matrix,
    extract_opinions,
    friedkin_johnsen_step,
    run_opinion_dynamics,
)
from engine.trust import (
    compute_agreement,
    compute_social_proof,
    update_trust_single,
    update_all_trust,
)


def make_agent(agent_id: int, ideology_0: float, party: PartyID, stubbornness: float = 0.5) -> VoterAgent:
    """Helper to create a minimal agent for testing."""
    demo = Demographics(age=40, income=60000, education=3, race="white", gender="male", urban_rural="urban")
    ideology = np.zeros(10)
    ideology[0] = ideology_0
    agent = VoterAgent(
        agent_id=agent_id,
        demographics=demo,
        ideology=ideology,
        party_id=party,
        stubbornness=stubbornness,
    )
    return agent


class TestInfluenceMatrix:
    def test_row_stochastic(self):
        """Influence matrix rows must sum to 1."""
        agents = {
            0: make_agent(0, -0.5, PartyID.LEAN_D),
            1: make_agent(1, -0.3, PartyID.LEAN_D),
            2: make_agent(2, 0.5, PartyID.LEAN_R),
        }
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])

        W = build_influence_matrix(agents, G)
        for row in W:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_isolated_node_self_loop(self):
        """Isolated node should have self-loop (W[i,i] = 1)."""
        agents = {
            0: make_agent(0, 0.0, PartyID.INDEPENDENT),
            1: make_agent(1, 0.5, PartyID.LEAN_R),
        }
        G = nx.Graph()
        G.add_node(0)
        G.add_node(1)
        # No edges

        W = build_influence_matrix(agents, G)
        assert W[0, 0] == 1.0
        assert W[1, 1] == 1.0

    def test_bounded_confidence_filters(self):
        """Agents far apart in opinion should not influence each other."""
        agents = {
            0: make_agent(0, -0.9, PartyID.STRONG_D),
            1: make_agent(1, -0.8, PartyID.STRONG_D),
            2: make_agent(2, 0.9, PartyID.STRONG_R),
        }
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2)])

        # With epsilon=0.5, agent 0 (-0.9) should not be influenced by agent 2 (0.9)
        W = build_influence_matrix(agents, G, dimension=0, epsilon=0.5)
        # Agent 0 should only be influenced by agent 1
        assert W[0, 2] == 0.0
        assert W[0, 1] > 0.0


class TestFriedkinJohnsen:
    def test_fully_stubborn_no_change(self):
        """Agent with stubbornness=1.0 should never change opinion."""
        agents = {
            0: make_agent(0, -0.8, PartyID.STRONG_D, stubbornness=1.0),
            1: make_agent(1, 0.8, PartyID.STRONG_R, stubbornness=1.0),
        }
        G = nx.Graph()
        G.add_edge(0, 1)

        x_initial = extract_opinions(agents, 0)
        Lambda = build_stubbornness_matrix(agents)
        W = build_influence_matrix(agents, G, 0)

        x = x_initial.copy()
        for _ in range(50):
            x = friedkin_johnsen_step(x, x_initial, Lambda, W)

        np.testing.assert_array_almost_equal(x, x_initial)

    def test_zero_stubbornness_converges_to_average(self):
        """Agents with stubbornness=0 should converge toward each other."""
        agents = {
            0: make_agent(0, -0.6, PartyID.LEAN_D, stubbornness=0.0),
            1: make_agent(1, 0.6, PartyID.LEAN_R, stubbornness=0.0),
        }
        G = nx.Graph()
        G.add_edge(0, 1)

        x_initial = extract_opinions(agents, 0)
        Lambda = build_stubbornness_matrix(agents)
        W = build_influence_matrix(agents, G, 0)

        x = x_initial.copy()
        for _ in range(100):
            x = friedkin_johnsen_step(x, x_initial, Lambda, W)

        # Both should converge to approximately 0 (the average)
        assert abs(x[0] - x[1]) < 0.01

    def test_convergence_with_mixed_stubbornness(self):
        """High-stubbornness agents anchor; low-stubbornness agents move toward them."""
        agents = {
            0: make_agent(0, -0.8, PartyID.STRONG_D, stubbornness=0.9),
            1: make_agent(1, 0.0, PartyID.INDEPENDENT, stubbornness=0.1),
            2: make_agent(2, 0.8, PartyID.STRONG_R, stubbornness=0.9),
        }
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        history = run_opinion_dynamics(agents, G, dimension=0, n_steps=100)

        # Stubborn agents should barely move
        assert abs(history[-1][0] - history[0][0]) < 0.1
        assert abs(history[-1][2] - history[0][2]) < 0.1
        # Independent should be pulled somewhere between
        assert -0.8 < history[-1][1] < 0.8

    def test_opinions_stay_bounded(self):
        """Opinions should remain in [-1, 1] after dynamics."""
        agents = {
            i: make_agent(i, np.random.uniform(-1, 1), PartyID.INDEPENDENT, stubbornness=0.3)
            for i in range(20)
        }
        G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)

        history = run_opinion_dynamics(agents, G, dimension=0, n_steps=50)

        for state in history:
            assert np.all(state >= -1.0)
            assert np.all(state <= 1.0)


class TestTrustFormation:
    def test_agreement_identical_ideology(self):
        """Identical ideology → agreement = 1.0."""
        a = make_agent(0, 0.5, PartyID.LEAN_R)
        b = make_agent(1, 0.5, PartyID.LEAN_R)
        # Set full ideology identical
        b.ideology = a.ideology.copy()
        assert compute_agreement(a, b) == pytest.approx(1.0, abs=0.01)

    def test_agreement_opposite_ideology(self):
        """Opposite ideology → agreement close to 0."""
        a = make_agent(0, -0.8, PartyID.STRONG_D)
        a.ideology = np.full(10, -0.8)
        b = make_agent(1, 0.8, PartyID.STRONG_R)
        b.ideology = np.full(10, 0.8)
        agreement = compute_agreement(a, b)
        assert agreement < 0.2

    def test_social_proof_scales_with_trust(self):
        """Agent trusted by many → higher social proof."""
        agents = {i: make_agent(i, 0.0, PartyID.INDEPENDENT) for i in range(10)}
        # Make everyone trust agent 0
        for i in range(1, 10):
            agents[i].trust_scores[0] = 0.8

        proof = compute_social_proof(agents[0], agents)
        assert proof > 0.5

    def test_trust_update_increases_with_agreement(self):
        """High agreement should increase trust."""
        agents = {
            0: make_agent(0, -0.5, PartyID.LEAN_D),
            1: make_agent(1, -0.5, PartyID.LEAN_D),
        }
        agents[1].ideology = agents[0].ideology.copy()  # identical

        trust = update_trust_single(agents[0], agents[1], agents, betrayal_events={})
        # Initial trust 0.3, agreement high, should increase
        assert trust > 0.3

    def test_betrayal_decreases_trust(self):
        """Betrayal event should decrease trust."""
        agents = {
            0: make_agent(0, -0.5, PartyID.LEAN_D),
            1: make_agent(1, -0.5, PartyID.LEAN_D),
        }
        agents[0].trust_scores[1] = 0.8

        betrayals = {(0, 1): 1.0}  # max severity
        trust = update_trust_single(agents[0], agents[1], agents, betrayal_events=betrayals)
        assert trust < 0.8  # should decrease

    def test_update_all_trust_modifies_agents(self):
        """update_all_trust should modify agent trust scores in place."""
        agents = {
            0: make_agent(0, -0.5, PartyID.LEAN_D),
            1: make_agent(1, -0.3, PartyID.LEAN_D),
            2: make_agent(2, 0.5, PartyID.LEAN_R),
        }
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        update_all_trust(agents, G)

        # Agent 0 should now have trust for agent 1
        assert 1 in agents[0].trust_scores
        # Agent 1 should have trust for both 0 and 2
        assert 0 in agents[1].trust_scores
        assert 2 in agents[1].trust_scores
