"""Comprehensive tests for DelegationGraph."""

import math

from engine.delegation_graph import DelegationGraph


# ------------------------------------------------------------------
# 1. Simple delegation: A delegates to B, B votes directly
# ------------------------------------------------------------------


def test_simple_delegation():
    g = DelegationGraph()
    g.add_delegation("A", "B", "budget")

    weights = g.resolve_all("budget")

    # B votes directly (weight 1.0 for self) plus A's decayed vote (0.85^1).
    assert math.isclose(weights["B"], 1.0 + 0.85, rel_tol=1e-9)
    assert weights["A"] == 0.0


# ------------------------------------------------------------------
# 2. Chain delegation: A -> B -> C, C votes directly
# ------------------------------------------------------------------


def test_chain_delegation():
    g = DelegationGraph()
    g.add_delegation("A", "B", "budget")
    g.add_delegation("B", "C", "budget")

    weights = g.resolve_all("budget")

    # C: own vote (1.0) + B's delegation (0.85^1) + A's delegation (0.85^2)
    expected_c = 1.0 + 0.85 + 0.85**2
    assert math.isclose(weights["C"], expected_c, rel_tol=1e-9)
    assert weights["A"] == 0.0
    assert weights["B"] == 0.0


# ------------------------------------------------------------------
# 3. Cycle detection: A -> B -> C -> A, all lose votes
# ------------------------------------------------------------------


def test_cycle_all_votes_lost():
    g = DelegationGraph()
    g.add_delegation("A", "B", "tax")
    g.add_delegation("B", "C", "tax")
    g.add_delegation("C", "A", "tax")

    weights = g.resolve_all("tax")

    assert weights["A"] == 0.0
    assert weights["B"] == 0.0
    assert weights["C"] == 0.0


def test_detect_all_cycles():
    g = DelegationGraph()
    g.add_delegation("A", "B", "tax")
    g.add_delegation("B", "C", "tax")
    g.add_delegation("C", "A", "tax")

    cycles = g.detect_all_cycles("tax")
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C"}


# ------------------------------------------------------------------
# 4. Revocation: delegate then revoke
# ------------------------------------------------------------------


def test_revoke_returns_weight():
    g = DelegationGraph()
    g.add_delegation("A", "B", "budget")

    # Before revocation B has accumulated weight.
    weights_before = g.resolve_all("budget")
    assert math.isclose(weights_before["B"], 1.85, rel_tol=1e-9)

    g.revoke("A", "budget")
    weights_after = g.resolve_all("budget")

    # After revocation A is no longer in any edge, so resolve_all
    # returns an empty dict (no edges for this topic).
    assert weights_after.get("A", 0.0) == 0.0
    # B is also absent since there are no edges.
    assert weights_after.get("B", 0.0) == 0.0


def test_revoke_only_affects_topic():
    g = DelegationGraph()
    g.add_delegation("A", "B", "budget")
    g.add_delegation("A", "C", "defense")

    g.revoke("A", "budget")

    budget_weights = g.resolve_all("budget")
    defense_weights = g.resolve_all("defense")

    # Budget delegation gone.
    assert budget_weights.get("B", 0.0) == 0.0
    # Defense delegation intact.
    assert math.isclose(defense_weights["C"], 1.0 + 0.85, rel_tol=1e-9)


# ------------------------------------------------------------------
# 5. Weight cap
# ------------------------------------------------------------------


def test_weight_cap():
    g = DelegationGraph()
    # Four people delegate to E.
    for voter in ["A", "B", "C", "D"]:
        g.add_delegation(voter, "E", "budget")

    uncapped = g.resolve_all("budget")
    # E: 1.0 + 4 * 0.85 = 4.4
    assert math.isclose(uncapped["E"], 1.0 + 4 * 0.85, rel_tol=1e-9)

    capped = g.resolve_all("budget", weight_cap=2.0)
    assert capped["E"] == 2.0


# ------------------------------------------------------------------
# 6. Gini coefficient
# ------------------------------------------------------------------


def test_gini_equal_weights():
    """When everyone votes directly (no delegations), Gini should be 0."""
    g = DelegationGraph()
    # Create edges so nodes appear, but each pair is symmetric -- actually,
    # with no delegations there are no edges at all so resolve_all returns
    # nothing.  Instead, create a simple 1-hop chain to get two nodes with
    # known weights and compute manually.

    # For a perfectly equal distribution we need direct voters only.
    # Since resolve_all only sees nodes that participate in edges, we
    # fabricate a scenario: A->B and C->D (two independent pairs).
    g.add_delegation("A", "B", "t")
    g.add_delegation("C", "D", "t")

    # B gets 1.85, D gets 1.85, A=0, C=0.  Not perfectly equal but
    # weights are symmetric between the two delegates.
    gini = g.get_gini("t")
    # Two 0s and two 1.85s: Gini = 0.5
    assert math.isclose(gini, 0.5, abs_tol=0.01)


def test_gini_maximum_inequality():
    """One person holds almost all the weight -> Gini near 1."""
    g = DelegationGraph()
    for i in range(20):
        g.add_delegation(f"V{i}", "boss", "power")

    gini = g.get_gini("power")
    # boss has ~1 + 20*0.85 = 18.0, everyone else has 0.
    assert gini > 0.9


def test_gini_empty_graph():
    g = DelegationGraph()
    assert g.get_gini("nonexistent") == 0.0


# ------------------------------------------------------------------
# 7. Multiple topics: isolation
# ------------------------------------------------------------------


def test_multiple_topics_isolation():
    g = DelegationGraph()
    g.add_delegation("A", "B", "senate")
    g.add_delegation("C", "D", "house")

    senate_weights = g.resolve_all("senate")
    house_weights = g.resolve_all("house")

    # Senate should only involve A and B.
    assert "C" not in senate_weights
    assert "D" not in senate_weights
    assert math.isclose(senate_weights["B"], 1.85, rel_tol=1e-9)

    # House should only involve C and D.
    assert "A" not in house_weights
    assert "B" not in house_weights
    assert math.isclose(house_weights["D"], 1.85, rel_tol=1e-9)


# ------------------------------------------------------------------
# 8. Long chain (10 deep): viscous decay
# ------------------------------------------------------------------


def test_long_chain_viscous_decay():
    g = DelegationGraph()
    # Build chain: V0 -> V1 -> V2 -> ... -> V10 (V10 votes directly)
    for i in range(10):
        g.add_delegation(f"V{i}", f"V{i+1}", "deep")

    weights = g.resolve_all("deep")

    # V10's weight = sum of 0.85^k for k in 0..10
    expected = sum(0.85**k for k in range(11))
    assert math.isclose(weights["V10"], expected, rel_tol=1e-9)

    # Verify the chain_length helper.
    assert g.get_chain_length("V0", "deep") == 10
    assert g.get_chain_length("V10", "deep") == 0

    # Weight at depth 10 is quite small.
    assert DelegationGraph.viscous_decay(10) < 0.20


# ------------------------------------------------------------------
# 9. Mixed: direct voters, delegators, and cycles
# ------------------------------------------------------------------


def test_mixed_scenario():
    g = DelegationGraph()

    # Direct voter (no delegation): not in any edge, so won't appear in
    # resolve_all.  Instead, D1 delegates to DIRECT who votes directly.
    g.add_delegation("D1", "DIRECT", "mixed")

    # Chain: C1 -> C2 -> DIRECT
    g.add_delegation("C1", "C2", "mixed")
    g.add_delegation("C2", "DIRECT", "mixed")

    # Cycle: X -> Y -> Z -> X
    g.add_delegation("X", "Y", "mixed")
    g.add_delegation("Y", "Z", "mixed")
    g.add_delegation("Z", "X", "mixed")

    weights = g.resolve_all("mixed")

    # Cycle members lose their vote.
    assert weights["X"] == 0.0
    assert weights["Y"] == 0.0
    assert weights["Z"] == 0.0

    # DIRECT accumulates: own (1.0) + D1 (0.85) + C2 (0.85) + C1 (0.85^2)
    expected_direct = 1.0 + 0.85 + 0.85 + 0.85**2
    assert math.isclose(weights["DIRECT"], expected_direct, rel_tol=1e-9)

    # Delegators have 0 weight.
    assert weights["D1"] == 0.0
    assert weights["C1"] == 0.0
    assert weights["C2"] == 0.0


# ------------------------------------------------------------------
# 10. Empty graph edge cases
# ------------------------------------------------------------------


def test_empty_graph_resolve_all():
    g = DelegationGraph()
    assert g.resolve_all("anything") == {}


def test_empty_graph_detect_cycles():
    g = DelegationGraph()
    assert g.detect_all_cycles("anything") == []


def test_empty_graph_get_max_weight():
    g = DelegationGraph()
    assert g.get_max_weight("anything") == 0.0


def test_empty_graph_get_delegators():
    g = DelegationGraph()
    assert g.get_delegators("nobody", "anything") == []


def test_empty_graph_get_chain_length():
    g = DelegationGraph()
    # No edges, voter votes directly -> chain length 0.
    assert g.get_chain_length("ghost", "void") == 0


# ------------------------------------------------------------------
# Additional edge-case tests
# ------------------------------------------------------------------


def test_resolve_chain_direct_voter():
    """A voter with no delegation resolves to themselves with weight 1.0."""
    g = DelegationGraph()
    # Add an edge so the voter exists in the graph but has no delegation
    # on the queried topic.
    g.add_delegation("A", "B", "other_topic")
    final, depth, weight = g.resolve_chain("A", "budget")
    assert final == "A"
    assert depth == 0
    assert weight == 1.0


def test_get_delegators():
    g = DelegationGraph()
    g.add_delegation("A", "B", "t")
    g.add_delegation("C", "B", "t")
    g.add_delegation("D", "B", "other")

    delegators = g.get_delegators("B", "t")
    assert set(delegators) == {"A", "C"}

    # D delegated on a different topic.
    assert "D" not in delegators


def test_get_max_weight():
    g = DelegationGraph()
    g.add_delegation("A", "B", "t")
    g.add_delegation("C", "D", "t")

    max_w = g.get_max_weight("t")
    # Both B and D should have 1.85, max is 1.85.
    assert math.isclose(max_w, 1.85, rel_tol=1e-9)


def test_replace_delegation():
    """Adding a new delegation on the same topic replaces the old one."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t")
    g.add_delegation("A", "C", "t")

    weights = g.resolve_all("t")
    # A now delegates to C, not B.
    assert math.isclose(weights["C"], 1.0 + 0.85, rel_tol=1e-9)
    # B is no longer a target -- not in any edge for this topic.
    assert weights.get("B", 0.0) == 0.0


def test_chain_into_cycle_loses_vote():
    """A chain that terminates inside a cycle also loses its vote."""
    g = DelegationGraph()
    # Cycle: B -> C -> B
    g.add_delegation("B", "C", "t")
    g.add_delegation("C", "B", "t")
    # A delegates into the cycle.
    g.add_delegation("A", "B", "t")

    weights = g.resolve_all("t")

    assert weights["A"] == 0.0
    assert weights["B"] == 0.0
    assert weights["C"] == 0.0


def test_viscous_decay_values():
    assert DelegationGraph.viscous_decay(0) == 1.0
    assert math.isclose(DelegationGraph.viscous_decay(1), 0.85)
    assert math.isclose(DelegationGraph.viscous_decay(2), 0.7225)
    assert math.isclose(DelegationGraph.viscous_decay(5), 0.85**5)
    # Custom alpha
    assert DelegationGraph.viscous_decay(3, alpha=0.5) == 0.125


# ------------------------------------------------------------------
# K-way delegation splitting (k=2)
# ------------------------------------------------------------------

def test_k2_splits_weight_evenly():
    """A delegates 0.5 to B and 0.5 to C. Each gets A's half + own vote."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t", fraction=0.5)
    g.add_delegation("A", "C", "t", fraction=0.5)

    weights = g.resolve_all("t")
    # B gets own vote (1.0) + A's 0.5 * decay(1) = 1.0 + 0.425 = 1.425
    assert math.isclose(weights["B"], 1.0 + 0.5 * 0.85)
    assert math.isclose(weights["C"], 1.0 + 0.5 * 0.85)
    assert weights["A"] == 0.0


def test_k2_with_chain():
    """A→(B,C) with fraction 0.5 each, B→D."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t", fraction=0.5)
    g.add_delegation("A", "C", "t", fraction=0.5)
    g.add_delegation("B", "D", "t")

    weights = g.resolve_all("t")
    # C: own vote (decay 0) + A's 0.5 * decay(1) = 1.0 + 0.425
    assert math.isclose(weights["C"], 1.0 + 0.5 * 0.85)
    # D: own vote (decay 0) + B's full vote at decay(1) + A's 0.5 at decay(2)
    assert math.isclose(weights["D"], 1.0 + 0.85 + 0.5 * 0.85**2)
    assert weights["A"] == 0.0
    assert weights["B"] == 0.0


def test_k2_full_fraction_replaces():
    """fraction=1.0 (default) replaces existing delegation."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t")
    g.add_delegation("A", "C", "t")  # fraction=1.0 → replaces A→B

    weights = g.resolve_all("t")
    # C: own vote (1.0) + A's full delegation at decay(1) = 1.85
    assert math.isclose(weights["C"], 1.0 + 0.85)
    # B should not receive A's delegation (was replaced)
    assert weights.get("B", 0.0) == 0.0


def test_k3_splits_weight_three_ways():
    """A delegates 1/3 each to B, C, D."""
    g = DelegationGraph()
    g.add_delegation("A", "B", "t", fraction=1/3)
    g.add_delegation("A", "C", "t", fraction=1/3)
    g.add_delegation("A", "D", "t", fraction=1/3)

    weights = g.resolve_all("t")
    # Each delegate: own vote (1.0) + A's 1/3 * decay(1)
    for delegate in ["B", "C", "D"]:
        assert math.isclose(weights[delegate], 1.0 + (1/3) * 0.85, rel_tol=1e-6)
    assert weights["A"] == 0.0


def test_k2_multiple_delegators():
    """Two voters each split to the same delegate pair."""
    g = DelegationGraph()
    g.add_delegation("V1", "B", "t", fraction=0.5)
    g.add_delegation("V1", "C", "t", fraction=0.5)
    g.add_delegation("V2", "B", "t", fraction=0.5)
    g.add_delegation("V2", "C", "t", fraction=0.5)

    weights = g.resolve_all("t")
    # B: own (1.0) + V1's 0.5*0.85 + V2's 0.5*0.85 = 1.0 + 0.85 = 1.85
    assert math.isclose(weights["B"], 1.0 + 0.85)
    assert math.isclose(weights["C"], 1.0 + 0.85)
