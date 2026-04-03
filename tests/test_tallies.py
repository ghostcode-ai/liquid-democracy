"""Comprehensive tests for all four tally engines.

Covers FPTP, RCV, TRS, and DelegationTally with edge cases.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from tally.fptp import FPTPTally
from tally.rcv import RCVTally
from tally.trs import TRSTally
from tally.delegation_tally import DelegationTallyEngine


# ======================================================================
# FPTP Tests
# ======================================================================

class TestFPTP:
    def setup_method(self):
        self.engine = FPTPTally()

    def test_simple_majority_winner(self):
        votes = {
            "Alice": [1, 2, 3],
            "Bob": [4, 5],
            "Carol": [6],
        }
        result = self.engine.tally(votes)
        assert result["winner"] == "Alice"
        assert result["counts"] == {"Alice": 3, "Bob": 2, "Carol": 1}
        assert result["total_votes"] == 6

    def test_tie_handling_alphabetical(self):
        votes = {
            "Zara": [1, 2],
            "Alice": [3, 4],
            "Mike": [5, 6],
        }
        result = self.engine.tally(votes)
        # Three-way tie at 2 each — alphabetically first wins
        assert result["winner"] == "Alice"
        assert result["total_votes"] == 6

    def test_two_way_tie_alphabetical(self):
        votes = {
            "Bob": [1, 2, 3],
            "Alice": [4, 5, 6],
        }
        result = self.engine.tally(votes)
        assert result["winner"] == "Alice"

    def test_single_candidate(self):
        votes = {"Alice": [1, 2, 3]}
        result = self.engine.tally(votes)
        assert result["winner"] == "Alice"
        assert result["total_votes"] == 3

    def test_no_votes(self):
        result = self.engine.tally({})
        assert result["winner"] is None
        assert result["counts"] == {}
        assert result["total_votes"] == 0

    def test_candidate_with_no_voters(self):
        votes = {"Alice": [1, 2], "Bob": []}
        result = self.engine.tally(votes)
        assert result["winner"] == "Alice"
        assert result["counts"]["Bob"] == 0


# ======================================================================
# RCV Tests
# ======================================================================

class TestRCV:
    def setup_method(self):
        self.engine = RCVTally()

    def test_winner_on_first_round(self):
        ballots = [
            ["Alice", "Bob"],
            ["Alice", "Carol"],
            ["Alice", "Bob"],
            ["Bob", "Carol"],
            ["Carol", "Bob"],
        ]
        result = self.engine.tally(ballots)
        assert result["winner"] == "Alice"
        # Alice has 3/5 = 60% on first round — majority
        assert len(result["rounds"]) == 1
        assert result["eliminated"] == []

    def test_winner_after_elimination(self):
        # Round 1: Alice=2, Bob=2, Carol=1 — no majority
        # Carol eliminated, her ballot goes to Bob
        # Round 2: Alice=2, Bob=3 — Bob wins
        ballots = [
            ["Alice", "Carol"],
            ["Alice", "Bob"],
            ["Bob", "Alice"],
            ["Bob", "Carol"],
            ["Carol", "Bob"],
        ]
        result = self.engine.tally(ballots)
        assert result["winner"] == "Bob"
        assert "Carol" in result["eliminated"]
        assert len(result["rounds"]) == 2

    def test_exhausted_ballots(self):
        """One ballot exhausts after its only candidate is eliminated."""
        ballots = [
            ["Alice", "Bob"],
            ["Alice"],
            ["Bob"],           # exhausts after Bob eliminated
            ["Carol", "Alice"],
            ["Carol", "Bob"],
        ]
        result = self.engine.tally(ballots)
        # Round 1: Alice=2, Bob=1, Carol=2 — Bob eliminated (fewest)
        # After elimination, ballot 2 exhausts (only had Bob).
        # Round 2: Alice=2, Carol=2, active=4, no majority (need >2).
        # Tie-break elimination: first-round Alice=2, Carol=2 still tied,
        # alphabetical loser -> Alice eliminated.
        # Round 3: Carol is the sole remaining candidate -> Carol wins.
        assert result["winner"] == "Carol"
        assert "Bob" in result["eliminated"]
        assert "Alice" in result["eliminated"]
        assert len(result["rounds"]) == 3

    def test_exhausted_ballot_voter_ranked_fewer(self):
        """Voter who only ranked 1 candidate sees ballot exhaust."""
        ballots = [
            ["Alice", "Bob", "Carol"],
            ["Bob"],  # exhausts after Bob eliminated
            ["Carol", "Alice", "Bob"],
            ["Carol", "Bob", "Alice"],
            ["Alice", "Carol", "Bob"],
        ]
        result = self.engine.tally(ballots)
        # Round 1: Alice=2, Bob=1, Carol=2 — Bob eliminated
        # Ballot 1 exhausts. Round 2: Alice=2, Carol=2 (4 active, no majority).
        # After Bob eliminated, ballots redistribute:
        # ballot 0: [Alice, Carol] -> Alice; ballot 1: [] exhausted;
        # ballot 2: [Carol, Alice] -> Carol; ballot 3: [Carol, Alice] -> Carol;
        # ballot 4: [Alice, Carol] -> Alice. Counts: Alice=2, Carol=2.
        # Tiebreak elimination: Alice (alphabetical). Round 3: Carol wins.
        assert result["winner"] == "Carol"
        assert "Bob" in result["eliminated"]
        assert "Alice" in result["eliminated"]

    def test_three_way_race_with_elimination(self):
        ballots = [
            ["Alice", "Bob", "Carol"],
            ["Alice", "Carol", "Bob"],
            ["Bob", "Carol", "Alice"],
            ["Bob", "Carol", "Alice"],
            ["Carol", "Alice", "Bob"],
            ["Carol", "Alice", "Bob"],
            ["Carol", "Bob", "Alice"],
        ]
        result = self.engine.tally(ballots)
        # Round 1: Alice=2, Bob=2, Carol=3 — no majority (need >3.5)
        # Eliminate Alice (tied at 2 with Bob, alphabetically first)
        # Round 2: Bob=2, Carol=3+2=5? No — redistribute Alice's ballots
        # Ballot 0: Alice eliminated -> Bob; Ballot 1: Alice eliminated -> Carol
        # Round 2: Bob=2+1=3, Carol=3+1=4 — Carol has 4/7>50%? 4/7=57% yes
        assert result["winner"] == "Carol"
        assert "Alice" in result["eliminated"]

    def test_five_candidates_multiple_rounds(self):
        ballots = [
            ["A", "B", "C", "D", "E"],
            ["A", "C", "B", "E", "D"],
            ["B", "A", "C", "D", "E"],
            ["B", "C", "A", "E", "D"],
            ["C", "D", "E", "A", "B"],
            ["C", "E", "D", "B", "A"],
            ["D", "C", "B", "A", "E"],
            ["E", "D", "C", "B", "A"],
            ["E", "D", "A", "B", "C"],
            ["A", "B", "D", "C", "E"],
        ]
        result = self.engine.tally(ballots)
        # 10 ballots: A=3, B=2, C=2, D=1, E=2
        # Round 1: eliminate D (1 vote, fewest)
        # Multiple rounds should follow
        assert result["winner"] is not None
        assert len(result["rounds"]) >= 2
        assert len(result["eliminated"]) >= 1

    def test_empty_ballots(self):
        result = self.engine.tally([])
        assert result["winner"] is None
        assert result["rounds"] == []

    def test_single_candidate_all_first(self):
        ballots = [["Alice"], ["Alice"], ["Alice"]]
        result = self.engine.tally(ballots)
        assert result["winner"] == "Alice"
        assert len(result["rounds"]) == 1


# ======================================================================
# TRS Tests
# ======================================================================

class TestTRS:
    def setup_method(self):
        self.engine = TRSTally()

    def test_outright_round1_winner(self):
        """Candidate with >50% of cast and >=25% of registered wins R1."""
        votes = {
            "Alice": list(range(60)),
            "Bob": list(range(60, 90)),
            "Carol": list(range(90, 100)),
        }
        result = self.engine.round1(votes, registered_voters=200)
        assert result["winner"] == "Alice"
        assert result["round"] == 1
        assert result["counts"]["Alice"] == 60

    def test_no_round1_winner_insufficient_majority(self):
        """No one gets >50% of cast votes."""
        votes = {
            "Alice": list(range(40)),
            "Bob": list(range(40, 75)),
            "Carol": list(range(75, 100)),
        }
        result = self.engine.round1(votes, registered_voters=100)
        assert "winner" not in result
        assert "round2_candidates" in result

    def test_no_round1_winner_insufficient_registered_share(self):
        """Candidate has >50% of cast but <25% of registered."""
        # 10 votes cast out of 100 registered; Alice gets 6/10 = 60% of cast
        # but 6/100 = 6% of registered (needs 25%)
        votes = {
            "Alice": list(range(6)),
            "Bob": list(range(6, 10)),
        }
        result = self.engine.round1(votes, registered_voters=100)
        assert "winner" not in result
        assert "round2_candidates" in result

    def test_round2_plurality(self):
        votes = {
            "Alice": [1, 2, 3, 4, 5],
            "Bob": [6, 7, 8],
        }
        result = self.engine.round2(votes)
        assert result["winner"] == "Alice"
        assert result["round2_counts"]["Alice"] == 5

    def test_withdrawal_simulation(self):
        """Third-place+ candidates withdraw with probability withdrawal_prob."""
        random.seed(42)
        # 3 candidates qualify; top 2 always advance, 3rd may withdraw
        votes = {
            "Alice": list(range(40)),
            "Bob": list(range(40, 70)),
            "Carol": list(range(70, 90)),
        }
        # All 3 clear 12.5% of 100 registered
        result = self.engine.round1(
            votes, registered_voters=100, withdrawal_prob=0.6
        )
        assert "round2_candidates" in result
        # Top 2 must always be present
        assert "Bob" in result["round2_candidates"]
        assert "Alice" in result["round2_candidates"]

    def test_withdrawal_deterministic_with_seed(self):
        """Same seed produces same withdrawal outcome."""
        votes = {
            "Alice": list(range(30)),
            "Bob": list(range(30, 55)),
            "Carol": list(range(55, 80)),
            "Dave": list(range(80, 100)),
        }
        results = []
        for _ in range(2):
            random.seed(99)
            r = self.engine.round1(votes, registered_voters=100)
            results.append(r)
        assert results[0]["round2_candidates"] == results[1]["round2_candidates"]

    def test_threshold_qualification(self):
        """Only candidates with >=12.5% of registered advance."""
        # registered=100, threshold=12.5
        votes = {
            "Alice": list(range(40)),
            "Bob": list(range(40, 70)),
            "Carol": list(range(70, 80)),  # 10 votes = 10% < 12.5%
        }
        random.seed(0)
        result = self.engine.round1(votes, registered_voters=100)
        assert "round2_candidates" in result
        # Carol below threshold — only Alice and Bob qualify
        assert "Carol" not in result["round2_candidates"]

    def test_fewer_than_2_qualify_takes_top_2(self):
        """If <2 candidates clear threshold, top 2 by votes advance."""
        # registered=1000, threshold=125 votes
        votes = {
            "Alice": list(range(80)),   # 80 < 125
            "Bob": list(range(80, 130)),  # 50 < 125
            "Carol": list(range(130, 150)),  # 20 < 125
        }
        result = self.engine.round1(votes, registered_voters=1000)
        assert "round2_candidates" in result
        # None qualify, so top 2 advance: Alice (80) and Bob (50)
        assert len(result["round2_candidates"]) == 2
        assert "Alice" in result["round2_candidates"]
        assert "Bob" in result["round2_candidates"]

    def test_full_election_round1_win(self):
        votes = {
            "Alice": list(range(60)),
            "Bob": list(range(60, 100)),
        }
        mock_r2_fn = MagicMock()
        result = self.engine.full_election(votes, 200, mock_r2_fn)
        assert result["winner"] == "Alice"
        assert result["decided_in"] == 1
        mock_r2_fn.assert_not_called()

    def test_full_election_goes_to_round2(self):
        r1_votes = {
            "Alice": list(range(40)),
            "Bob": list(range(40, 75)),
            "Carol": list(range(75, 100)),
        }

        def r2_fn(candidates):
            return {c: list(range(50)) if c == "Alice" else list(range(50, 80))
                    for c in candidates}

        random.seed(0)
        result = self.engine.full_election(r1_votes, 100, r2_fn)
        assert result["decided_in"] == 2
        assert result["winner"] is not None
        assert result["round1_counts"] is not None
        assert result["round2_counts"] is not None


# ======================================================================
# Delegation Tally Tests
# ======================================================================

def _make_delegation_graph(delegations: dict[int, int]) -> MagicMock:
    """Create a mock DelegationGraph.

    Args:
        delegations: {delegator_id: delegate_id} — a simple one-level map.
                     Chains are resolved into weights on the direct voters.
    """
    graph = MagicMock()

    # Compute weights: each direct voter gets weight = 1 + number of people
    # who (transitively) delegated to them.
    def _resolve_weights(topic: str) -> dict[int, float]:
        # Build reverse: who delegates to whom
        incoming: dict[int, list[int]] = {}
        for src, dst in delegations.items():
            incoming.setdefault(dst, []).append(src)

        # All nodes that are endpoints (not delegating further, or targets)
        all_nodes = set(delegations.keys()) | set(delegations.values())
        direct_voters = all_nodes - set(delegations.keys())

        weights: dict[int, float] = {}
        for voter in direct_voters:
            # Count chain length: how many delegate into this voter
            w = 1.0
            stack = list(incoming.get(voter, []))
            while stack:
                node = stack.pop()
                w += 1.0
                stack.extend(incoming.get(node, []))
            weights[voter] = w
        return weights

    graph.resolve_weights = _resolve_weights
    graph.count_delegations = lambda topic: len(delegations)
    return graph


class TestDelegationTally:
    def setup_method(self):
        self.engine = DelegationTallyEngine()

    def test_all_direct_votes_no_delegation(self):
        """No delegations — should behave like FPTP."""
        graph = _make_delegation_graph({})
        direct_votes = {1: "Alice", 2: "Alice", 3: "Bob"}
        result = self.engine.tally(direct_votes, graph, "topic1")
        assert result["winner"] == "Alice"
        assert result["weighted_counts"]["Alice"] == pytest.approx(2.0)
        assert result["weighted_counts"]["Bob"] == pytest.approx(1.0)
        assert result["delegation_chains"] == 0

    def test_simple_delegation_changes_outcome(self):
        """Delegation flips the winner from Alice to Bob."""
        # Voters 1,2 vote Alice; voter 3 votes Bob.
        # But voters 4,5 delegate to voter 3 (Bob voter).
        # Voter 3 weight = 1 + 2 = 3
        # Alice: 1+1=2, Bob: 3 → Bob wins
        graph = _make_delegation_graph({4: 3, 5: 3})
        direct_votes = {1: "Alice", 2: "Alice", 3: "Bob"}
        result = self.engine.tally(direct_votes, graph, "topic1")
        assert result["winner"] == "Bob"
        assert result["weighted_counts"]["Bob"] == pytest.approx(3.0)
        assert result["weighted_counts"]["Alice"] == pytest.approx(2.0)
        assert result["max_weight"] == pytest.approx(3.0)

    def test_weight_cap(self):
        """Weight cap limits the influence of heavily-delegated voters."""
        # Voter 3 would have weight 3 (from 4,5 delegating), but cap=2.0
        graph = _make_delegation_graph({4: 3, 5: 3})
        direct_votes = {1: "Alice", 2: "Alice", 3: "Bob"}
        result = self.engine.tally(
            direct_votes, graph, "topic1", weight_cap=2.0
        )
        # Bob: min(3, 2) = 2, Alice: 1+1=2 — tie, Alice wins alphabetically
        assert result["winner"] == "Alice"
        assert result["weighted_counts"]["Bob"] == pytest.approx(2.0)
        assert result["weighted_counts"]["Alice"] == pytest.approx(2.0)
        assert result["max_weight"] == pytest.approx(2.0)

    def test_gini_coefficient_uniform(self):
        """All equal weights should produce Gini ~= 0."""
        graph = _make_delegation_graph({})
        direct_votes = {1: "A", 2: "B", 3: "C", 4: "A"}
        result = self.engine.tally(direct_votes, graph, "topic1")
        assert result["gini"] == pytest.approx(0.0, abs=0.01)

    def test_gini_coefficient_unequal(self):
        """Unequal weights should produce a positive Gini."""
        # Voter 1 has weight 1, voter 2 has weight 4 (3 delegators)
        graph = _make_delegation_graph({3: 2, 4: 2, 5: 2})
        direct_votes = {1: "Alice", 2: "Bob"}
        result = self.engine.tally(direct_votes, graph, "topic1")
        assert result["gini"] > 0.0
        # Weights are [1, 4]. Gini for [1,4]: |1-4|/(2*5) = 3/10 = 0.3
        assert result["gini"] == pytest.approx(0.3, abs=0.05)

    def test_chain_delegation(self):
        """Multi-hop delegation: 4->3->2, voter 2 is direct."""
        graph = _make_delegation_graph({4: 3, 3: 2})
        direct_votes = {1: "Alice", 2: "Bob"}
        result = self.engine.tally(direct_votes, graph, "topic1")
        # Voter 2 has weight 3 (self + 3 + 4)
        assert result["weighted_counts"]["Bob"] == pytest.approx(3.0)
        assert result["total_effective_votes"] == pytest.approx(4.0)

    def test_empty_election(self):
        """No direct votes at all."""
        graph = _make_delegation_graph({})
        result = self.engine.tally({}, graph, "topic1")
        assert result["winner"] is None
        assert result["total_effective_votes"] == pytest.approx(0.0)
        assert result["gini"] == pytest.approx(0.0)
