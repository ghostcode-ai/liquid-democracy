"""Delegation tally engine.

Resolves delegation chains via a DelegationGraph, applies accumulated
weights to direct voters, and sums weighted votes per candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.delegation_graph import DelegationGraph


class DelegationTallyEngine:
    """Tally that resolves liquid-democracy delegations into weighted votes."""

    def tally(
        self,
        direct_votes: dict[int, str],
        delegation_graph: DelegationGraph,
        topic: str,
        weight_cap: float | None = None,
    ) -> dict:
        """Run a delegation-weighted tally.

        Args:
            direct_votes: {voter_id: candidate_name} for voters who cast
                          a ballot directly.
            delegation_graph: The delegation graph to resolve chains from.
            topic: Topic/race identifier used to look up delegations.
            weight_cap: Optional maximum weight any single voter can carry.

        Returns:
            dict with keys:
                winner: str
                weighted_counts: dict[str, float]
                gini: float - Gini coefficient of voter weights
                max_weight: float - largest single-voter weight
                delegation_chains: int - number of delegations resolved
                total_effective_votes: float - sum of all weighted votes
        """
        # Resolve weights: each direct voter accumulates weight from their
        # delegation chain (themselves + everyone who delegated to them,
        # transitively).
        weights = delegation_graph.resolve_weights(topic)
        delegation_chains = delegation_graph.count_delegations(topic)

        # Build weighted counts
        weighted_counts: dict[str, float] = {}
        voter_weights: list[float] = []

        for voter_id, candidate in direct_votes.items():
            w = weights.get(voter_id, 1.0)
            if weight_cap is not None:
                w = min(w, weight_cap)
            voter_weights.append(w)
            weighted_counts[candidate] = weighted_counts.get(candidate, 0.0) + w

        total_effective = sum(weighted_counts.values())
        max_weight = max(voter_weights) if voter_weights else 0.0
        gini = self._gini(voter_weights)

        # Determine winner (alphabetical tiebreak)
        if not weighted_counts:
            winner = None
        else:
            max_val = max(weighted_counts.values())
            tied = sorted(c for c, v in weighted_counts.items() if v == max_val)
            winner = tied[0]

        return {
            "winner": winner,
            "weighted_counts": weighted_counts,
            "gini": gini,
            "max_weight": max_weight,
            "delegation_chains": delegation_chains,
            "total_effective_votes": total_effective,
        }

    @staticmethod
    def _gini(values: list[float]) -> float:
        """Compute the Gini coefficient for a list of values.

        Returns 0.0 for empty or uniform distributions.
        """
        if not values or len(values) == 1:
            return 0.0
        n = len(values)
        sorted_vals = sorted(values)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumulative += v
            weighted_sum += (2 * (i + 1) - n - 1) * v
        return weighted_sum / (n * total)
