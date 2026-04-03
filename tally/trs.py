"""Two-Round System (French-style) tally engine.

Round 1: outright majority (>50% of cast AND >25% of registered) wins.
Otherwise, candidates clearing 12.5% of registered voters advance to
round 2, where simple plurality decides.
"""

from __future__ import annotations

import random


class TRSTally:
    """Two-Round System election engine."""

    def round1(
        self,
        votes: dict[str, list[int]],
        registered_voters: int,
        withdrawal_prob: float = 0.6,
    ) -> dict:
        """Execute round 1.

        Args:
            votes: {candidate_name: [voter_ids]}
            registered_voters: Total registered electorate size.
            withdrawal_prob: Probability that a qualifying 3rd+ place
                             candidate withdraws before round 2.

        Returns:
            If outright winner:
                {"winner": str, "round": 1, "counts": dict}
            Otherwise:
                {"round2_candidates": list[str], "round1_counts": dict}
        """
        counts = {c: len(v) for c, v in votes.items()}
        total_cast = sum(counts.values())

        if not counts:
            return {"winner": None, "round": 1, "counts": {}}

        # Check for outright majority
        for candidate, n in counts.items():
            if (
                total_cast > 0
                and n > total_cast / 2
                and n >= registered_voters * 0.25
            ):
                return {"winner": candidate, "round": 1, "counts": counts}

        # Determine who qualifies for round 2
        threshold = registered_voters * 0.125
        qualifiers = sorted(
            [c for c, n in counts.items() if n >= threshold],
            key=lambda c: (-counts[c], c),
        )

        # If fewer than 2 qualify, take top 2 by vote count
        if len(qualifiers) < 2:
            qualifiers = sorted(counts, key=lambda c: (-counts[c], c))[:2]

        # Top 2 always advance; 3rd+ may withdraw
        top2 = qualifiers[:2]
        rest = qualifiers[2:]
        advancing = list(top2)
        for candidate in rest:
            if random.random() >= withdrawal_prob:
                advancing.append(candidate)

        # Sort for determinism
        advancing.sort(key=lambda c: (-counts[c], c))

        return {"round2_candidates": advancing, "round1_counts": counts}

    def round2(self, votes: dict[str, list[int]]) -> dict:
        """Execute round 2 (simple plurality).

        Args:
            votes: {candidate_name: [voter_ids]}

        Returns:
            {"winner": str, "round2_counts": dict}
        """
        counts = {c: len(v) for c, v in votes.items()}
        if not counts:
            return {"winner": None, "round2_counts": {}}

        max_count = max(counts.values())
        tied = sorted(c for c, v in counts.items() if v == max_count)
        return {"winner": tied[0], "round2_counts": counts}

    def full_election(
        self,
        round1_votes: dict[str, list[int]],
        registered_voters: int,
        round2_vote_fn,
        withdrawal_prob: float = 0.6,
    ) -> dict:
        """Run a complete two-round election.

        Args:
            round1_votes: {candidate_name: [voter_ids]} for round 1.
            registered_voters: Total registered electorate size.
            round2_vote_fn: Callable(round2_candidates) -> dict of
                            {candidate: [voter_ids]} for round 2.
            withdrawal_prob: Withdrawal probability for 3rd+ qualifiers.

        Returns:
            dict with full election results including both rounds.
        """
        r1 = self.round1(round1_votes, registered_voters, withdrawal_prob)

        if "winner" in r1:
            return {
                "winner": r1["winner"],
                "decided_in": 1,
                "round1_counts": r1["counts"],
                "round2_counts": None,
            }

        round2_votes = round2_vote_fn(r1["round2_candidates"])
        r2 = self.round2(round2_votes)

        return {
            "winner": r2["winner"],
            "decided_in": 2,
            "round1_counts": r1["round1_counts"],
            "round2_counts": r2["round2_counts"],
        }
