"""Ranked Choice Voting (Instant Runoff) tally engine.

Iteratively eliminates the weakest candidate and redistributes their
ballots until one candidate holds a majority of the remaining active votes.
"""

from __future__ import annotations

from collections import Counter


class RCVTally:
    """Ranked Choice Voting / Instant Runoff Voting."""

    def tally(self, ranked_ballots: list[list[str]]) -> dict:
        """Run an RCV election.

        Args:
            ranked_ballots: Each ballot is a list of candidate names ordered
                            by preference (index 0 = first choice).

        Returns:
            dict with keys:
                winner: str - the winning candidate
                rounds: list[dict[str, int]] - vote counts per round
                eliminated: list[str] - candidates in elimination order
        """
        if not ranked_ballots:
            return {"winner": None, "rounds": [], "eliminated": []}

        # Deep-copy ballots so we don't mutate the caller's data
        active_ballots: list[list[str]] = [list(b) for b in ranked_ballots]
        all_candidates = self._all_candidates(active_ballots)

        if not all_candidates:
            return {"winner": None, "rounds": [], "eliminated": []}

        eliminated: list[str] = []
        eliminated_set: set[str] = set()
        rounds: list[dict[str, int]] = []

        # Snapshot of first-round counts (used for tiebreaking elimination)
        first_round_counts: dict[str, int] | None = None

        while True:
            # Count first valid choice on each ballot
            counts = self._count_top_choices(active_ballots, eliminated_set)

            # Ensure every active candidate appears in the round counts
            for c in all_candidates:
                if c not in eliminated_set:
                    counts.setdefault(c, 0)

            rounds.append(dict(counts))

            if first_round_counts is None:
                first_round_counts = dict(counts)

            active_votes = sum(counts.values())
            if active_votes == 0:
                # All ballots exhausted — no winner
                return {"winner": None, "rounds": rounds, "eliminated": eliminated}

            # Check for majority
            max_count = max(counts.values())
            if max_count * 2 > active_votes:
                winner = self._top_candidate(counts)
                return {"winner": winner, "rounds": rounds, "eliminated": eliminated}

            # Only one candidate remaining
            remaining = [c for c in all_candidates if c not in eliminated_set]
            if len(remaining) == 1:
                return {
                    "winner": remaining[0],
                    "rounds": rounds,
                    "eliminated": eliminated,
                }

            # Eliminate the weakest candidate
            loser = self._pick_loser(counts, first_round_counts)
            eliminated.append(loser)
            eliminated_set.add(loser)

            # Strip loser from every ballot (redistributes automatically on
            # next count because we skip eliminated candidates)
            for ballot in active_ballots:
                while ballot and ballot[0] in eliminated_set:
                    ballot.pop(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_candidates(ballots: list[list[str]]) -> set[str]:
        return {c for ballot in ballots for c in ballot}

    @staticmethod
    def _count_top_choices(
        ballots: list[list[str]], eliminated: set[str]
    ) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for ballot in ballots:
            for choice in ballot:
                if choice not in eliminated:
                    counter[choice] += 1
                    break
            # If all choices exhausted, ballot is simply not counted
        return dict(counter)

    @staticmethod
    def _top_candidate(counts: dict[str, int]) -> str:
        max_v = max(counts.values())
        return sorted(c for c, v in counts.items() if v == max_v)[0]

    @staticmethod
    def _pick_loser(
        counts: dict[str, int], first_round_counts: dict[str, int]
    ) -> str:
        """Pick the candidate to eliminate.

        Tie-breaking order:
        1. Fewest votes this round
        2. Fewest first-round votes
        3. Alphabetically first
        """
        min_v = min(counts.values())
        tied = [c for c, v in counts.items() if v == min_v]

        if len(tied) == 1:
            return tied[0]

        # Secondary: fewest first-round votes
        min_first = min(first_round_counts.get(c, 0) for c in tied)
        still_tied = [
            c for c in tied if first_round_counts.get(c, 0) == min_first
        ]

        # Tertiary: alphabetical
        return sorted(still_tied)[0]
