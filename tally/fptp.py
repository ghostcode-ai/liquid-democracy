"""First Past The Post tally engine.

Counts raw votes per candidate. The candidate with the most votes wins.
Ties broken alphabetically for determinism.
"""

from __future__ import annotations


class FPTPTally:
    """Simple plurality voting: highest vote count wins."""

    def tally(self, votes: dict[str, list[int]]) -> dict:
        """Tally FPTP votes.

        Args:
            votes: Mapping of candidate_name -> list of voter IDs who voted
                   for that candidate.

        Returns:
            dict with keys:
                winner: str - winning candidate (alphabetically first on tie)
                counts: dict[str, int] - vote count per candidate
                total_votes: int - sum of all votes cast
        """
        counts = {candidate: len(voters) for candidate, voters in votes.items()}
        total_votes = sum(counts.values())

        if not counts:
            return {"winner": None, "counts": {}, "total_votes": 0}

        max_count = max(counts.values())
        # All candidates tied at the max — pick alphabetically first
        tied = sorted(c for c, v in counts.items() if v == max_count)
        winner = tied[0]

        return {"winner": winner, "counts": counts, "total_votes": total_votes}
