"""Unified election runner: runs all 4 voting systems on the same electorate.

Compares FPTP, RCV, TRS, and liquid-democracy delegation tallies side by side,
producing a divergence analysis that highlights where systems agree or disagree.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from engine.delegation_graph import DelegationGraph
from tally.fptp import FPTPTally
from tally.rcv import RCVTally
from tally.trs import TRSTally
from tally.delegation_tally import DelegationTallyEngine


class ElectionRunner:
    """Run all 4 voting systems on the same electorate."""

    def __init__(self, delegation_graph: DelegationGraph | None = None):
        self.fptp = FPTPTally()
        self.rcv = RCVTally()
        self.trs = TRSTally()
        self.delegation = DelegationTallyEngine()
        self.delegation_graph = delegation_graph or DelegationGraph()

    def run_all(
        self,
        fptp_votes: dict[str, list[int]],
        ranked_ballots: list[list[str]],
        trs_round1_votes: dict[str, list[int]],
        registered_voters: int,
        direct_votes: dict[int, str],
        topic: str,
        trs_round2_vote_fn: Callable[[list[str]], dict[str, list[int]]] | None = None,
        weight_cap: float | None = None,
        trs_withdrawal_prob: float = 0.6,
        seed: int | None = None,
    ) -> dict:
        """Run all 4 systems and return combined results.

        Args:
            fptp_votes: {candidate: [voter_ids]} for plurality voting.
            ranked_ballots: List of ranked ballots (each is [1st, 2nd, ...])
                for RCV.
            trs_round1_votes: {candidate: [voter_ids]} for TRS round 1.
            registered_voters: Total registered electorate (needed by TRS).
            direct_votes: {voter_id: candidate} for delegation tally.
            topic: Topic string used by the delegation graph.
            trs_round2_vote_fn: Optional callable(round2_candidates) ->
                {candidate: [voter_ids]}.  If None, a simple redistribution
                of round-1 votes among qualifying candidates is used.
            weight_cap: Optional max weight for delegation tally.
            trs_withdrawal_prob: Probability 3rd+ qualifier withdraws in TRS.
            seed: Random seed for reproducibility.

        Returns:
            dict with keys: fptp, rcv, trs, delegation, comparison.
            ``comparison`` shows which systems agree/disagree on the winner.
        """
        if seed is not None:
            random.seed(seed)

        # --- FPTP ---
        fptp_result = self.fptp.tally(fptp_votes)

        # --- RCV ---
        rcv_result = self.rcv.tally(ranked_ballots)

        # --- TRS ---
        round2_fn = trs_round2_vote_fn or self._default_round2_fn(trs_round1_votes)
        trs_result = self.trs.full_election(
            trs_round1_votes,
            registered_voters,
            round2_fn,
            trs_withdrawal_prob,
        )

        # --- Delegation ---
        delegation_result = self.delegation.tally(
            direct_votes,
            self.delegation_graph,
            topic,
            weight_cap=weight_cap,
        )

        results = {
            "fptp": fptp_result,
            "rcv": rcv_result,
            "trs": trs_result,
            "delegation": delegation_result,
        }

        results["comparison"] = self.compare_outcomes(results)
        return results

    def compare_outcomes(self, results: dict) -> dict:
        """Analyze outcome differences across systems.

        Returns a dict with:
            winner_map: {system_name: winner_name}
            agreement_matrix: 4x4 bool matrix (system_i agrees with system_j)
            unique_winners: set of distinct winners
            divergence_score: float in [0, 1]
                0.0 = unanimous (all 4 pick the same winner)
                1.0 = maximum divergence (every system picks a different winner)
            divergence_reasons: list of human-readable notes explaining splits
            consensus_winner: the most-agreed-upon winner, or None on a 4-way tie
        """
        system_names = ["fptp", "rcv", "trs", "delegation"]

        winner_map: dict[str, str | None] = {}
        for name in system_names:
            winner_map[name] = results.get(name, {}).get("winner")

        # --- Agreement matrix ---
        agreement_matrix: dict[str, dict[str, bool]] = {}
        for a in system_names:
            agreement_matrix[a] = {}
            for b in system_names:
                agreement_matrix[a][b] = (
                    winner_map[a] is not None
                    and winner_map[a] == winner_map[b]
                )

        # --- Unique winners (excluding None) ---
        non_none_winners = [w for w in winner_map.values() if w is not None]
        unique_winners = set(non_none_winners)

        # --- Divergence score ---
        # 0 = all agree, 1 = all different
        # Calculated as (num_unique - 1) / (num_systems - 1) when we have at
        # least 2 non-None results.  Systems that returned None are treated as
        # a separate "no winner" bucket.
        num_systems_with_result = len(non_none_winners)
        if num_systems_with_result <= 1:
            divergence_score = 0.0
        else:
            num_unique = len(unique_winners)
            divergence_score = (num_unique - 1) / (num_systems_with_result - 1)

        # --- Consensus winner ---
        # The winner that appears in the most systems.
        consensus_winner: str | None = None
        if non_none_winners:
            from collections import Counter

            counts = Counter(non_none_winners)
            top_count = counts.most_common(1)[0][1]
            # Resolve ties alphabetically for determinism
            top_winners = sorted(w for w, c in counts.items() if c == top_count)
            consensus_winner = top_winners[0] if top_count > 1 else None

        # --- Divergence reasons ---
        divergence_reasons = self._explain_divergences(
            winner_map, results, system_names
        )

        return {
            "winner_map": winner_map,
            "agreement_matrix": agreement_matrix,
            "unique_winners": unique_winners,
            "divergence_score": round(divergence_score, 4),
            "divergence_reasons": divergence_reasons,
            "consensus_winner": consensus_winner,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_round2_fn(
        round1_votes: dict[str, list[int]],
    ) -> Callable[[list[str]], dict[str, list[int]]]:
        """Build a simple round-2 vote function from round-1 data.

        Voters whose round-1 candidate didn't qualify are redistributed
        evenly among the qualifying candidates.  This is a crude
        approximation -- in a real simulation, agents would re-evaluate.
        """

        def fn(round2_candidates: list[str]) -> dict[str, list[int]]:
            r2_votes: dict[str, list[int]] = {c: [] for c in round2_candidates}

            # Voters whose candidate advanced keep their vote.
            orphaned_voters: list[int] = []
            for candidate, voters in round1_votes.items():
                if candidate in r2_votes:
                    r2_votes[candidate].extend(voters)
                else:
                    orphaned_voters.extend(voters)

            # Redistribute orphaned voters round-robin among qualifiers.
            if round2_candidates and orphaned_voters:
                for i, voter in enumerate(orphaned_voters):
                    target = round2_candidates[i % len(round2_candidates)]
                    r2_votes[target].append(voter)

            return r2_votes

        return fn

    @staticmethod
    def _explain_divergences(
        winner_map: dict[str, str | None],
        results: dict,
        system_names: list[str],
    ) -> list[str]:
        """Produce human-readable notes about why systems diverge."""
        reasons: list[str] = []

        winners = {name: winner_map[name] for name in system_names}
        non_none = {n: w for n, w in winners.items() if w is not None}

        if len(set(non_none.values())) <= 1:
            if non_none:
                w = next(iter(non_none.values()))
                reasons.append(f"All systems agree: {w} wins.")
            else:
                reasons.append("No system produced a winner.")
            return reasons

        # FPTP vs RCV divergence
        if (
            winners.get("fptp") is not None
            and winners.get("rcv") is not None
            and winners["fptp"] != winners["rcv"]
        ):
            reasons.append(
                f"FPTP elected {winners['fptp']} but RCV elected "
                f"{winners['rcv']}. This typically happens when the FPTP "
                "plurality leader lacks broad second-choice support, and "
                "RCV's redistribution shifts the outcome."
            )

        # FPTP vs TRS divergence
        if (
            winners.get("fptp") is not None
            and winners.get("trs") is not None
            and winners["fptp"] != winners["trs"]
        ):
            trs_decided_in = results.get("trs", {}).get("decided_in")
            if trs_decided_in == 2:
                reasons.append(
                    f"FPTP elected {winners['fptp']} but TRS elected "
                    f"{winners['trs']} after a runoff. The plurality leader "
                    "could not consolidate support in a head-to-head round 2."
                )
            else:
                reasons.append(
                    f"FPTP elected {winners['fptp']} but TRS elected "
                    f"{winners['trs']} outright in round 1 (different vote "
                    "thresholds apply)."
                )

        # Delegation vs plurality systems
        delegation_winner = winners.get("delegation")
        if delegation_winner is not None:
            # Check if delegation disagrees with all traditional methods
            traditional_winners = {
                n: w
                for n, w in non_none.items()
                if n != "delegation"
            }
            traditional_set = set(traditional_winners.values())
            if delegation_winner not in traditional_set:
                # Gather delegation-specific stats
                deleg_data = results.get("delegation", {})
                gini = deleg_data.get("gini", 0.0)
                max_w = deleg_data.get("max_weight", 0.0)
                chains = deleg_data.get("delegation_chains", 0)
                reasons.append(
                    f"Delegation tally elected {delegation_winner}, disagreeing "
                    f"with all traditional systems. Weight concentration may be "
                    f"a factor (Gini={gini:.3f}, max_weight={max_w:.1f}, "
                    f"delegation_chains={chains})."
                )
            else:
                agreeing = [
                    n for n, w in traditional_winners.items()
                    if w == delegation_winner
                ]
                reasons.append(
                    f"Delegation tally agrees with {', '.join(agreeing)} on "
                    f"winner {delegation_winner}."
                )

        # RCV vs TRS divergence
        if (
            winners.get("rcv") is not None
            and winners.get("trs") is not None
            and winners["rcv"] != winners["trs"]
        ):
            rcv_data = results.get("rcv", {})
            num_rounds = len(rcv_data.get("rounds", []))
            reasons.append(
                f"RCV ({num_rounds} rounds of redistribution) elected "
                f"{winners['rcv']} while TRS elected {winners['trs']}. "
                "RCV redistributes all eliminated candidates' ballots "
                "iteratively, while TRS only holds one runoff between "
                "the top qualifiers -- different consolidation dynamics."
            )

        return reasons
