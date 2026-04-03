"""VoterAgent: core agent model for the liquid democracy simulation.

Each agent represents a voter with demographics, ideology, behavioral parameters,
trust network, and per-race voting state. Derived from CES survey data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# -- Ideology dimensions (10-D vector, each in [-1, +1]) --
IDEOLOGY_DIMS = [
    "economy",
    "social",
    "immigration",
    "foreign_policy",
    "guns",
    "environment",
    "healthcare",
    "criminal_justice",
    "trade",
    "government_size",
]


class PartyID(str, Enum):
    STRONG_D = "strong_D"
    LEAN_D = "lean_D"
    INDEPENDENT = "independent"
    LEAN_R = "lean_R"
    STRONG_R = "strong_R"


class VoteAction(str, Enum):
    CAST_VOTE = "vote"
    DELEGATE = "delegate"
    ABSTAIN = "abstain"


# Canonical candidate ideological positions (shared with simulation engine)
_AGENT_CANDIDATE_IDEOLOGIES: dict[str, float] = {
    "Democrat": -0.6, "Republican": 0.6, "Libertarian": 0.4,
    "Green": -0.8, "Independent": 0.0,
}

# Partisan anchoring: how much party loyalty biases candidate choice.
# The bonus reduces effective distance to the party's candidate, making
# defection require a meaningful ideology shift — but not an impossible one.
# At 0.15/0.25, strong partisans still defect if a third-party candidate is
# very close ideologically, and moderate opinion shifts can flip leaners.
_PARTY_ANCHOR: dict[PartyID, tuple[str, float]] = {
    PartyID.STRONG_D: ("Democrat", 0.25),
    PartyID.LEAN_D: ("Democrat", 0.10),
    PartyID.INDEPENDENT: ("Independent", 0.0),
    PartyID.LEAN_R: ("Republican", 0.10),
    PartyID.STRONG_R: ("Republican", 0.25),
}


@dataclass
class Demographics:
    age: int
    income: float  # annual, USD
    education: int  # 0-5 scale (no HS, HS, some college, BA, MA, PhD)
    race: str
    gender: str
    urban_rural: str  # "urban", "suburban", "rural"


@dataclass
class RaceState:
    """Per-race state for an agent."""

    knowledge_level: float = 0.0  # 0-1
    preference: float = 0.0  # -1 (D) to +1 (R)
    delegation_targets: list[int] = field(default_factory=list)
    voted: bool = False
    vote_choice: Optional[str] = None
    ranked_choices: list[str] = field(default_factory=list)  # for RCV


@dataclass
class ActionResult:
    """Result of an agent's voting decision."""

    action: VoteAction
    race_id: str
    choice: Optional[str] = None  # candidate name or delegate agent_id
    ranked_choices: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class VoterAgent:
    """A voter in the liquid democracy simulation.

    Identity is seeded from CES data. Behavioral parameters are derived
    from demographics. Trust network emerges from opinion dynamics.
    """

    agent_id: int
    demographics: Demographics
    ideology: np.ndarray  # shape (10,), values in [-1, +1]
    party_id: PartyID

    # Behavioral parameters (derived from demographics)
    political_knowledge: float = 0.5  # 0-1
    engagement_level: float = 0.5  # 0-1
    stubbornness: float = 0.5  # 0-1 (Friedkin-Johnsen anchoring)
    social_desirability_gap: float = 0.03  # sigma for N(0, sigma)

    # Trust network (emergent)
    trust_scores: dict[int, float] = field(default_factory=dict)
    delegation_inertia: float = 0.7  # P(NOT revoking stale delegation)

    # Per-race state
    race_states: dict[str, RaceState] = field(default_factory=dict)

    # LLM flag
    is_llm_agent: bool = False

    # Network neighbors (set by social network builder)
    neighbors: list[int] = field(default_factory=list)

    # --- Behavioral parameter derivation ---

    @staticmethod
    def derive_knowledge(education: int) -> float:
        """Knowledge = 0.3 + 0.4 * (education / 5)."""
        return 0.3 + 0.4 * (education / 5.0)

    @staticmethod
    def derive_engagement(age: int, income: float, education: int) -> float:
        """Logistic engagement from age, income, education."""
        # Normalize inputs
        age_norm = min(age, 90) / 90.0
        income_norm = min(income, 200_000) / 200_000.0
        edu_norm = education / 5.0
        z = -2.0 + 2.0 * age_norm + 1.5 * income_norm + 1.5 * edu_norm
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derive_stubbornness(party_id: PartyID) -> float:
        """Stubbornness from partisan strength.

        Lower values = more susceptible to peer influence.
        Strong partisans (0.55) still resist but visibly shift over 80 ticks.
        Independents (0.10) are highly persuadable.
        """
        strength = {
            PartyID.STRONG_D: 1.0,
            PartyID.STRONG_R: 1.0,
            PartyID.LEAN_D: 0.5,
            PartyID.LEAN_R: 0.5,
            PartyID.INDEPENDENT: 0.0,
        }
        return 0.10 + 0.45 * strength[party_id]

    @classmethod
    def from_profile(
        cls,
        agent_id: int,
        demographics: Demographics,
        ideology: np.ndarray,
        party_id: PartyID,
        is_llm: bool = False,
    ) -> VoterAgent:
        """Create a VoterAgent with behavioral parameters derived from demographics."""
        return cls(
            agent_id=agent_id,
            demographics=demographics,
            ideology=ideology,
            party_id=party_id,
            political_knowledge=cls.derive_knowledge(demographics.education),
            engagement_level=cls.derive_engagement(
                demographics.age, demographics.income, demographics.education
            ),
            stubbornness=cls.derive_stubbornness(party_id),
            social_desirability_gap=abs(np.random.normal(0, 0.03)),
            is_llm_agent=is_llm,
        )

    # --- Decision logic ---

    def get_private_preference(self, race_id: str) -> float:
        """Private preference diverges from public by social desirability gap."""
        state = self.race_states.get(race_id)
        if state is None:
            return 0.0
        return state.preference + np.random.normal(0, self.social_desirability_gap)

    def find_best_delegate(self, race_id: str) -> Optional[int]:
        """Find the most trusted neighbor for a given race."""
        top = self.find_best_k_delegates(race_id, k=1)
        return top[0] if top else None

    def find_best_k_delegates(self, race_id: str, k: int = 1) -> list[int]:
        """Find the k most trusted neighbors for a given race."""
        if not self.trust_scores or k <= 0:
            return []
        candidates = {
            aid: score
            for aid, score in self.trust_scores.items()
            if aid in self.neighbors
        }
        if not candidates:
            return []
        sorted_by_trust = sorted(candidates.items(), key=lambda x: -x[1])
        return [aid for aid, _ in sorted_by_trust[:k]]

    # ------------------------------------------------------------------
    # Decoupled abstention model
    # ------------------------------------------------------------------

    def abstain_apathy_prob(self) -> float:
        """Disinterest: low engagement -> higher chance of not bothering.

        ~5% at engagement=0, ~0% at engagement=1.
        """
        return 0.05 * (1.0 - self.engagement_level)

    def abstain_futility_prob(self, pvi_lean: float) -> float:
        """Strategic futility: being a strong minority in a lopsided district.

        A Republican in a D+30 district thinks "why bother?"
        ~0% in competitive districts, up to ~5% at |PVI|=30.
        """
        agent_lean = (
            -1.0 if self.party_id in (PartyID.STRONG_D, PartyID.LEAN_D)
            else (1.0 if self.party_id in (PartyID.STRONG_R, PartyID.LEAN_R) else 0.0)
        )
        # Positive when agent opposes district lean
        misalignment = max(0.0, -agent_lean * pvi_lean)
        return min(0.05, misalignment * 0.05 / 30.0)

    def abstain_efficacy_prob(self) -> float:
        """Lack of efficacy: low knowledge -> "my vote doesn't matter."

        ~4% at knowledge=0, ~0% at knowledge>=0.7.
        """
        return max(0.0, 0.04 * (1.0 - self.political_knowledge / 0.7))

    def abstain_access_prob(self) -> float:
        """Access barriers: low income + rural -> harder to get to polls.

        ~3% worst-case (low income, rural), ~0% for high-income urban.
        """
        income_factor = max(0.0, 1.0 - self.demographics.income / 80_000)
        rural_factor = (
            1.0 if self.demographics.urban_rural == "rural"
            else (0.3 if self.demographics.urban_rural == "suburban" else 0.0)
        )
        return 0.03 * income_factor * (0.5 + 0.5 * rural_factor)

    def check_abstention(
        self, pvi_lean: float = 0.0, rng: Optional[np.random.Generator] = None,
    ) -> tuple[bool, str]:
        """Run independent abstention checks. Returns (True, reason) if any fires."""
        _rand = rng.random if rng is not None else np.random.random
        checks = [
            (self.abstain_apathy_prob(), "apathy"),
            (self.abstain_futility_prob(pvi_lean), "futility"),
            (self.abstain_efficacy_prob(), "efficacy"),
            (self.abstain_access_prob(), "access_barrier"),
        ]
        for prob, reason in checks:
            if _rand() < prob:
                return True, reason
        return False, ""

    def roll_off_probability(self, race_type: str) -> float:
        """Legacy roll-off rate (kept for backward compatibility)."""
        rates = {
            "senate": 0.01, "house": 0.03, "state_legislature": 0.15,
            "judicial": 0.23, "proposition": 0.20, "local": 0.31,
        }
        base_rate = rates.get(race_type, 0.10)
        return base_rate * (2.0 - self.engagement_level)

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide_action(
        self,
        race_id: str,
        race_type: str = "house",
        knowledge_threshold: float = 0.6,
        delegation_threshold: float = 0.5,
        candidates: Optional[list[str]] = None,
        pvi_lean: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> ActionResult:
        """Core agent decision: vote directly, delegate, or abstain.

        Decision flow (abstention decoupled from vote choice):
        1. Abstention gate — independent checks for apathy, futility,
           efficacy, and access barriers.
        2. Knowledge check — informed agents vote directly.
        3. Delegation check — agents with a trusted delegate delegate.
        4. Party-line fallback — remaining agents vote by party heuristic.
        """
        state = self.race_states.get(race_id)
        if state is None:
            state = RaceState()
            self.race_states[race_id] = state

        # 1. Abstention gate (checked BEFORE vote choice)
        abstained, reason = self.check_abstention(pvi_lean, rng=rng)
        if abstained:
            return ActionResult(
                action=VoteAction.ABSTAIN,
                race_id=race_id,
                reason=reason,
            )

        # 2. High knowledge → vote directly
        if state.knowledge_level > knowledge_threshold:
            private_pref = self.get_private_preference(race_id)
            choice = self._preference_to_candidate(private_pref, candidates)
            return ActionResult(
                action=VoteAction.CAST_VOTE,
                race_id=race_id,
                choice=choice,
                reason="sufficient knowledge to vote directly",
            )

        # 3. Check for trusted delegate
        best_delegate = self.find_best_delegate(race_id)
        if (
            best_delegate is not None
            and self.trust_scores.get(best_delegate, 0) > delegation_threshold
        ):
            return ActionResult(
                action=VoteAction.DELEGATE,
                race_id=race_id,
                choice=str(best_delegate),
                reason="delegating to trusted contact",
            )

        # 4. Fallback: party-line heuristic
        choice = self._party_heuristic(candidates)
        return ActionResult(
            action=VoteAction.CAST_VOTE,
            race_id=race_id,
            choice=choice,
            reason="party-line fallback",
        )

    def _preference_to_candidate(
        self, preference: float, candidates: Optional[list[str]]
    ) -> Optional[str]:
        """Pick the candidate closest to the agent's ideology, anchored by party loyalty.

        Uses ideology distance to known candidate positions. Strong partisans
        get a distance penalty applied to all candidates except their party's,
        making defection require a large ideology shift (realistic).
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        anchor_candidate, anchor_bonus = _PARTY_ANCHOR.get(
            self.party_id, ("Independent", 0.0)
        )

        best_candidate = None
        best_score = float("inf")
        for c in candidates:
            c_ideo = _AGENT_CANDIDATE_IDEOLOGIES.get(c, 0.0)
            dist = abs(preference - c_ideo)
            # Partisan anchoring: reduce effective distance to party's candidate
            if c == anchor_candidate:
                dist = max(0.0, dist - anchor_bonus)
            best_score, best_candidate = min(
                (best_score, best_candidate), (dist, c), key=lambda x: x[0]
            )
        return best_candidate

    def _party_heuristic(self, candidates: Optional[list[str]]) -> Optional[str]:
        """Vote for party-aligned candidate using ideology distance."""
        if not candidates:
            return None
        # Use the same ideology-distance logic with full partisan anchoring
        return self._preference_to_candidate(float(self.ideology[0]), candidates)
