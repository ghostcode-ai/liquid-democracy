"""Media agents: non-voter actors that inject content into opinion dynamics.

MediaAgent models a broadcast outlet or platform (e.g., cable news, local TV,
social-media feed) that shifts voter ideology through repeated exposure.
AdCampaign models a time-bounded, budget-constrained ad buy that micro-targets
a segment of the electorate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agents.voter_agent import VoterAgent


# ---------------------------------------------------------------------------
# MediaAgent
# ---------------------------------------------------------------------------

@dataclass
class MediaAgent:
    """A media/advertising agent that influences voter opinions.

    Not a voter -- injects influence into the social network.
    """

    agent_id: str  # e.g., "fox_news", "msnbc", "local_tv"
    ideology: np.ndarray  # 10-D, same space as voter agents
    reach: float  # 0-1, fraction of population exposed per tick
    credibility: float  # 0-1, how much agents trust this source
    targeting_precision: float  # 0-1, how well it micro-targets
    budget: float  # simulation units for ad spend

    # Targeting
    target_party: str | None = None  # "D", "R", or None (broad)
    target_demographics: dict = field(default_factory=dict)

    def get_influenced_agents(
        self, agents: dict[int, VoterAgent], rng: np.random.Generator
    ) -> list[int]:
        """Determine which agents are exposed to this media outlet.

        Exposure is stochastic.  Base probability equals ``self.reach``.
        If ``target_party`` is set, aligned agents are 2x more likely to
        be exposed (they self-select into congruent media).  Demographic
        targeting further modulates exposure probability.
        """
        agent_ids: list[int] = []

        for aid, agent in agents.items():
            prob = self.reach

            # Party alignment boosts exposure (self-selection / algorithm)
            if self.target_party is not None:
                party_str = agent.party_id.value  # e.g., "strong_D"
                if self.target_party == "D" and "D" in party_str:
                    prob = min(1.0, prob * 2.0)
                elif self.target_party == "R" and "R" in party_str:
                    prob = min(1.0, prob * 2.0)
                elif "independent" in party_str:
                    # Independents are slightly less likely to tune in
                    prob *= 0.7

            # Demographic targeting: each matching attribute multiplies
            # exposure by (1 + targeting_precision).
            if self.target_demographics:
                demo = agent.demographics
                for key, desired in self.target_demographics.items():
                    actual = getattr(demo, key, None)
                    if actual is not None and actual == desired:
                        prob = min(1.0, prob * (1.0 + self.targeting_precision))

            if rng.random() < prob:
                agent_ids.append(aid)

        return agent_ids

    def compute_influence(self, agent: VoterAgent) -> float:
        """How much this media shifts the agent's opinion.

        Influence is higher when:
        - Agent ideology is already close to media ideology (confirmation bias
          makes congruent messages more persuasive).
        - Agent stubbornness is low.
        - Media credibility is high.

        Returns a scalar in [0, 1] representing shift magnitude.
        """
        # Ideological distance (Euclidean, normalized to [0, 1])
        diff = agent.ideology - self.ideology
        distance = float(np.linalg.norm(diff)) / np.sqrt(len(self.ideology))

        # Confirmation bias: closer ideology -> stronger influence
        # We use a Gaussian kernel so influence drops off with distance.
        alignment_factor = float(np.exp(-2.0 * distance * distance))

        # Stubbornness dampens influence: shift * (1 - stubbornness)
        openness = 1.0 - agent.stubbornness

        # Combine
        raw_influence = self.credibility * alignment_factor * openness

        return float(np.clip(raw_influence, 0.0, 1.0))

    def apply_influence(
        self, agents: dict[int, VoterAgent], rng: np.random.Generator
    ) -> dict[int, float]:
        """Apply media influence to exposed agents.

        For each exposed agent the ideology vector is shifted toward the
        media's ideology by a magnitude proportional to ``compute_influence``.
        A small noise term prevents perfect convergence.

        Returns:
            {agent_id: opinion_shift_magnitude} for every agent that was
            actually influenced (shift > 0).
        """
        exposed = self.get_influenced_agents(agents, rng)
        shifts: dict[int, float] = {}

        for aid in exposed:
            agent = agents[aid]
            influence = self.compute_influence(agent)

            if influence < 1e-6:
                continue

            # Direction: from agent toward media ideology
            direction = self.ideology - agent.ideology

            # Scale by influence magnitude (small step)
            step = direction * influence * 0.05  # 5% max step per tick

            # Add noise so agents don't converge to identical positions
            noise = rng.normal(0, 0.005, size=agent.ideology.shape)
            step = step + noise

            agent.ideology = np.clip(agent.ideology + step, -1.0, 1.0)

            shift_magnitude = float(np.linalg.norm(step))
            shifts[aid] = shift_magnitude

        return shifts


# ---------------------------------------------------------------------------
# AdCampaign
# ---------------------------------------------------------------------------

@dataclass
class AdCampaign:
    """A targeted advertising campaign."""

    sponsor: str
    budget: float  # in simulation dollars
    target_ideology_range: tuple[float, float]  # (min, max) on primary axis
    message_ideology: float  # the opinion the ad pushes toward (-1 to +1)
    precision: float  # 0-1, how precisely it targets
    duration_ticks: int  # how many ticks the campaign runs
    start_tick: int = 0

    def is_active(self, current_tick: int) -> bool:
        """True if the campaign is running at ``current_tick``."""
        return self.start_tick <= current_tick < self.start_tick + self.duration_ticks

    def compute_reach(self, agent: VoterAgent) -> float:
        """Probability that ``agent`` is reached by this ad.

        Higher when the agent's primary ideology dimension (index 0, "economy")
        falls inside the target range.  Precision controls how sharply the
        targeting boundary is enforced -- low precision lets ads leak to
        agents outside the range.
        """
        primary_ideo = float(agent.ideology[0])  # economy dimension
        lo, hi = self.target_ideology_range

        # Distance from the target range center
        center = (lo + hi) / 2.0
        half_width = (hi - lo) / 2.0
        dist_from_center = abs(primary_ideo - center)

        if dist_from_center <= half_width:
            # Inside target range
            base_reach = 0.8
        else:
            # Outside: exponential falloff scaled by inverse precision
            overshoot = dist_from_center - half_width
            falloff_rate = 1.0 + 4.0 * self.precision  # sharper with precision
            base_reach = 0.8 * float(np.exp(-falloff_rate * overshoot))

        # Budget modulates overall reach (normalized: $10k = full power)
        budget_factor = min(1.0, self.budget / 10_000.0)

        return float(np.clip(base_reach * budget_factor, 0.0, 1.0))

    def compute_effect(self, agent: VoterAgent) -> float:
        """How much the ad shifts the agent's primary ideology dimension.

        Effect is proportional to:
        - Distance between agent's position and the message (larger gap =
          more potential movement, but capped by stubbornness).
        - Inverse of agent's stubbornness.
        - Budget (more spend = higher production value / repetition).

        Returns a signed float: positive pushes agent right (+1), negative
        pushes left (-1), matching ``message_ideology`` direction.
        """
        primary_ideo = float(agent.ideology[0])
        gap = self.message_ideology - primary_ideo

        # Openness: (1 - stubbornness) determines max shift
        openness = 1.0 - agent.stubbornness

        # Diminishing returns on budget
        budget_multiplier = min(1.0, np.log1p(self.budget / 1000.0) / 3.0)

        # Raw effect: small fraction of gap, scaled by openness and budget
        raw_effect = gap * 0.03 * openness * budget_multiplier

        # Clamp to prevent extreme single-tick jumps
        max_shift = 0.05
        return float(np.clip(raw_effect, -max_shift, max_shift))


# ---------------------------------------------------------------------------
# Default media environment
# ---------------------------------------------------------------------------

def create_default_media_environment(seed: int = 42) -> list[MediaAgent]:
    """Create a realistic US media landscape.

    Returns ~10 outlets spanning the ideological spectrum, each with
    calibrated reach, credibility, and targeting attributes.
    """
    rng = np.random.default_rng(seed)

    def _ideo(anchor: float, noise: float = 0.15) -> np.ndarray:
        """Build a 10-D ideology vector centered on *anchor* with noise."""
        base = np.full(10, anchor)
        perturbation = rng.normal(0, noise, size=10)
        return np.clip(base + perturbation, -1.0, 1.0)

    outlets: list[MediaAgent] = [
        # --- Left-leaning ---
        MediaAgent(
            agent_id="msnbc",
            ideology=_ideo(-0.7),
            reach=0.25,
            credibility=0.55,
            targeting_precision=0.5,
            budget=50_000.0,
            target_party="D",
        ),
        MediaAgent(
            agent_id="cnn",
            ideology=_ideo(-0.4),
            reach=0.28,
            credibility=0.50,
            targeting_precision=0.4,
            budget=60_000.0,
            target_party="D",
        ),
        MediaAgent(
            agent_id="nyt",
            ideology=_ideo(-0.5),
            reach=0.15,
            credibility=0.70,
            targeting_precision=0.6,
            budget=30_000.0,
            target_party="D",
            target_demographics={"education": 4},  # grad-school readers
        ),

        # --- Right-leaning ---
        MediaAgent(
            agent_id="fox_news",
            ideology=_ideo(0.7),
            reach=0.30,
            credibility=0.50,
            targeting_precision=0.5,
            budget=70_000.0,
            target_party="R",
        ),
        MediaAgent(
            agent_id="daily_wire",
            ideology=_ideo(0.8),
            reach=0.12,
            credibility=0.40,
            targeting_precision=0.7,
            budget=20_000.0,
            target_party="R",
            target_demographics={"age": 30},  # younger right-leaning
        ),
        MediaAgent(
            agent_id="wsj",
            ideology=_ideo(0.4),
            reach=0.13,
            credibility=0.72,
            targeting_precision=0.55,
            budget=35_000.0,
            target_party="R",
            target_demographics={"education": 4},
        ),

        # --- Centrist ---
        MediaAgent(
            agent_id="pbs",
            ideology=_ideo(-0.1),
            reach=0.10,
            credibility=0.80,
            targeting_precision=0.2,
            budget=15_000.0,
            target_party=None,
        ),
        MediaAgent(
            agent_id="ap_reuters",
            ideology=_ideo(0.0),
            reach=0.20,
            credibility=0.85,
            targeting_precision=0.2,
            budget=25_000.0,
            target_party=None,
        ),
        MediaAgent(
            agent_id="bbc_us",
            ideology=_ideo(-0.15),
            reach=0.08,
            credibility=0.78,
            targeting_precision=0.25,
            budget=12_000.0,
            target_party=None,
        ),

        # --- Local / moderate ---
        MediaAgent(
            agent_id="local_tv",
            ideology=_ideo(0.05),
            reach=0.10,
            credibility=0.65,
            targeting_precision=0.1,
            budget=8_000.0,
            target_party=None,
            target_demographics={"urban_rural": "suburban"},
        ),
        MediaAgent(
            agent_id="local_newspaper",
            ideology=_ideo(0.1),
            reach=0.06,
            credibility=0.60,
            targeting_precision=0.15,
            budget=5_000.0,
            target_party=None,
            target_demographics={"urban_rural": "rural"},
        ),
    ]

    return outlets


# ---------------------------------------------------------------------------
# Media cycle runner
# ---------------------------------------------------------------------------

def apply_media_cycle(
    media_agents: list[MediaAgent],
    agents: dict[int, VoterAgent],
    ad_campaigns: list[AdCampaign],
    current_tick: int,
    rng: np.random.Generator,
) -> dict:
    """Run one tick of media influence.

    1. Each media outlet exposes a stochastic subset of voters and shifts
       their ideology vectors.
    2. Active ad campaigns further nudge targeted voters' primary ideology
       dimension.

    Returns:
        dict with keys:
            total_agents_influenced: int
            total_opinion_shift: float (sum of all shift magnitudes)
            media_breakdown: {outlet_id: {"reached": int, "avg_shift": float}}
            ad_breakdown: {sponsor: {"reached": int, "avg_effect": float}}
    """
    media_breakdown: dict[str, dict] = {}
    total_influenced = 0
    total_shift = 0.0

    # --- Media outlets ---
    for outlet in media_agents:
        shifts = outlet.apply_influence(agents, rng)
        reached = len(shifts)
        avg_shift = sum(shifts.values()) / reached if reached else 0.0

        media_breakdown[outlet.agent_id] = {
            "reached": reached,
            "avg_shift": round(avg_shift, 6),
        }

        total_influenced += reached
        total_shift += sum(shifts.values())

    # --- Ad campaigns ---
    ad_breakdown: dict[str, dict] = {}

    for campaign in ad_campaigns:
        if not campaign.is_active(current_tick):
            continue

        reached_count = 0
        total_effect = 0.0

        for aid, agent in agents.items():
            reach_prob = campaign.compute_reach(agent)
            if rng.random() < reach_prob:
                effect = campaign.compute_effect(agent)
                # Apply effect to the primary ideology dimension
                agent.ideology[0] = float(
                    np.clip(agent.ideology[0] + effect, -1.0, 1.0)
                )
                reached_count += 1
                total_effect += abs(effect)

        avg_effect = total_effect / reached_count if reached_count else 0.0

        sponsor_key = campaign.sponsor
        if sponsor_key in ad_breakdown:
            # Multiple campaigns from same sponsor: accumulate
            ad_breakdown[sponsor_key]["reached"] += reached_count
            prev_avg = ad_breakdown[sponsor_key]["avg_effect"]
            prev_count = ad_breakdown[sponsor_key].get("_count", 1)
            new_count = prev_count + 1
            ad_breakdown[sponsor_key]["avg_effect"] = round(
                (prev_avg * prev_count + avg_effect) / new_count, 6
            )
            ad_breakdown[sponsor_key]["_count"] = new_count
        else:
            ad_breakdown[sponsor_key] = {
                "reached": reached_count,
                "avg_effect": round(avg_effect, 6),
                "_count": 1,
            }

        total_influenced += reached_count
        total_shift += total_effect

    # Clean up internal _count keys
    for entry in ad_breakdown.values():
        entry.pop("_count", None)

    return {
        "total_agents_influenced": total_influenced,
        "total_opinion_shift": round(total_shift, 6),
        "media_breakdown": media_breakdown,
        "ad_breakdown": ad_breakdown,
    }
