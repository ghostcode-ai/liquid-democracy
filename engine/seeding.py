"""Agent seeding engine — synthetic and real-data modes.

Synthetic mode: generates agents from parametric distributions (Cook PVI lean).
Real-data mode: uses actual 2024 district profiles (demographics, PVI, results)
                to calibrate the agent population.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from agents.voter_agent import Demographics, PartyID, VoterAgent


# -- Party distribution by Cook PVI lean --
# PVI is expressed as D+X or R+X. We model as a float: negative=D, positive=R.
# E.g., D+15 = -15, R+8 = +8, EVEN = 0.


def pvi_to_party_probs(pvi_lean: float) -> dict[PartyID, float]:
    """Convert Cook PVI lean to party ID distribution.

    Args:
        pvi_lean: negative = D lean, positive = R lean. Range roughly [-30, +30].
    """
    # Sigmoid mapping: PVI lean shifts the party distribution
    lean_normalized = pvi_lean / 30.0  # normalize to roughly [-1, 1]

    # Base probabilities at PVI=0 (national average)
    base = {
        PartyID.STRONG_D: 0.20,
        PartyID.LEAN_D: 0.12,
        PartyID.INDEPENDENT: 0.16,
        PartyID.LEAN_R: 0.12,
        PartyID.STRONG_R: 0.20,
    }

    # Shift: negative lean_normalized (D lean) should boost D probs
    # positive lean_normalized (R lean) should boost R probs
    shift = lean_normalized * 0.15
    probs = {
        PartyID.STRONG_D: base[PartyID.STRONG_D] - shift * 1.5,
        PartyID.LEAN_D: base[PartyID.LEAN_D] - shift * 0.5,
        PartyID.INDEPENDENT: base[PartyID.INDEPENDENT] - abs(shift) * 0.3,
        PartyID.LEAN_R: base[PartyID.LEAN_R] + shift * 0.5,
        PartyID.STRONG_R: base[PartyID.STRONG_R] + shift * 1.5,
    }

    # Clamp and normalize
    probs = {k: max(0.02, v) for k, v in probs.items()}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def generate_demographics(
    rng: np.random.Generator, party_id: PartyID
) -> Demographics:
    """Generate correlated demographics for a given party ID.

    Preserves realistic correlations: e.g., education and income are correlated,
    age distribution differs by party.
    """
    # Age: Republicans skew older, Democrats skew younger
    age_shift = {"strong_D": -3, "lean_D": -1, "independent": 0, "lean_R": 2, "strong_R": 5}
    age = int(rng.normal(45 + age_shift.get(party_id.value, 0), 15))
    age = max(18, min(95, age))

    # Education: slight D lean at higher education
    edu_base = 2.5
    if party_id in (PartyID.STRONG_D, PartyID.LEAN_D):
        edu_base = 2.8
    elif party_id in (PartyID.STRONG_R, PartyID.LEAN_R):
        edu_base = 2.3
    education = int(rng.normal(edu_base, 1.2))
    education = max(0, min(5, education))

    # Income: correlated with education + age
    income_base = 30_000 + education * 15_000 + max(0, age - 25) * 500
    income = max(10_000, rng.normal(income_base, 20_000))

    # Race: simplified distribution
    race_probs = {"white": 0.60, "black": 0.13, "hispanic": 0.18, "asian": 0.06, "other": 0.03}
    race = rng.choice(list(race_probs.keys()), p=list(race_probs.values()))

    # Gender
    gender = rng.choice(["male", "female"], p=[0.48, 0.52])

    # Urban/rural: Democrats more urban
    if party_id in (PartyID.STRONG_D, PartyID.LEAN_D):
        urban_probs = [0.55, 0.30, 0.15]
    elif party_id in (PartyID.STRONG_R, PartyID.LEAN_R):
        urban_probs = [0.25, 0.35, 0.40]
    else:
        urban_probs = [0.40, 0.35, 0.25]
    urban_rural = rng.choice(["urban", "suburban", "rural"], p=urban_probs)

    return Demographics(
        age=age,
        income=float(income),
        education=education,
        race=str(race),
        gender=str(gender),
        urban_rural=str(urban_rural),
    )


def generate_ideology(
    rng: np.random.Generator,
    party_id: PartyID,
    ideology_std: float = 0.3,
) -> np.ndarray:
    """Generate a 10-D ideology vector correlated with party ID.

    Dimensions: economy, social, immigration, foreign_policy, guns,
    environment, healthcare, criminal_justice, trade, government_size.
    Convention: -1 = left/liberal, +1 = right/conservative.
    """
    # Party-based mean ideology
    party_means = {
        PartyID.STRONG_D: -0.7,
        PartyID.LEAN_D: -0.35,
        PartyID.INDEPENDENT: 0.0,
        PartyID.LEAN_R: 0.35,
        PartyID.STRONG_R: 0.7,
    }
    base_mean = party_means[party_id]

    # Per-dimension offsets (parties aren't uniform across issues)
    dim_offsets = np.array([
        0.0,    # economy: on-party
        0.05,   # social: slightly more conservative than econ for D
        -0.05,  # immigration: slightly more liberal
        0.0,    # foreign policy
        -0.10,  # guns: skews liberal
        0.05,   # environment: skews conservative
        0.0,    # healthcare
        0.0,    # criminal justice
        0.0,    # trade
        0.0,    # government size
    ])

    means = np.full(10, base_mean) + dim_offsets * np.sign(base_mean)
    ideology = rng.normal(means, ideology_std)
    return np.clip(ideology, -1.0, 1.0)


def build_social_network(
    agents: dict[int, VoterAgent],
    k: int = 8,
    p: float = 0.1,
    homophily: float = 0.65,
    rng: Optional[np.random.Generator] = None,
) -> nx.Graph:
    """Build Watts-Strogatz small-world network with political homophily.

    1. Start with Watts-Strogatz (n, k, p)
    2. Rewire 65% of edges to same-party connections
    3. Keep 10-15% cross-partisan weak ties

    Args:
        agents: {agent_id: VoterAgent}
        k: each node connected to k nearest neighbors
        p: rewiring probability
        homophily: fraction of edges rewired to same-party
        rng: random generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng()

    agent_ids = sorted(agents.keys())
    n = len(agent_ids)

    if n < k + 1:
        # Too few agents for Watts-Strogatz; use complete graph
        G = nx.complete_graph(n)
        mapping = {i: agent_ids[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        _assign_neighbors(agents, G)
        return G

    # Create Watts-Strogatz base graph
    G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(0, 2**31)))

    # Relabel nodes to agent IDs
    mapping = {i: agent_ids[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    # Homophily rewiring: replace some cross-party edges with same-party edges
    _apply_homophily(G, agents, homophily, rng)

    # Assign neighbor lists to agents
    _assign_neighbors(agents, G)

    return G


def _apply_homophily(
    G: nx.Graph,
    agents: dict[int, VoterAgent],
    homophily: float,
    rng: np.random.Generator,
) -> None:
    """Rewire cross-party edges to same-party connections."""
    # Group agents by party leaning (D, I, R)
    party_groups: dict[str, list[int]] = {"D": [], "I": [], "R": []}
    for aid, agent in agents.items():
        if agent.party_id in (PartyID.STRONG_D, PartyID.LEAN_D):
            party_groups["D"].append(aid)
        elif agent.party_id in (PartyID.STRONG_R, PartyID.LEAN_R):
            party_groups["R"].append(aid)
        else:
            party_groups["I"].append(aid)

    def get_party(aid: int) -> str:
        agent = agents[aid]
        if agent.party_id in (PartyID.STRONG_D, PartyID.LEAN_D):
            return "D"
        elif agent.party_id in (PartyID.STRONG_R, PartyID.LEAN_R):
            return "R"
        return "I"

    edges_to_remove = []
    edges_to_add = []

    for u, v in list(G.edges()):
        if get_party(u) != get_party(v):
            if rng.random() < homophily:
                # Try to rewire to same-party
                party = get_party(u)
                same_party = party_groups[party]
                if len(same_party) > 1:
                    # Pick a random same-party agent not already connected
                    attempts = 0
                    while attempts < 5:
                        new_v = rng.choice(same_party)
                        if new_v != u and not G.has_edge(u, new_v):
                            edges_to_remove.append((u, v))
                            edges_to_add.append((u, new_v))
                            break
                        attempts += 1

    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)


def _assign_neighbors(agents: dict[int, VoterAgent], G: nx.Graph) -> None:
    """Write neighbor lists from graph back to agent objects."""
    for aid in agents:
        if aid in G:
            agents[aid].neighbors = list(G.neighbors(aid))
        else:
            agents[aid].neighbors = []


def seed_agents(
    n_agents: int = 10_000,
    pvi_lean: float = 0.0,
    ideology_std: float = 0.3,
    seed: Optional[int] = None,
) -> dict[int, VoterAgent]:
    """Generate a synthetic agent population for a district.

    Args:
        n_agents: number of agents to create
        pvi_lean: Cook PVI lean (negative=D, positive=R)
        ideology_std: standard deviation for ideology vectors
        seed: random seed for reproducibility

    Returns:
        {agent_id: VoterAgent}
    """
    rng = np.random.default_rng(seed)
    party_probs = pvi_to_party_probs(pvi_lean)
    parties = list(party_probs.keys())
    probs = list(party_probs.values())

    agents: dict[int, VoterAgent] = {}

    for i in range(n_agents):
        party_id_str = rng.choice([p.value for p in parties], p=probs)
        party_id = PartyID(party_id_str)
        demographics = generate_demographics(rng, party_id)
        ideology = generate_ideology(rng, party_id, ideology_std)

        agent = VoterAgent.from_profile(
            agent_id=i,
            demographics=demographics,
            ideology=ideology,
            party_id=party_id,
        )
        agents[i] = agent

    return agents


# ---------------------------------------------------------------------------
# Real-data seeding from district profiles
# ---------------------------------------------------------------------------


def seed_from_district(
    district_id: str,
    n_agents: int = 10_000,
    ideology_std: float = 0.25,
    seed: Optional[int] = None,
) -> dict[int, VoterAgent]:
    """Generate agents calibrated to a real congressional district.

    Uses actual 2024 demographics, Cook PVI, and election results
    from the district profile to build a realistic electorate.

    Args:
        district_id: e.g., "CA-27", "OH-SEN"
        n_agents: population size
        ideology_std: ideology spread (lower = more homogeneous)
        seed: random seed

    Returns:
        {agent_id: VoterAgent}

    Raises:
        ValueError: if district_id is not found
    """
    from data.districts import get_district

    profile = get_district(district_id)
    if profile is None:
        from data.districts import list_district_ids
        available = ", ".join(list_district_ids()[:10]) + "..."
        raise ValueError(
            f"Unknown district '{district_id}'. Available: {available}"
        )

    rng = np.random.default_rng(seed)

    # Use real PVI for party distribution
    party_probs = pvi_to_party_probs(profile.pvi)
    parties = list(party_probs.keys())
    probs = list(party_probs.values())

    agents: dict[int, VoterAgent] = {}

    for i in range(n_agents):
        party_id_str = rng.choice([p.value for p in parties], p=probs)
        party_id = PartyID(party_id_str)

        # Generate demographics calibrated to the real district
        demographics = _generate_district_demographics(rng, party_id, profile)
        ideology = generate_ideology(rng, party_id, ideology_std)

        agent = VoterAgent.from_profile(
            agent_id=i,
            demographics=demographics,
            ideology=ideology,
            party_id=party_id,
        )
        agents[i] = agent

    return agents


def _generate_district_demographics(
    rng: np.random.Generator,
    party_id: PartyID,
    profile,
) -> Demographics:
    """Generate demographics calibrated to a real district profile.

    Uses the district's actual demographic distributions (median age,
    income, education rates, racial composition, urbanization) instead
    of national averages.
    """
    # Age: centered on district median, party-shifted
    age_shift = {
        "strong_D": -3, "lean_D": -1, "independent": 0,
        "lean_R": 2, "strong_R": 5,
    }
    age = int(rng.normal(profile.median_age + age_shift.get(party_id.value, 0), 12))
    age = max(18, min(95, age))

    # Education: calibrated to district college rate
    # pct_college maps roughly to mean education level on 0-5 scale
    edu_mean = 1.5 + 3.5 * profile.pct_college  # 0.18 -> 2.1, 0.58 -> 3.5
    if party_id in (PartyID.STRONG_D, PartyID.LEAN_D):
        edu_mean += 0.3  # D skews slightly more educated
    elif party_id in (PartyID.STRONG_R, PartyID.LEAN_R):
        edu_mean -= 0.2
    education = int(rng.normal(edu_mean, 1.0))
    education = max(0, min(5, education))

    # Income: centered on district median, correlated with education
    income_base = profile.median_income + (education - 3) * 12_000
    income = max(10_000, rng.normal(income_base, profile.median_income * 0.35))

    # Race: use district racial composition directly
    race_probs = {
        "white": profile.pct_white,
        "black": profile.pct_black,
        "hispanic": profile.pct_hispanic,
        "asian": profile.pct_asian,
        "other": max(0.01, 1.0 - profile.pct_white - profile.pct_black
                     - profile.pct_hispanic - profile.pct_asian),
    }
    # Normalize
    total = sum(race_probs.values())
    race_probs = {k: v / total for k, v in race_probs.items()}
    race = rng.choice(list(race_probs.keys()), p=list(race_probs.values()))

    # Gender
    gender = rng.choice(["male", "female"], p=[0.48, 0.52])

    # Urban/rural: use district urbanization
    urban_p = profile.pct_urban
    suburban_p = min(0.40, (1.0 - urban_p) * 0.6)
    rural_p = max(0.0, 1.0 - urban_p - suburban_p)
    total_ur = urban_p + suburban_p + rural_p
    urban_rural = rng.choice(
        ["urban", "suburban", "rural"],
        p=[urban_p / total_ur, suburban_p / total_ur, rural_p / total_ur],
    )

    return Demographics(
        age=age,
        income=float(income),
        education=education,
        race=str(race),
        gender=str(gender),
        urban_rural=str(urban_rural),
    )
