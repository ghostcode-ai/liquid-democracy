"""Trust formation model for the liquid democracy simulation.

Trust evolves based on:
    trust(i->j, t+1) = alpha * trust(i->j, t)
                      + beta * agreement(i, j)
                      + gamma * social_proof(j)
                      - delta * betrayal(i, j, t)

Default weights: alpha=0.7, beta=0.15, gamma=0.05, delta=0.3
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from agents.voter_agent import VoterAgent


# Default trust model parameters
ALPHA = 0.7  # inertia (prior trust)
BETA = 0.15  # agreement bonus
GAMMA = 0.05  # social proof (popular = trustworthy)
DELTA = 0.3  # betrayal penalty


def compute_agreement(agent_i: VoterAgent, agent_j: VoterAgent) -> float:
    """Ideological agreement between two agents.

    Returns value in [0, 1] where 1 = perfectly aligned.
    Based on cosine similarity of ideology vectors, rescaled to [0, 1].
    """
    ideo_i = agent_i.ideology
    ideo_j = agent_j.ideology

    norm_i = np.linalg.norm(ideo_i)
    norm_j = np.linalg.norm(ideo_j)

    if norm_i < 1e-10 or norm_j < 1e-10:
        return 0.5  # neutral if no ideology signal

    cosine = np.dot(ideo_i, ideo_j) / (norm_i * norm_j)
    # Rescale from [-1, 1] to [0, 1]
    return (cosine + 1.0) / 2.0


def compute_social_proof(
    agent_j: VoterAgent, agents: dict[int, VoterAgent]
) -> float:
    """Social proof: fraction of the population that trusts agent j.

    Returns value in [0, 1].
    """
    if len(agents) <= 1:
        return 0.0

    trust_count = 0
    for aid, agent in agents.items():
        if aid == agent_j.agent_id:
            continue
        if agent.trust_scores.get(agent_j.agent_id, 0) > 0.3:
            trust_count += 1

    return trust_count / (len(agents) - 1)


def compute_betrayal(
    agent_i: VoterAgent,
    agent_j: VoterAgent,
    betrayal_events: dict[tuple[int, int], float],
) -> float:
    """Betrayal score for agent j from agent i's perspective.

    betrayal_events: {(i_id, j_id): severity} where severity in [0, 1].
    A betrayal occurs when a delegate votes against the delegator's preference.
    """
    return betrayal_events.get((agent_i.agent_id, agent_j.agent_id), 0.0)


def update_trust_single(
    agent_i: VoterAgent,
    agent_j: VoterAgent,
    agents: dict[int, VoterAgent],
    betrayal_events: dict[tuple[int, int], float],
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
) -> float:
    """Update trust from agent i toward agent j.

    Returns new trust score in [0, 1].
    """
    prior_trust = agent_i.trust_scores.get(agent_j.agent_id, 0.3)
    agreement = compute_agreement(agent_i, agent_j)
    social_proof = compute_social_proof(agent_j, agents)
    betrayal = compute_betrayal(agent_i, agent_j, betrayal_events)

    new_trust = (
        alpha * prior_trust
        + beta * agreement
        + gamma * social_proof
        - delta * betrayal
    )

    # Clamp to [0, 1]
    return max(0.0, min(1.0, new_trust))


def update_all_trust(
    agents: dict[int, VoterAgent],
    social_graph: nx.Graph,
    betrayal_events: Optional[dict[tuple[int, int], float]] = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
) -> None:
    """Update trust scores for all agents toward their neighbors.

    Modifies agents in place.
    """
    if betrayal_events is None:
        betrayal_events = {}

    for aid, agent in agents.items():
        if aid not in social_graph:
            continue

        for neighbor_id in social_graph.neighbors(aid):
            if neighbor_id not in agents:
                continue

            new_trust = update_trust_single(
                agent,
                agents[neighbor_id],
                agents,
                betrayal_events,
                alpha,
                beta,
                gamma,
                delta,
            )
            agent.trust_scores[neighbor_id] = new_trust


def detect_betrayals(
    agents: dict[int, VoterAgent],
    race_id: str,
    delegation_edges: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Detect betrayals: delegate voted against delegator's preference.

    Returns {(delegator_id, delegate_id): severity}.
    Severity = abs difference in preferences, normalized to [0, 1].
    """
    betrayals: dict[tuple[int, int], float] = {}

    for delegator_id, delegate_id in delegation_edges:
        delegator = agents.get(delegator_id)
        delegate = agents.get(delegate_id)

        if delegator is None or delegate is None:
            continue

        d_state = delegator.race_states.get(race_id)
        e_state = delegate.race_states.get(race_id)

        if d_state is None or e_state is None:
            continue

        # Betrayal = delegate voted opposite to delegator's preference
        pref_diff = abs(d_state.preference - e_state.preference)
        if pref_diff > 0.5:  # threshold for "betrayal"
            severity = min(1.0, pref_diff / 2.0)
            betrayals[(delegator_id, delegate_id)] = severity

    return betrayals

