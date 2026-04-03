"""Friedkin-Johnsen opinion dynamics with bounded confidence filter.

Implements the opinion update rule:
    x(t+1) = Lambda * x(0) + (I - Lambda) * W * x(t)

Where:
    Lambda = diagonal matrix of stubbornness (0=fully susceptible, 1=immovable)
    W = row-stochastic influence matrix (from social network + trust)
    x(0) = initial opinions
    x(t) = current opinions

Bounded confidence: agents only update from peers within epsilon distance.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from agents.voter_agent import VoterAgent


def build_influence_matrix(
    agents: dict[int, VoterAgent],
    social_graph: nx.Graph,
    dimension: int = 0,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    """Build row-stochastic influence matrix W from social graph and trust.

    Args:
        agents: {agent_id: VoterAgent}
        social_graph: undirected social network
        dimension: which ideology dimension to use for bounded confidence
        epsilon: bounded confidence threshold (None = no filter)

    Returns:
        W: (n, n) row-stochastic matrix
    """
    agent_ids = sorted(agents.keys())
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    n = len(agent_ids)
    W = np.zeros((n, n))

    for aid in agent_ids:
        i = id_to_idx[aid]
        agent = agents[aid]

        # Get neighbors in social graph
        if aid not in social_graph:
            W[i, i] = 1.0  # self-loop if isolated
            continue

        neighbors = list(social_graph.neighbors(aid))
        if not neighbors:
            W[i, i] = 1.0
            continue

        # Filter by bounded confidence
        if epsilon is not None:
            agent_opinion = agent.ideology[dimension]
            neighbors = [
                nid
                for nid in neighbors
                if nid in agents
                and abs(agents[nid].ideology[dimension] - agent_opinion) < epsilon
            ]

        if not neighbors:
            W[i, i] = 1.0
            continue

        # Weight by trust scores (default to 0.5 if no trust data)
        # Include self-weight so the matrix is well-behaved
        weights = [(i, 0.5)]  # self-influence
        for nid in neighbors:
            if nid not in id_to_idx:
                continue
            trust = agent.trust_scores.get(nid, 0.5)
            weights.append((id_to_idx[nid], max(trust, 0.01)))

        # Normalize to row-stochastic
        total = sum(w for _, w in weights)
        for j, w in weights:
            W[i, j] = w / total

    return W


def build_stubbornness_matrix(agents: dict[int, VoterAgent]) -> np.ndarray:
    """Build diagonal stubbornness matrix Lambda.

    Lambda[i,i] = stubbornness of agent i (0 = fully susceptible, 1 = immovable).
    """
    agent_ids = sorted(agents.keys())
    n = len(agent_ids)
    Lambda = np.zeros((n, n))
    for i, aid in enumerate(agent_ids):
        Lambda[i, i] = agents[aid].stubbornness
    return Lambda


def extract_opinions(
    agents: dict[int, VoterAgent], dimension: int = 0
) -> np.ndarray:
    """Extract opinion vector for a single ideology dimension."""
    agent_ids = sorted(agents.keys())
    return np.array([agents[aid].ideology[dimension] for aid in agent_ids])


def friedkin_johnsen_step(
    x_current: np.ndarray,
    x_initial: np.ndarray,
    Lambda: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Single Friedkin-Johnsen update step.

    x(t+1) = Lambda * x(0) + (I - Lambda) * W * x(t)
    """
    n = len(x_current)
    I = np.eye(n)
    return Lambda @ x_initial + (I - Lambda) @ W @ x_current


def run_opinion_dynamics(
    agents: dict[int, VoterAgent],
    social_graph: nx.Graph,
    dimension: int = 0,
    n_steps: int = 80,
    epsilon: Optional[float] = None,
    rebuild_W_every: int = 10,
) -> list[np.ndarray]:
    """Run Friedkin-Johnsen dynamics for multiple steps.

    Args:
        agents: {agent_id: VoterAgent}
        social_graph: undirected social network
        dimension: ideology dimension to update
        n_steps: number of update steps
        epsilon: bounded confidence threshold
        rebuild_W_every: rebuild influence matrix every N steps
            (captures evolving trust)

    Returns:
        history: list of opinion vectors at each step
    """
    agent_ids = sorted(agents.keys())
    x_initial = extract_opinions(agents, dimension)
    x_current = x_initial.copy()
    Lambda = build_stubbornness_matrix(agents)

    history = [x_initial.copy()]

    W = build_influence_matrix(agents, social_graph, dimension, epsilon)

    for step in range(n_steps):
        # Periodically rebuild W to capture trust evolution
        if step > 0 and step % rebuild_W_every == 0:
            W = build_influence_matrix(agents, social_graph, dimension, epsilon)

        x_current = friedkin_johnsen_step(x_current, x_initial, Lambda, W)

        # Clamp to [-1, 1]
        x_current = np.clip(x_current, -1.0, 1.0)

        history.append(x_current.copy())

    # Write final opinions back to agents
    for i, aid in enumerate(agent_ids):
        agents[aid].ideology[dimension] = x_current[i]

    return history


def run_all_dimensions(
    agents: dict[int, VoterAgent],
    social_graph: nx.Graph,
    n_steps: int = 80,
    epsilon: Optional[float] = None,
    dimensions: Optional[list[int]] = None,
) -> dict[int, list[np.ndarray]]:
    """Run opinion dynamics on multiple ideology dimensions.

    Returns {dimension: [opinion_history]}.
    """
    if dimensions is None:
        dimensions = list(range(10))

    results = {}
    for dim in dimensions:
        results[dim] = run_opinion_dynamics(
            agents, social_graph, dimension=dim, n_steps=n_steps, epsilon=epsilon
        )

    return results
