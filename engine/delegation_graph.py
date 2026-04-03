"""Delegation graph for liquid democracy simulation.

Wraps a NetworkX DiGraph to model voter-to-delegate delegation chains
with per-topic edges, cycle detection, viscous decay, and weight capping.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx


class DelegationGraph:
    """Directed graph of voter delegations, keyed by topic."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_delegation(
        self, voter_id: str, delegate_id: str, topic: str, fraction: float = 1.0,
    ) -> None:
        """Add a directed delegation edge from *voter_id* to *delegate_id* on *topic*.

        When ``fraction`` is 1.0 (default, k=1), any existing delegation for
        this voter+topic is replaced.  When ``fraction`` < 1.0 (k>1), multiple
        outgoing edges are allowed so the vote is split among delegates.
        """
        if fraction >= 1.0:
            self.revoke(voter_id, topic)
        self._graph.add_edge(voter_id, delegate_id, topic=topic, fraction=fraction)

    def revoke(self, voter_id: str, topic: str) -> None:
        """Remove the delegation for *voter_id* on *topic*, if one exists."""
        if voter_id not in self._graph:
            return
        to_remove: list[tuple[str, str]] = []
        for _, target, data in self._graph.out_edges(voter_id, data=True):
            if data.get("topic") == topic:
                to_remove.append((voter_id, target))
        for edge in to_remove:
            self._graph.remove_edge(*edge)

    # ------------------------------------------------------------------
    # Topic subgraph helpers
    # ------------------------------------------------------------------

    def _topic_subgraph(self, topic: str) -> nx.DiGraph:
        """Return a DiGraph containing only edges for *topic*."""
        sub = nx.DiGraph()
        for u, v, data in self._graph.edges(data=True):
            if data.get("topic") == topic:
                sub.add_edge(u, v)
        return sub

    def _all_topic_nodes(self, topic: str) -> set[str]:
        """Return every node that participates in *topic* (has an edge)."""
        nodes: set[str] = set()
        for u, v, data in self._graph.edges(data=True):
            if data.get("topic") == topic:
                nodes.add(u)
                nodes.add(v)
        return nodes

    # ------------------------------------------------------------------
    # Chain resolution
    # ------------------------------------------------------------------

    @staticmethod
    def viscous_decay(depth: int, alpha: float = 0.85) -> float:
        """Return the weight multiplier for a delegation chain of *depth* hops."""
        return alpha ** depth

    def resolve_chain(
        self,
        voter_id: str,
        topic: str,
        visited: Optional[set[str]] = None,
        depth: int = 0,
    ) -> tuple[str, int, float]:
        """Follow the delegation chain starting at *voter_id* for *topic*.

        Returns ``(final_voter_id, depth, weight)``.

        For k=1 (single outgoing edge) this follows the chain linearly.
        For k>1 use :meth:`resolve_paths` which handles branching.

        * If the voter votes directly (no outgoing delegation on *topic*),
          ``depth`` is 0 and ``weight`` is 1.0.
        * If the chain ends at a direct voter after *n* hops, ``weight``
          equals ``viscous_decay(depth)``.
        * If a cycle is detected the vote is lost: ``weight`` is 0.0.
        """
        if visited is None:
            visited = set()

        if voter_id in visited:
            return (voter_id, depth, 0.0)

        visited.add(voter_id)

        # Find outgoing delegation edges for this topic.
        outgoing: list[tuple[str, float]] = []
        if voter_id in self._graph:
            for _, target, data in self._graph.out_edges(voter_id, data=True):
                if data.get("topic") == topic:
                    outgoing.append((target, data.get("fraction", 1.0)))

        if not outgoing:
            return (voter_id, depth, self.viscous_decay(depth))

        # Single delegation (k=1) — fast path
        if len(outgoing) == 1:
            target, fraction = outgoing[0]
            final, d, w = self.resolve_chain(target, topic, visited, depth + 1)
            return (final, d, w * fraction)

        # k>1 — shouldn't reach here via resolve_chain; use resolve_paths
        # Return first edge as fallback (resolve_all uses resolve_paths)
        target, fraction = outgoing[0]
        final, d, w = self.resolve_chain(target, topic, visited, depth + 1)
        return (final, d, w * fraction)

    def resolve_paths(
        self,
        voter_id: str,
        topic: str,
        cycle_nodes: set[str],
        visited: Optional[set[str]] = None,
        depth: int = 0,
    ) -> list[tuple[str, float]]:
        """Resolve all delegation paths from *voter_id*, supporting k-way splits.

        Returns a list of ``(final_voter_id, weight)`` tuples — one per
        terminal path.  Fractional edge weights and viscous decay are applied.
        """
        if visited is None:
            visited = set()

        if voter_id in visited or voter_id in cycle_nodes:
            return []

        visited.add(voter_id)

        outgoing: list[tuple[str, float]] = []
        if voter_id in self._graph:
            for _, target, data in self._graph.out_edges(voter_id, data=True):
                if data.get("topic") == topic:
                    outgoing.append((target, data.get("fraction", 1.0)))

        if not outgoing:
            return [(voter_id, self.viscous_decay(depth))]

        results: list[tuple[str, float]] = []
        for target, fraction in outgoing:
            sub_paths = self.resolve_paths(
                target, topic, cycle_nodes, visited.copy(), depth + 1,
            )
            for final, weight in sub_paths:
                results.append((final, weight * fraction))
        return results

    def resolve_all(
        self, topic: str, weight_cap: Optional[float] = None
    ) -> dict[str, float]:
        """Resolve every delegation chain for *topic*.

        Returns a dict mapping each *final voter* (the person who actually
        votes) to their accumulated effective weight.  Delegators whose
        votes are lost to cycles receive weight 0.

        Supports k-way splits: when a voter delegates to multiple delegates
        with fractional weights, each path is resolved independently.

        If *weight_cap* is set, no single voter's accumulated weight can
        exceed it.
        """
        nodes = self._all_topic_nodes(topic)

        # Identify nodes in cycles — their votes are entirely lost.
        sub = self._topic_subgraph(topic)
        cycle_nodes: set[str] = set()
        for cycle in nx.simple_cycles(sub):
            cycle_nodes.update(cycle)

        weights: dict[str, float] = {}

        for node in nodes:
            if node in cycle_nodes:
                weights[node] = 0.0
                continue

            paths = self.resolve_paths(node, topic, cycle_nodes)

            if not paths:
                weights.setdefault(node, 0.0)
                continue

            for final, weight in paths:
                if final in cycle_nodes:
                    weights.setdefault(node, 0.0)
                    continue
                weights.setdefault(final, 0.0)
                weights[final] += weight
                if node != final:
                    weights.setdefault(node, 0.0)

        if weight_cap is not None:
            for voter in weights:
                if weights[voter] > weight_cap:
                    weights[voter] = weight_cap

        return weights

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def detect_all_cycles(self, topic: str) -> list[list[str]]:
        """Return every simple cycle in the delegation subgraph for *topic*."""
        sub = self._topic_subgraph(topic)
        return list(nx.simple_cycles(sub))

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_gini(self, topic: str) -> float:
        """Compute the Gini coefficient of the vote-weight distribution for *topic*.

        Returns a value in [0, 1].  0 means perfect equality; 1 means
        maximum inequality.  Returns 0.0 for empty distributions.
        """
        weights = self.resolve_all(topic)
        values = sorted(weights.values())
        n = len(values)
        if n == 0:
            return 0.0
        total = sum(values)
        if total == 0:
            return 0.0
        cumulative = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(values):
            cumulative += v
            weighted_sum += (2 * (i + 1) - n - 1) * v
        return weighted_sum / (n * total)

    def get_delegators(self, delegate_id: str, topic: str) -> list[str]:
        """Return the voter IDs that *directly* delegated to *delegate_id* on *topic*."""
        delegators: list[str] = []
        if delegate_id not in self._graph:
            return delegators
        for source, _, data in self._graph.in_edges(delegate_id, data=True):
            if data.get("topic") == topic:
                delegators.append(source)
        return delegators

    def get_chain_length(self, voter_id: str, topic: str) -> int:
        """Return the number of hops in the delegation chain from *voter_id*."""
        _, depth, _ = self.resolve_chain(voter_id, topic)
        return depth

    def get_max_weight(self, topic: str) -> float:
        """Return the maximum effective weight any single voter holds for *topic*."""
        weights = self.resolve_all(topic)
        if not weights:
            return 0.0
        return max(weights.values())

    def get_delegate(self, voter_id: str, topic: str) -> Optional[str]:
        """Return the direct delegate for *voter_id* on *topic*, or None."""
        if voter_id not in self._graph:
            return None
        for _, target, data in self._graph.out_edges(voter_id, data=True):
            if data.get("topic") == topic:
                return target
        return None

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX graph (read-only convenience)."""
        return self._graph
