"""Network visualization helpers using Plotly + NetworkX layouts."""

from __future__ import annotations

import networkx as nx
import numpy as np
import plotly.graph_objects as go

PARTY_COLORS = {
    "strong_D": "#1f77b4",
    "lean_D": "#6baed6",
    "independent": "#999999",
    "lean_R": "#fc8d62",
    "strong_R": "#d62728",
}


def plot_delegation_network(
    delegation_graph, agents: dict, race_id: str, max_nodes: int = 500,
) -> go.Figure:
    """Plotly force-graph: nodes colored by party, sized by delegation weight."""
    sub = delegation_graph._topic_subgraph(race_id)

    # Limit to max_nodes for rendering performance
    all_nodes = list(sub.nodes())
    if len(all_nodes) > max_nodes:
        # Keep nodes with highest degree
        degree_sorted = sorted(all_nodes, key=lambda n: sub.degree(n), reverse=True)
        keep = set(degree_sorted[:max_nodes])
        sub = sub.subgraph(keep).copy()

    if len(sub.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"Delegation Network - {race_id.capitalize()} (no delegations)",
            template="plotly_dark", height=500,
        )
        return fig

    # Compute layout
    pos = nx.spring_layout(sub, seed=42, k=1.5 / max(1, len(sub.nodes()) ** 0.5))

    # Resolve weights for node sizing
    weights = delegation_graph.resolve_all(race_id)

    # Build edge traces
    edge_x, edge_y = [], []
    for u, v in sub.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        hoverinfo="none", showlegend=False,
    )

    # Build node traces
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in sub.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Color by party
        agent_id = int(node)
        agent = agents.get(agent_id)
        if agent is not None:
            party = agent.party_id.value
            node_color.append(PARTY_COLORS.get(party, "#999999"))
        else:
            node_color.append("#999999")

        # Size by weight
        w = weights.get(node, 1.0)
        node_size.append(max(4, min(30, 4 + w * 3)))

        # Hover text
        label = f"Agent {node}"
        if agent is not None:
            label += f"<br>Party: {agent.party_id.value}"
        label += f"<br>Weight: {w:.2f}"
        node_text.append(label)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color="#333")),
        text=node_text, hoverinfo="text", showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Delegation Network - {race_id.capitalize()} ({len(sub.nodes())} nodes)",
        template="plotly_dark", height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_delegation_sankey(
    delegation_graph, agents: dict, direct_votes: dict[int, str],
    race_id: str, candidates: list[str],
) -> go.Figure:
    """Sankey diagram: party groups -> delegates -> candidates."""
    weights = delegation_graph.resolve_all(race_id)
    sub = delegation_graph._topic_subgraph(race_id)

    if len(sub.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"Vote Flow - {race_id.capitalize()} (no delegations)",
            template="plotly_dark", height=450,
        )
        return fig

    # Identify terminal voters (those with accumulated weight > 1)
    terminal_voters = {nid: w for nid, w in weights.items() if w > 1.0}

    # Limit to top delegates for readability
    top_delegates = sorted(terminal_voters.items(), key=lambda x: x[1], reverse=True)[:15]
    if not top_delegates:
        fig = go.Figure()
        fig.update_layout(
            title=f"Vote Flow - {race_id.capitalize()} (no significant delegation)",
            template="plotly_dark", height=450,
        )
        return fig

    # Build Sankey nodes: [party_groups..., delegates..., candidates...]
    party_groups = ["Democrat-leaning", "Independent", "Republican-leaning"]
    delegate_labels = [f"Agent {d[0]} (w={d[1]:.1f})" for d in top_delegates]
    all_labels = party_groups + delegate_labels + candidates
    n_party = len(party_groups)
    n_delegates = len(top_delegates)

    # Node colors
    node_colors = (
        ["#1f77b4", "#999999", "#d62728"]
        + ["#AB63FA"] * n_delegates
        + ["#1f77b4" if "Democrat" in c else "#d62728" if "Republican" in c else "#999999"
           for c in candidates]
    )

    sources, targets, values = [], [], []

    # Party group -> delegate links
    for di, (delegate_id, w) in enumerate(top_delegates):
        delegators = delegation_graph.get_delegators(delegate_id, race_id)
        d_count = 0
        r_count = 0
        i_count = 0
        for did in delegators:
            agent = agents.get(int(did))
            if agent is None:
                i_count += 1
                continue
            pv = agent.party_id.value
            if "D" in pv:
                d_count += 1
            elif "R" in pv:
                r_count += 1
            else:
                i_count += 1
        delegate_node = n_party + di
        if d_count > 0:
            sources.append(0)
            targets.append(delegate_node)
            values.append(d_count)
        if i_count > 0:
            sources.append(1)
            targets.append(delegate_node)
            values.append(i_count)
        if r_count > 0:
            sources.append(2)
            targets.append(delegate_node)
            values.append(r_count)

    # Delegate -> candidate links
    for di, (delegate_id, w) in enumerate(top_delegates):
        delegate_node = n_party + di
        vote = direct_votes.get(int(delegate_id))
        if vote is None:
            # Infer from ideology
            agent = agents.get(int(delegate_id))
            if agent is not None:
                vote = candidates[0] if float(agent.ideology[0]) < 0 else candidates[-1]
            else:
                vote = candidates[0]
        if vote in candidates:
            cand_node = n_party + n_delegates + candidates.index(vote)
            sources.append(delegate_node)
            targets.append(cand_node)
            values.append(w)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, label=all_labels,
            color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values),
        textfont=dict(size=13, color="white", family="sans-serif"),
    )])
    fig.update_layout(
        title=dict(text=f"Vote Flow Through Delegates - {race_id.capitalize()}", font=dict(size=15, color="white")),
        template="plotly_dark", height=450,
        margin=dict(l=20, r=80, t=50, b=20),
    )
    return fig
