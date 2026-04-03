"""Distribution and timeline visualizations for the liquid democracy dashboard.

All charts use Plotly with a dark-friendly color palette.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Consistent palette across panels
COLORS = {
    "FPTP": "#636EFA",
    "RCV": "#EF553B",
    "TRS": "#00CC96",
    "Delegation": "#AB63FA",
}

PARTY_COLORS = {
    "strong_D": "#1f77b4",
    "lean_D": "#6baed6",
    "independent": "#999999",
    "lean_R": "#fc8d62",
    "strong_R": "#d62728",
}

# Canonical orderings for consistent graph layout (left-to-right: D → Independent → R)
PARTY_ORDER = ["strong_D", "lean_D", "independent", "lean_R", "strong_R"]
CANDIDATE_ORDER = ["Democrat", "Green", "Independent", "Libertarian", "Republican", "Abstained"]


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values."""
    if not values or len(values) < 2:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    total = sum(arr)
    if total == 0:
        return 0.0
    weighted = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(arr))
    return weighted / (n * total)


def _lorenz_curve(weights: list[float]) -> tuple[list[float], list[float]]:
    """Return (x, y) arrays for a Lorenz curve from raw weights."""
    if not weights:
        return [0.0, 1.0], [0.0, 1.0]
    arr = sorted(weights)
    n = len(arr)
    total = sum(arr)
    if total == 0:
        return [0.0, 1.0], [0.0, 1.0]
    cum = np.cumsum(arr) / total
    x = [0.0] + [(i + 1) / n for i in range(n)]
    y = [0.0] + cum.tolist()
    return x, y


def plot_lorenz_curve(voter_weights: list[float], system_name: str = "Delegation") -> go.Figure:
    """Plot a single Lorenz curve with Gini annotation."""
    x, y = _lorenz_curve(voter_weights)
    gini = _gini(voter_weights)
    color = COLORS.get(system_name, "#AB63FA")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Perfect equality", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", fill="tozeroy",
        line=dict(color=color, width=2),
        name=f"{system_name} (Gini={gini:.3f})",
        fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
    ))
    fig.update_layout(
        title=f"Lorenz Curve - {system_name}",
        xaxis_title="Cumulative share of voters",
        yaxis_title="Cumulative share of vote weight",
        template="plotly_dark", height=400, margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def plot_deviation_from_equality(
    delegation_weights: list[float],
) -> go.Figure:
    """Plot the gap between the delegation Lorenz curve and perfect equality.

    Instead of showing the full Lorenz curve (where FPTP/RCV/TRS all sit on the
    diagonal), this zooms into the interesting part: how much delegation deviates
    from equal-weight voting. The shaded area IS the Gini coefficient.
    """
    x, y = _lorenz_curve(delegation_weights)
    gini = _gini(delegation_weights)

    # Deviation = diagonal - lorenz (positive = inequality)
    deviation = [xi - yi for xi, yi in zip(x, y)]

    fig = go.Figure()

    # Shaded deviation area
    fig.add_trace(go.Scatter(
        x=x, y=deviation, mode="lines", fill="tozeroy",
        line=dict(color=COLORS["Delegation"], width=2),
        fillcolor="rgba(171, 99, 250, 0.25)",
        name=f"Delegation inequality (Gini={gini:.3f})",
    ))

    # Zero line = perfect equality
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Perfect equality (FPTP/RCV/TRS)",
        showlegend=True,
    ))

    # Annotate peak deviation
    if deviation:
        max_dev = max(deviation)
        max_idx = deviation.index(max_dev)
        fig.add_annotation(
            x=x[max_idx], y=max_dev,
            text=f"Peak gap: {max_dev:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="white",
            font=dict(size=11, color="white"),
        )

    fig.update_layout(
        title="Deviation from Equal Voting Power",
        xaxis_title="Cumulative share of voters (poorest to richest in vote weight)",
        yaxis_title="Gap from equality (higher = more concentration)",
        template="plotly_dark", height=350,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def plot_wasted_votes(
    fptp_result: dict,
    rcv_result: dict,
    trs_result: dict,
    deleg_result: dict,
    n_agents: int,
) -> go.Figure:
    """Bar chart comparing wasted votes and representation efficiency across systems.

    Wasted votes = votes that don't contribute to electing the winner.
    - FPTP: all votes for losing candidates + winner votes beyond plurality threshold
    - RCV: only final-round loser votes (redistributed votes are "recovered")
    - TRS: round 1 votes for eliminated candidates (round 2 gets a fresh vote)
    - Delegation: weighted votes for the losing candidate

    Also computes the Gallagher index of disproportionality for each system.
    """
    systems = []
    wasted_pcts = []
    gallagher_indices = []
    winner_margins = []
    colors = []

    labels = [
        ("US System\n(FPTP)", fptp_result, COLORS["FPTP"]),
        ("Ranked Choice\n(RCV)", rcv_result, COLORS["RCV"]),
        ("Two-Round\n(TRS)", trs_result, COLORS["TRS"]),
        ("Liquid\nDelegation", deleg_result, COLORS["Delegation"]),
    ]

    for label, result, color in labels:
        winner = result.get("winner")
        counts = result.get("counts") or result.get("weighted_counts") or result.get("round1_counts") or {}
        total = sum(counts.values()) if counts else 0

        if total == 0 or winner is None:
            systems.append(label)
            wasted_pcts.append(0)
            gallagher_indices.append(0)
            winner_margins.append(0)
            colors.append(color)
            continue

        winner_votes = counts.get(winner, 0)
        loser_votes = total - winner_votes

        # Wasted votes calculation varies by system
        if "RCV" in label:
            # RCV: only votes still with losing candidates in the final round
            rounds = result.get("rounds", [])
            if rounds:
                final_round = rounds[-1]
                final_total = sum(final_round.values())
                final_winner_votes = final_round.get(winner, 0)
                wasted = final_total - final_winner_votes
            else:
                wasted = loser_votes
        elif "Two-Round" in label:
            # TRS: if decided in round 1, same as FPTP;
            # if round 2, only round 2 loser votes are wasted
            r2 = result.get("round2_counts")
            if r2:
                r2_total = sum(r2.values())
                r2_winner = r2.get(winner, 0)
                wasted = r2_total - r2_winner
            else:
                wasted = loser_votes
        else:
            # FPTP and Delegation: all loser votes are wasted
            wasted = loser_votes

        wasted_pct = (wasted / total * 100) if total > 0 else 0
        margin = ((winner_votes - loser_votes) / total * 100) if total > 0 else 0

        # Gallagher index: sqrt(0.5 * sum((vote_share - seat_share)^2))
        # With single-winner, seat_share is 100% for winner, 0% for others
        gallagher_sq_sum = 0
        for cand, v in counts.items():
            vote_share = v / total * 100
            seat_share = 100.0 if cand == winner else 0.0
            gallagher_sq_sum += (vote_share - seat_share) ** 2
        gallagher = (0.5 * gallagher_sq_sum) ** 0.5

        systems.append(label)
        wasted_pcts.append(round(wasted_pct, 1))
        gallagher_indices.append(round(gallagher, 1))
        winner_margins.append(round(margin, 1))
        colors.append(color)

    fig = go.Figure()

    # Wasted votes bars
    fig.add_trace(go.Bar(
        x=systems, y=wasted_pcts,
        marker_color=colors, opacity=0.85,
        text=[f"{w}%" for w in wasted_pcts],
        textposition="outside",
        name="Wasted votes %",
    ))

    # Gallagher index as line overlay
    fig.add_trace(go.Scatter(
        x=systems, y=gallagher_indices,
        mode="lines+markers+text",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=8, color="white"),
        text=[f"G={g}" for g in gallagher_indices],
        textposition="top center",
        textfont=dict(size=10, color="white"),
        name="Gallagher index",
        yaxis="y2",
    ))

    fig.update_layout(
        title="Wasted Votes & Disproportionality",
        yaxis=dict(title="Wasted votes (%)", range=[0, max(wasted_pcts + [50]) * 1.3]),
        yaxis2=dict(
            title="Gallagher index", overlaying="y", side="right",
            range=[0, max(gallagher_indices + [50]) * 1.3],
        ),
        template="plotly_dark", height=380,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=50, r=60, t=50, b=70),
        showlegend=True,
    )
    return fig


def plot_gini_over_time(gini_history: list[dict[str, float]]) -> go.Figure:
    """Line chart of Gini coefficient per race over voting ticks.

    Campaign-phase ticks always read 0 (delegations aren't committed until
    voting), so we trim them and only show the voting phase where Gini
    actually evolves.
    """
    fig = go.Figure()
    if not gini_history:
        fig.update_layout(title="Gini Over Time (no data)", template="plotly_dark", height=350)
        return fig

    race_ids = list(gini_history[0].keys())

    # Find first tick with any non-zero Gini (= voting phase start)
    first_nonzero = next(
        (i for i, snap in enumerate(gini_history)
         if any(v > 1e-9 for v in snap.values())),
        0,
    )
    # Include one zero tick before the transition for context
    start = max(0, first_nonzero - 1)
    trimmed = gini_history[start:]

    ticks = list(range(start, start + len(trimmed)))
    race_colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A", "#FF6692"]

    for i, race_id in enumerate(race_ids):
        values = [snap.get(race_id, 0.0) for snap in trimmed]
        fig.add_trace(go.Scatter(
            x=ticks, y=values, mode="lines+markers",
            name=race_id.capitalize(),
            line=dict(color=race_colors[i % len(race_colors)], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title="Gini Coefficient Over Time (Voting Phase)",
        xaxis_title="Tick", yaxis_title="Gini",
        template="plotly_dark", height=350,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def plot_chain_length_distribution(delegation_graph, race_id: str) -> go.Figure:
    """Histogram of delegation chain lengths for a given race."""
    nodes = delegation_graph._all_topic_nodes(race_id)
    lengths = []
    for node in nodes:
        length = delegation_graph.get_chain_length(node, race_id)
        if length > 0:
            lengths.append(length)

    fig = go.Figure()
    if lengths:
        fig.add_trace(go.Histogram(
            x=lengths, nbinsx=max(lengths) + 1,
            marker_color="#AB63FA", opacity=0.85,
        ))
    fig.update_layout(
        title=f"Chain Length Distribution - {race_id.capitalize()}",
        xaxis_title="Chain length (hops)", yaxis_title="Count",
        template="plotly_dark", height=350,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def plot_weight_distribution(delegation_graph, race_id: str) -> go.Figure:
    """Histogram of delegate weight distribution for a given race."""
    weights = delegation_graph.resolve_all(race_id)
    nonzero = [w for w in weights.values() if w > 0]

    fig = go.Figure()
    if nonzero:
        fig.add_trace(go.Histogram(
            x=nonzero, nbinsx=30,
            marker_color="#00CC96", opacity=0.85,
        ))
    fig.update_layout(
        title=f"Vote Weight Distribution - {race_id.capitalize()}",
        xaxis_title="Effective weight", yaxis_title="Count",
        template="plotly_dark", height=350,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def plot_opinion_timeline(
    opinion_history: dict[int, list],
    agents: dict,
) -> go.Figure:
    """Line chart of opinion *shift* per party over simulation ticks.

    Shows change from initial opinion (tick 0) so that small but real
    movements are visible even when stubbornness keeps absolute values flat.

    opinion_history: {dimension: [list_of_opinions_per_tick]}
    agents: {agent_id: VoterAgent}
    """
    fig = go.Figure()
    snapshots = opinion_history.get(0, [])
    if not snapshots:
        fig.update_layout(title="Opinion Timeline (no data)", template="plotly_dark", height=400)
        return fig

    agent_ids = sorted(agents.keys())
    # Group agent indices by party
    party_indices: dict[str, list[int]] = {}
    for idx, aid in enumerate(agent_ids):
        party = agents[aid].party_id.value
        party_indices.setdefault(party, []).append(idx)

    ticks = list(range(len(snapshots)))

    ordered_parties = [p for p in PARTY_ORDER if p in party_indices]
    # Include any parties not in canonical order (shouldn't happen, but safe)
    ordered_parties += [p for p in sorted(party_indices) if p not in ordered_parties]

    for party in ordered_parties:
        indices = party_indices[party]
        means = []
        for snap in snapshots:
            arr = np.array(snap)
            means.append(float(np.mean(arr[indices])))
        # Delta from initial opinion — highlights the actual shift
        initial = means[0]
        deltas = [m - initial for m in means]
        fig.add_trace(go.Scatter(
            x=ticks, y=deltas, mode="lines",
            name=party.replace("_", " ").title(),
            line=dict(color=PARTY_COLORS.get(party, "#999999"), width=2),
            hovertemplate=(
                "%{fullData.name}<br>"
                "Tick %{x}: %{y:+.4f} shift<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Opinion Shift During Campaign (Dimension 0: Economy)",
        xaxis_title="Tick",
        yaxis_title="Change from initial mean opinion",
        template="plotly_dark", height=400,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Preference shift Sankey (initial expected → final vote)
# ---------------------------------------------------------------------------

_CANDIDATE_IDEOLOGIES: dict[str, float] = {
    "Democrat": -0.6, "Republican": 0.6, "Libertarian": 0.4,
    "Green": -0.8, "Independent": 0.0,
}

_CANDIDATE_COLORS: dict[str, str] = {
    "Democrat": "#3b82f6",
    "Republican": "#ef4444",
    "Libertarian": "#f59e0b",
    "Green": "#22c55e",
    "Independent": "#a855f7",
}


def _closest_candidate(ideology: float, candidates: list[str]) -> str:
    return min(candidates, key=lambda c: abs(ideology - _CANDIDATE_IDEOLOGIES.get(c, 0.0)))


def plot_preference_shift(
    agents: dict,
    initial_ideologies: dict[int, float],
    race_id: str,
    candidates: list[str],
    vote_choices: dict[int, str] | None = None,
) -> go.Figure:
    """Build a Sankey showing initial expected candidate → final vote.

    *vote_choices* overrides agent.race_states[race_id].vote_choice when
    provided (useful for showing different systems' outcomes).
    """
    from collections import Counter

    # Count flows: (initial_expected, final_vote) → count
    flows: Counter[tuple[str, str]] = Counter()

    for aid, agent in agents.items():
        initial_ideo = initial_ideologies.get(aid)
        if initial_ideo is None:
            continue

        if vote_choices is not None:
            final = vote_choices.get(aid)
        else:
            rs = agent.race_states.get(race_id)
            final = rs.vote_choice if rs else None

        if final is None:
            final = "Abstained"

        expected = _closest_candidate(initial_ideo, candidates)
        flows[(expected, final)] += 1

    if not flows:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=350)
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Build node list with consistent ordering (D → Green → Ind → Lib → R → Abstain)
    _raw_initial = {f for f, _ in flows.keys()}
    _raw_final = {t for _, t in flows.keys()}
    all_initial = [c for c in CANDIDATE_ORDER if c in _raw_initial]
    all_initial += [c for c in sorted(_raw_initial) if c not in all_initial]
    all_final = [c for c in CANDIDATE_ORDER if c in _raw_final]
    all_final += [c for c in sorted(_raw_final) if c not in all_final]

    _SHORT = {"Democrat": "Dem", "Republican": "Rep", "Libertarian": "Lib", "Green": "Grn", "Independent": "Ind"}
    left_nodes = [f"Start: {_SHORT.get(c, c)}" for c in all_initial]
    right_nodes = [_SHORT.get(c, c) if c != "Abstained" else "Abstain" for c in all_final]
    nodes = left_nodes + right_nodes

    left_idx = {c: i for i, c in enumerate(all_initial)}
    right_idx = {c: len(left_nodes) + i for i, c in enumerate(all_final)}

    sources, targets, values, link_colors = [], [], [], []
    for (src, tgt), count in flows.items():
        sources.append(left_idx[src])
        targets.append(right_idx[tgt])
        values.append(count)
        # Color links by whether the voter shifted
        if src == tgt:
            link_colors.append("rgba(150,150,150,0.3)")  # stayed
        else:
            link_colors.append("rgba(255,165,0,0.5)")  # shifted

    # Node colors
    node_colors = []
    for c in all_initial:
        node_colors.append(_CANDIDATE_COLORS.get(c, "#6b7280"))
    for c in all_final:
        node_colors.append(_CANDIDATE_COLORS.get(c, "#6b7280"))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=25, thickness=20,
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
        ),
        textfont=dict(size=14, color="white", family="sans-serif"),
    )])
    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=5, r=80, t=10, b=10),
    )
    return fig
