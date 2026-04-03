"""Streamlit dashboard for the liquid democracy simulation.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from engine.simulation import LiquidDemocracyModel, SimulationConfig
from dashboard.network_viz import plot_delegation_network, plot_delegation_sankey
from dashboard.distribution_panels import (
    plot_chain_length_distribution,
    plot_deviation_from_equality,
    plot_gini_over_time,
    plot_opinion_timeline,
    plot_preference_shift,
    plot_wasted_votes,
    plot_weight_distribution,
)
from dashboard.llm_analysis import analyze_results, generate_graph_summaries

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Liquid Democracy Simulation", layout="wide")

# ---------------------------------------------------------------------------
# Tooltips / help text
# ---------------------------------------------------------------------------

PARAM_HELP = {
    "n_agents": (
        "Population size. More agents = more realistic delegation dynamics "
        "but slower runs. 500-1000 is fast; 5000+ is research-grade."
    ),
    "pvi_lean": (
        "Cook Partisan Voter Index. Negative = Democratic lean (e.g. -15 is D+15), "
        "positive = Republican lean. 0 = perfectly even district."
    ),
    "delegation_prob": (
        "Fraction of agents who consider delegating their vote each campaign tick. "
        "Higher values = more delegation, higher Gini. Real-world range: 3-17%."
    ),
    "viscous_alpha": (
        "How much weight survives each hop in a delegation chain. "
        "At 0.85, a 3-hop chain carries only 61% of the original weight. "
        "1.0 = no decay (pure liquid democracy). 0.0 = delegation has no effect."
    ),
    "weight_cap": (
        "Hard ceiling on how many votes any single delegate can carry. "
        "For example, a cap of 50 means no delegate can cast more than 50 votes, "
        "even if 200 people delegated to them — the excess delegations are simply lost. "
        "'None' = unlimited (realistic but allows extreme concentration). "
        "In a 1000-agent sim without a cap, top delegates routinely accumulate 30-80 votes. "
        "Caps reduce power concentration but weaken liquid democracy's core promise "
        "that the most trusted people should carry the most weight."
    ),
    "delegation_k": (
        "How many delegates each voter can split their vote across. "
        "k=1 is standard delegation — your full vote goes to one person. "
        "k=2 means your vote is split evenly (0.5 each) between two delegates. "
        "The delegate does not choose how to split; every delegator's vote is divided "
        "equally among their k chosen trustees. This dramatically reduces max weight "
        "from O(sqrt(n)) to O(log(n)) — the single strongest concentration mitigation."
    ),
    "pref_gamma": (
        "Controls how strongly existing popularity drives new delegations. "
        "The probability of delegating to someone is proportional to their current "
        "delegation count raised to this exponent. At gamma=1.0, delegation is purely "
        "proportional to popularity. At gamma=1.5, someone with 10 delegations is ~31x "
        "more attractive than someone with 1 (10^1.5). The empirical value from "
        "LiquidFeedback (~1.38) means real users exhibit strong but not extreme "
        "'rich get richer' behavior — popular delegates snowball, but not as fast "
        "as pure squared (gamma=2) would predict."
    ),
    "bandwagon": (
        "Herding effect: how much seeing others delegate to someone increases "
        "your own likelihood of doing the same. At 0.25, a voter is 25% more likely "
        "to delegate to someone they see others choosing, independent of that "
        "delegate's actual expertise. Based on Muchnik et al.'s research showing "
        "a ~25% herding bias in online voting — people follow the crowd even when "
        "they have their own information."
    ),
    "bounded_conf": (
        "Bounded confidence controls whose opinions an agent listens to. "
        "Each agent has an ideology score from -1 (far left) to +1 (far right). "
        "An agent will only be influenced by peers within this distance of their "
        "own position.\n\n"
        "At **epsilon = 0.2**, a moderate at 0.0 only listens to people between "
        "-0.2 and +0.2 — they never hear from anyone with strong convictions. "
        "A strong partisan at +0.7 only hears from +0.5 to +0.9 — a tight echo "
        "chamber. The electorate fragments into isolated ideological clusters that "
        "drift apart over time.\n\n"
        "At **epsilon = 0.4**, the listening window doubles. That same moderate now "
        "hears from -0.4 to +0.4 (mild partisans included), and the strong partisan "
        "hears from +0.3 to +1.0 (a broader slice of their side). There's enough "
        "overlap between groups to allow some cross-partisan influence.\n\n"
        "At **epsilon = 1.0 or None**, everyone listens to everyone — maximum exposure, "
        "opinions tend to moderate toward the center."
    ),
}

GRAPH_HELP = {
    "outcomes": (
        "Each column shows the winner under a different voting system, all run "
        "on the same population. When systems disagree, the column header turns "
        "red — this is where electoral system choice changes who wins."
    ),
    "network": (
        "Each dot is a voter. Blue = Democrat, Red = Republican, Gray = Independent. "
        "Bigger dots have more delegated voting power. Lines show delegation edges. "
        "Clusters reveal partisan delegation bubbles."
    ),
    "lorenz": (
        "The Lorenz curve shows how equally voting power is distributed. "
        "Voters are sorted from least to most powerful along the x-axis "
        "(so 0.4 = the bottom 40% by voting weight). The y-axis shows what fraction "
        "of total voting power that group holds. On the diagonal, the bottom 40% "
        "holds exactly 40% of power — perfect equality. In FPTP/RCV/TRS everyone "
        "has one vote, so they sit on the diagonal. In liquid delegation, people who "
        "delegated away their vote have weight 0, while delegates accumulate extra "
        "weight — so the curve bows below. The Gini coefficient (0-1) measures the "
        "area between the curve and the diagonal: 0 = equal, 1 = one person holds all votes."
    ),
    "sankey": (
        "Traces the flow of delegated votes from party groups (left) through "
        "top delegates (middle) to candidates (right). Wider bands = more votes "
        "flowing through that path. Shows who the power brokers are.\n\n"
        "**Cross-party flow** (a D-leaning voter delegating to an R-leaning delegate, "
        "or vice versa) is rare under default settings because 65% of social network "
        "ties are same-party (political homophily). To see cross-party delegation, try: "
        "a competitive district (PVI near 0), higher delegation probability (40%+), "
        "or the Celebrity scenario (fame overrides party alignment)."
    ),
    "chain_dist": (
        "How many hops delegation chains are. Longer chains mean more trust "
        "placed transitively — your delegate's delegate's delegate votes for you. "
        "Each hop decays the weight by the viscous decay factor."
    ),
    "weight_dist": (
        "How delegation weight is distributed among delegates. "
        "A few delegates with very high weight = concentrated power. "
        "Most delegates near weight 1.0 = healthy distribution."
    ),
    "opinion": (
        "Tracks the average ideology of each party group over the campaign. "
        "Convergence = opinion moderation through social influence. "
        "Divergence = echo chambers. The Friedkin-Johnsen model balances "
        "stubbornness (anchoring to initial beliefs) with peer influence."
    ),
    "gini_time": (
        "How the Gini coefficient (power concentration) evolves during the "
        "campaign. Rising Gini = delegation accumulating over time. "
        "Flat = delegation in equilibrium."
    ),
}

SCENARIO_INFO = {
    "Baseline": {
        "desc": "Organic delegation with default parameters. No external shocks.",
        "watch_for": "Natural Gini equilibrium — typically 0.4-0.7.",
    },
    "Celebrity": {
        "desc": "A famous person (think podcaster or influencer — not a candidate) joins the "
                "delegation network mid-campaign. They don't run for office; they vote like "
                "anyone else, but their fame causes a huge number of people to delegate to them. "
                "One person ends up casting dozens or hundreds of votes.",
        "watch_for": "Gini spike after entry. Max weight can jump to 20+.",
    },
    "Hub Attack": {
        "desc": "An adversary compromises the top delegation hubs and shifts their ideology.",
        "watch_for": "Delegation system outcome may flip vs FPTP. Demonstrates systemic risk.",
    },
    "Stale Decay": {
        "desc": "Delegations persist without review. Delegates gradually drift ideologically.",
        "watch_for": "Gini rising over time as stale delegations accumulate.",
    },
    "k=2": {
        "desc": "Each voter delegates to 2 people (vote split). The strongest concentration fix.",
        "watch_for": "Gini and max weight should both drop significantly vs baseline.",
    },
}

# ---------------------------------------------------------------------------
# Quick Start dialog
# ---------------------------------------------------------------------------

QUICK_START_MD = """
### Why This Exists

Most democracies lock citizens into a single voting mechanism chosen centuries ago.
But what if a different system — ranked choice, runoffs, or letting people delegate
their vote to someone they trust — could produce **more representative outcomes**?

This simulator lets you stress-test that question. It runs the **same population**
through four electoral systems side by side, so you can see exactly where they
agree, where they diverge, and which designs concentrate power versus distributing it.

---

### How This Simulator Works

This tool models a congressional election where the same population of synthetic voters
is run through **four different voting systems** simultaneously:

| System | How it works |
|--------|-------------|
| **FPTP** | Current US system. One vote, highest count wins. |
| **RCV** | Rank candidates. Lowest eliminated until someone has a majority. |
| **TRS** | French-style two rounds. Top candidates advance to a runoff. |
| **Delegation** | Vote directly OR delegate to someone you trust. Delegations chain transitively. |

The core question: **does letting people delegate produce better outcomes, or does it
concentrate too much power in too few hands?**

---

### How the Simulation Runs

The simulation runs in **91 ticks** across three phases:

**Phase 1 — Campaign (ticks 1-80):** Agents form opinions and build trust.
Each tick, agents update their ideology based on conversations with neighbors
(the Friedkin-Johnsen model — stubborn agents resist, open-minded ones shift).
Trust accumulates between agents who agree, and erodes when delegates betray
their delegators' positions. Agents discover potential delegates through their
trust network. Popular delegates attract more delegators (preferential attachment).

**Phase 2 — Voting (ticks 81-90):** Each agent decides: vote directly (if they
know enough about the race), delegate to a trusted neighbor (if they don't),
or abstain. Decisions are staggered — high-engagement agents act first, late
deciders fill in over 10 ticks. Delegations are registered in the delegation
graph, forming chains (A delegates to B, B delegates to C → C casts 3 votes).

**Phase 3 — Tally (tick 91):** The same population's decisions are counted
under all four voting systems simultaneously. FPTP just counts. RCV runs
elimination rounds. TRS checks for a majority, then runs a runoff if needed.
Delegation resolves all chains, applies viscous decay, and tallies weighted votes.

**Agent decisions:** 95% of agents use rule-based logic (ideology distance to
candidates, knowledge level, trust scores). The remaining 5% can optionally
use Claude to make decisions based on a persona prompt — this is off by default
for speed but can be enabled with the LLM toggle.

**Convergence:** Opinion dynamics typically stabilize by tick 40-60. Stubborn
partisans barely move; independents moderate toward their local network average.
Trust networks solidify by tick 30, and most delegations form between ticks 40-80.
The system reaches approximate equilibrium before voting begins.

---

### Interesting Things to Try

**1. Crank up delegation probability to 30-40%** and watch the Gini coefficient rise.
When a third of the population delegates, a handful of "super-delegates" end up
controlling hundreds of votes.

**2. Try the Celebrity scenario** — it injects a famous figure mid-campaign who
attracts delegations through fame rather than expertise. Watch one node on the
network graph balloon to 20x normal size.

**3. Compare k=1 vs k=2** — allowing voters to split their delegation across two
trustees is the single most effective fix for power concentration. The max vote
weight drops from O(sqrt(n)) to O(log(n)).

**4. Set a strong D or R lean** (PVI -20 or +20) then see which voting systems
still produce different winners. Even in lopsided districts, RCV and FPTP can diverge.

**5. Play with viscous decay** — at 1.0, every hop in a delegation chain carries
full weight. At 0.5, a 3-hop chain carries only 12.5% of the original vote.
This single parameter controls how "liquid" the democracy actually is.

---

### Key Behavior to Watch

- **Gini coefficient**: this measures power concentration in the delegation network, how unevenly voting weight is distributed among delegates. 0 = perfectly equal weight distribution (everyone carries one vote). 1 = one delegate holds all the power. The dashboard plots it alongside Lorenz curves to compare how FPTP, RCV, TRS, and liquid delegation concentrate or distribute influence. Real liquid democracy deployments (LiquidFeedback, Google Votes) consistently produce Gini 0.64-0.99. Named after Italian statistician Corrado Gini (1912))
- **Max delegate weight**: the single highest number of votes any one person carries.
- **System divergence**: when different voting systems produce different winners on the same population — this is where electoral system design matters most.
"""

# Show quick start on first visit
if "seen_quick_start" not in st.session_state:
    st.session_state["seen_quick_start"] = False


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "Baseline": {},
    "Celebrity": {
        "preferential_attachment_gamma": 3.0,
        "delegation_probability_base": 0.25,
    },
    "Hub Attack": {
        "preferential_attachment_gamma": 2.5,
        "delegation_probability_base": 0.30,
        "weight_cap": 50.0,
    },
    "Stale Decay": {
        "viscous_decay_alpha": 0.50,
        "delegation_probability_base": 0.20,
    },
    "k=2": {
        "delegation_options_k": 2,
    },
}


# ---------------------------------------------------------------------------
# Campaign trail summary (news-style)
# ---------------------------------------------------------------------------


def build_campaign_summary(model: "LiquidDemocracyModel") -> str:
    """Generate a news-style summary of what happened during the campaign."""
    import numpy as np

    n = len(model.agents)
    cfg = model.config

    # --- Opinion shifts ---
    initial = model._opinion_snapshots[0]
    final_campaign = model._opinion_snapshots[min(cfg.campaign_ticks, len(model._opinion_snapshots) - 1)]
    agent_ids = sorted(model.agents.keys())

    party_groups: dict[str, list[int]] = {}
    for idx, aid in enumerate(agent_ids):
        party = model._initial_parties.get(aid, "independent")
        party_groups.setdefault(party, []).append(idx)

    biggest_shift_party = ""
    biggest_shift_val = 0.0
    group_summaries = []
    for party, indices in sorted(party_groups.items()):
        init_mean = float(np.mean(initial[indices]))
        final_mean = float(np.mean(final_campaign[indices]))
        shift = final_mean - init_mean
        direction = "right" if shift > 0 else "left"
        label = party.replace("_", " ").title()
        if abs(shift) > abs(biggest_shift_val):
            biggest_shift_val = shift
            biggest_shift_party = label
        if abs(shift) > 0.001:
            group_summaries.append(f"**{label}** voters drifted {direction} by {abs(shift):.3f}")

    # --- Convergence ---
    if model._convergence_tick is not None:
        conv_line = f"Public opinion stabilized early, converging by tick {model._convergence_tick} of {cfg.campaign_ticks}."
    else:
        conv_line = f"Opinions remained in flux through all {cfg.campaign_ticks} campaign ticks — no convergence reached."

    # --- Cross-party movement ---
    cross_party = 0
    for aid in model.agents:
        init_ideo = model._initial_ideologies.get(aid, 0.0)
        curr_ideo = float(model.agents[aid].ideology[0])
        init_lean = "D" if init_ideo < -0.15 else ("R" if init_ideo > 0.15 else "I")
        curr_lean = "D" if curr_ideo < -0.15 else ("R" if curr_ideo > 0.15 else "I")
        if init_lean != curr_lean:
            cross_party += 1
    cross_pct = cross_party / n * 100 if n else 0

    # --- Delegation intentions ---
    n_intending = sum(
        1 for a in model.agents.values()
        if any(s.delegation_targets for s in a.race_states.values())
    )
    deleg_pct = n_intending / n * 100 if n else 0

    # --- Trust ---
    trust_vals = [t for a in model.agents.values() for t in a.trust_scores.values()]
    avg_trust = float(np.mean(trust_vals)) if trust_vals else 0.0

    # --- Build the story ---
    lines = []
    lines.append(
        f"**{n:,} voters** completed an {cfg.campaign_ticks}-tick campaign season. "
        f"Here's what happened on the trail."
    )
    lines.append("")

    # Lede: biggest shift
    if biggest_shift_party and abs(biggest_shift_val) > 0.001:
        direction = "rightward" if biggest_shift_val > 0 else "leftward"
        lines.append(
            f"**The big story:** {biggest_shift_party} voters saw the largest ideological "
            f"movement, shifting {direction} by {abs(biggest_shift_val):.3f} points on "
            f"the -1 (D) to +1 (R) scale."
        )
    else:
        lines.append(
            "**The big story:** No party group moved significantly — "
            "voters largely held their ground throughout the campaign."
        )

    # Convergence
    lines.append("")
    lines.append(f"**Consensus watch:** {conv_line}")

    # Group-by-group shifts
    if group_summaries:
        lines.append("")
        lines.append("**By the numbers:** " + " | ".join(group_summaries) + ".")

    # Cross-party
    lines.append("")
    if cross_party > 0:
        lines.append(
            f"**Crossing the aisle:** {cross_party:,} voters ({cross_pct:.1f}%) "
            f"shifted far enough to change their partisan lean entirely."
        )
    else:
        lines.append(
            "**Crossing the aisle:** Not a single voter shifted enough "
            "to change their partisan lean — a deeply entrenched electorate."
        )

    # Delegation
    lines.append("")
    lines.append(
        f"**Delegation forecast:** {n_intending:,} voters ({deleg_pct:.0f}%) "
        f"identified a trusted delegate and plan to hand off their vote. "
        f"Average trust across all relationships: {avg_trust:.2f}."
    )

    # --- Influence attribution ---
    if not hasattr(model, "get_influence_summary"):
        return "\n".join(lines)  # stale model from previous session
    influence = model.get_influence_summary()
    if influence["peer_total"] > 0 or influence["media_total"] > 0:
        lines.append("")
        if influence["media_total"] > 0:
            lines.append(
                f"**Who moved the needle?** Peer influence (neighbors, trusted contacts) "
                f"accounted for {influence['peer_pct']:.0f}% of opinion shifts, "
                f"while media coverage drove {influence['media_pct']:.0f}%. "
                f"Cumulative shift: {influence['peer_total']:.4f} (peers) vs "
                f"{influence['media_total']:.4f} (media)."
            )
        else:
            lines.append(
                f"**Who moved the needle?** All opinion movement came from peer influence "
                f"(neighbors and trusted contacts). Media was disabled or had no effect."
            )

    # --- LLM vs rule-based voter comparison ---
    llm_stats = model.get_llm_vs_rule_stats()
    if llm_stats.get("llm_enabled") and llm_stats.get("n_llm", 0) > 0:
        n_llm = llm_stats["n_llm"]
        n_rule = llm_stats["n_rule"]
        race_id = next(iter(model.config.races))
        llm_race = llm_stats.get(race_id, {}).get("llm", {})
        rule_race = llm_stats.get(race_id, {}).get("rule", {})

        llm_vote_pct = llm_race.get("actions", {}).get("vote", 0)
        rule_vote_pct = rule_race.get("actions", {}).get("vote", 0)
        llm_deleg_pct = llm_race.get("actions", {}).get("delegate", 0)
        rule_deleg_pct = rule_race.get("actions", {}).get("delegate", 0)

        lines.append("")
        lines.append(
            f"**Free thinkers vs rule followers:** {n_llm} LLM-powered voters (Claude) "
            f"voted directly {llm_vote_pct:.0f}% of the time vs {rule_vote_pct:.0f}% "
            f"for {n_rule} rule-based voters. Delegation rates: "
            f"{llm_deleg_pct:.0f}% (LLM) vs {rule_deleg_pct:.0f}% (rule-based). "
            f"LLM agents weigh their persona, knowledge, and trust network before deciding; "
            f"rule-based agents follow fixed thresholds (knowledge > 0.6 = vote, "
            f"trusted neighbor available = delegate, else abstain/party-line)."
        )

        # Top decision reasons
        llm_reasons = llm_race.get("top_reasons", [])
        rule_reasons = rule_race.get("top_reasons", [])
        if llm_reasons or rule_reasons:
            lines.append("")
            parts = []
            if llm_reasons:
                top = llm_reasons[0][0]
                parts.append(f"LLM agents most commonly cited: \"{top}\"")
            if rule_reasons:
                top = rule_reasons[0][0]
                parts.append(f"rule-based agents followed: \"{top}\"")
            lines.append("**Why they decided:** " + "; ".join(parts) + ".")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cached simulation runner
# ---------------------------------------------------------------------------

def run_simulation_with_progress(config_dict: dict) -> dict:
    """Run simulation tick-by-tick with a Streamlit progress bar."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(SimulationConfig)}
    sim_kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}
    cfg = SimulationConfig(**sim_kwargs)

    n = cfg.n_agents
    progress = st.progress(0, text=f"Seeding {n:,} voters...")
    model = LiquidDemocracyModel(cfg)

    total_steps = cfg.total_ticks + 1  # +1 for post-processing
    campaign_end = cfg.campaign_ticks
    voting_end = campaign_end + cfg.voting_ticks

    _current_tick = [0]

    def _live_progress(msg: str) -> None:
        """Callback from the simulation engine."""
        pct = min(_current_tick[0] / total_steps, 0.99)
        progress.progress(pct, text=msg)

    model.progress_callback = _live_progress

    for tick in range(1, cfg.total_ticks + 1):
        _current_tick[0] = tick
        model.step()
        pct = tick / total_steps

        if tick <= campaign_end:
            progress.progress(pct, text=f"Campaign — forming opinions and trust ({tick}/{campaign_end})")
        elif tick <= voting_end:
            n_voted = sum(len(v) for v in model._direct_votes.values())
            progress.progress(pct, text=f"Voting — {n_voted:,} votes cast")
        else:
            progress.progress(pct, text="Tallying results across all four systems...")

    progress.progress(0.99, text="Finalizing results...")
    results = model.get_results()
    st.session_state["_model"] = model
    progress.empty()

    return {
        "fptp_results": results.fptp_results,
        "rcv_results": results.rcv_results,
        "trs_results": results.trs_results,
        "delegation_results": results.delegation_results,
        "delegation_stats": results.delegation_stats,
        "opinion_history": results.opinion_history,
    }


def _get_winner(result: dict) -> str | None:
    if "winner" in result:
        return result["winner"]
    counts = result.get("counts", result.get("weighted_counts", {}))
    if not counts:
        return None
    return max(counts, key=counts.get)


def _get_counts(result: dict) -> dict:
    """Return the vote counts that decided the winner.

    Each voting system stores counts under different keys:
    - FPTP: "counts"
    - RCV:  "rounds" (list of dicts, final round decided the winner)
    - TRS:  "round1_counts" or "round2_counts" depending on decided_in
    - Delegation: "weighted_counts"
    """
    # TRS round 2 takes priority when present — it decided the winner
    if result.get("round2_counts"):
        return result["round2_counts"]
    # RCV: final round counts decided the winner
    rounds = result.get("rounds")
    if rounds:
        return rounds[-1]
    for key in ("counts", "weighted_counts", "round1_counts"):
        if key in result:
            return result[key]
    return {}


def _collect_graph_data(
    results: dict, config_dict: dict, model, selected_race: str,
) -> dict[str, str]:
    """Collect compact data summaries for each chart to send to Claude."""
    import numpy as np

    stats = results["delegation_stats"]
    fptp_r = results["fptp_results"].get(selected_race, {})
    rcv_r = results["rcv_results"].get(selected_race, {})
    trs_r = results["trs_results"].get(selected_race, {})
    deleg_r = results["delegation_results"].get(selected_race, {})

    gini = stats.get("gini_per_race", {}).get(selected_race, 0)
    max_w = stats.get("max_weight_per_race", {}).get(selected_race, 0)
    n_agents = config_dict.get("n_agents", 0)

    data: dict[str, str] = {}

    # Election outcomes (pie charts)
    winners = {
        "FPTP": fptp_r.get("winner"),
        "RCV": rcv_r.get("winner"),
        "TRS": trs_r.get("winner"),
        "Delegation": deleg_r.get("winner"),
    }
    unique = set(w for w in winners.values() if w)
    wc = deleg_r.get("weighted_counts", {})
    wc_round = {k: round(v, 1) for k, v in wc.items()} if wc else {}
    data["election_outcomes"] = (
        f"Pie charts showing vote share per candidate across 4 electoral systems. "
        f"Winners: {winners}. FPTP counts: {fptp_r.get('counts', {})}. "
        f"Delegation weighted counts: {wc_round}. "
        f"{'All 4 systems agree on winner.' if len(unique) == 1 else 'Systems DISAGREE on winner.'}"
    )

    # Network
    if model:
        sub = model.delegation_graph._topic_subgraph(selected_race)
        data["network"] = (
            f"Force-directed network of delegation relationships. "
            f"{sub.number_of_nodes()} participating nodes, "
            f"{sub.number_of_edges()} delegation edges. "
            f"Node size = delegation weight, color = party."
        )

    # Deviation from equality
    data["deviation"] = (
        f"Area chart showing gap between delegation weights and equal power (weight=1). "
        f"Gini={gini:.4f}, max delegate weight={max_w:.2f}, {n_agents} agents. "
        f"Purple area represents inequality."
    )

    # Wasted votes
    data["wasted_votes"] = (
        f"Bar chart of wasted votes % + Gallagher disproportionality index per system. "
        f"FPTP counts: {fptp_r.get('counts', {})} (total={fptp_r.get('total_votes', '?')}). "
        f"Winners: FPTP={fptp_r.get('winner')}, RCV={rcv_r.get('winner')}, "
        f"TRS={trs_r.get('winner')}, Delegation={deleg_r.get('winner')}."
    )

    # Sankey
    if model:
        weights = model.delegation_graph.resolve_all(
            selected_race, weight_cap=config_dict.get("weight_cap"),
        )
        top = sorted(weights.items(), key=lambda x: -x[1])[:5] if weights else []
        top_str = ", ".join(f"Agent {k}={v:.1f}" for k, v in top)
        data["sankey"] = (
            f"Sankey diagram: party groups -> top delegates -> candidates. "
            f"Top delegates by weight: {top_str}. "
            f"Total delegators: {stats.get('total_delegators', 0)}."
        )

    # Chain lengths
    chains = stats.get("chain_count_per_race", {}).get(selected_race, 0)
    avg_len = stats.get("avg_chain_length_per_race", {}).get(selected_race, 0)
    data["chain_lengths"] = (
        f"Histogram of delegation chain lengths (hops). "
        f"{chains} chains, avg length={avg_len:.2f} hops."
    )

    # Weight distribution
    data["weight_dist"] = (
        f"Histogram of effective vote weights among delegates. "
        f"Max weight={max_w:.2f}, Gini={gini:.4f}."
    )

    # Opinion timeline
    opinion_hist = results.get("opinion_history", {}).get(0, [])
    if opinion_hist and len(opinion_hist) >= 2:
        initial_mean = float(np.mean(opinion_hist[0]))
        final_mean = float(np.mean(opinion_hist[-1]))
        initial_std = float(np.std(opinion_hist[0]))
        final_std = float(np.std(opinion_hist[-1]))
        data["opinion_timeline"] = (
            f"Line chart of mean opinion per party over campaign ticks "
            f"(dim 0: economy, -1=D, +1=R). "
            f"Overall mean: initial={initial_mean:.3f}, final={final_mean:.3f}. "
            f"Std dev: initial={initial_std:.3f}, final={final_std:.3f}. "
            f"{len(opinion_hist)} snapshots."
        )

    # Gini over time
    gini_hist = stats.get("gini_history", [])
    if gini_hist:
        first = gini_hist[0]
        last = gini_hist[-1]
        data["gini_time"] = (
            f"Line chart of Gini coefficient over {len(gini_hist)} ticks. "
            f"First tick values: {first}. Last tick values: {last}."
        )

    # Convergence
    if model and hasattr(model, "get_dynamics_report"):
        dynamics = model.get_dynamics_report()
        conv = (
            f"converged at tick {dynamics['convergence_tick']}"
            if dynamics["converged"]
            else "did not converge"
        )
        data["convergence"] = (
            f"Line chart of mean absolute opinion change per tick. "
            f"Simulation {conv}. "
            f"Cross-party shifts: {dynamics['total_cross_party_shifts']} "
            f"({dynamics['cross_party_pct']}%). "
            f"Stance changes: {dynamics['total_stance_changes']}."
        )

    return data


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Liquid Democracy Simulation")

# Quick Start — uses native expander so closing doesn't trigger rerun
if st.session_state.get("seen_quick_start", False) and not st.session_state.get("_running"):
    with st.expander("Quick Start Guide", expanded=True):
        st.markdown(QUICK_START_MD)

# ---------------------------------------------------------------------------
# Sidebar: Quick Start button + parameters with help tooltips
# ---------------------------------------------------------------------------

_qs_label = "Close Quick Start Guide" if st.session_state.get("seen_quick_start") else "Quick Start Guide"
if st.sidebar.button(_qs_label, type="secondary", width="stretch"):
    st.session_state["seen_quick_start"] = not st.session_state.get("seen_quick_start", False)
    st.rerun()

st.sidebar.title("Parameters")

scenario = st.sidebar.radio(
    "Scenario preset",
    list(SCENARIOS.keys()),
    horizontal=True,
    help="Pre-configured parameter sets that highlight different dynamics. "
         "You can still adjust individual parameters after selecting a preset.",
)
preset = SCENARIOS[scenario]

# Show scenario description
info = SCENARIO_INFO.get(scenario, {})
if info:
    st.sidebar.caption(f"_{info['desc']}_")
    st.sidebar.caption(f"Watch for: **{info['watch_for']}**")

st.sidebar.divider()
st.sidebar.subheader("Population")

data_source = st.sidebar.radio(
    "Data source",
    ["Synthetic", "2024 District Profiles", "2024 CES Survey"],
    horizontal=True,
    help="**Synthetic**: generates a fictional electorate using a partisan lean slider. "
         "Fast and flexible, but not based on real people.\n\n"
         "**2024 District Profiles**: seeds agents from real data for 30 actual US congressional "
         "districts. Uses **Cook PVI** (Partisan Voter Index) — a score that measures how much "
         "a district leans Democrat or Republican compared to the national average. For example, "
         "D+15 means the district voted 15 points more Democratic than the country overall. "
         "Demographics come from the US Census (age, race, income, education, urbanization).\n\n"
         "**2024 CES Survey**: uses real individual-level survey responses from the "
         "**Cooperative Election Study** (CES), a 60,000-person academic survey run by Harvard "
         "that asks Americans about their political views, party identity, ideology, and "
         "demographics. Each congressional district has ~100 real respondents. Agents are "
         "cloned from these real people (resampled to your target population size). "
         "Downloads ~184 MB from Harvard Dataverse on first use (public, no login required).",
)

selected_district = None
use_ces = False

if data_source == "2024 District Profiles":
    from data.districts import DISTRICTS, get_district

    district_options = sorted(
        DISTRICTS.keys(),
        key=lambda d: (DISTRICTS[d].state, DISTRICTS[d].district_num),
    )
    district_labels = {
        did: (
            f"{DISTRICTS[did].state} — {did} "
            f"({'D' if DISTRICTS[did].pvi < 0 else 'R'}+"
            f"{abs(DISTRICTS[did].pvi):.0f})"
        )
        for did in district_options
    }
    selected_district = st.sidebar.selectbox(
        "District",
        district_options,
        format_func=lambda d: district_labels[d],
        help="Pick a real 2024 congressional district or Senate race. "
             "Type to search by state or district ID.",
    )
    dp = get_district(selected_district)
    if dp:
        pvi_lean = dp.pvi
        st.sidebar.caption(
            f"**{dp.state}** | PVI {'D' if pvi_lean < 0 else 'R'}+{abs(pvi_lean):.0f} | "
            f"2024: {dp.actual_winner_party} +{dp.actual_margin:.1f}"
        )
        st.sidebar.caption(
            f"{dp.pct_white:.0%} white, {dp.pct_black:.0%} Black, "
            f"{dp.pct_hispanic:.0%} Hispanic | "
            f"{dp.pct_college:.0%} college | {dp.pct_urban:.0%} urban"
        )
    else:
        pvi_lean = 0.0

elif data_source == "2024 CES Survey":
    from data.ces_loader import is_ces_available

    use_ces = True
    ces_available = is_ces_available()

    if not ces_available:
        st.sidebar.warning(
            "CES data not yet downloaded. Click **RUN** to auto-download "
            "~184 MB from Harvard Dataverse (one-time)."
        )

    # Try to load district list from cache; fall back to text input
    ces_districts: list[str] = []
    if ces_available:
        try:
            from data.ces_loader import list_available_districts
            ces_districts = list_available_districts()
        except Exception:
            ces_districts = []

    if ces_districts:
        def _ces_sort_key(d: str) -> tuple[str, int]:
            parts = d.split("-", 1)
            try:
                return (parts[0], int(parts[1]))
            except (IndexError, ValueError):
                return (parts[0], 0)

        ces_sorted = sorted(ces_districts, key=_ces_sort_key)
        default_idx = ces_sorted.index("PA-07") if "PA-07" in ces_sorted else 0
        selected_district = st.sidebar.selectbox(
            "District",
            ces_sorted,
            index=default_idx,
            help="Pick a congressional district. ~100 real CES respondents per district.",
        )
    else:
        selected_district = st.sidebar.text_input(
            "District ID",
            value="PA-07",
            help="Enter a congressional district (e.g., CA-27, NY-14, TX-13).",
        )
        selected_district = selected_district.strip().upper() if selected_district else "PA-07"

    pvi_lean = 0.0
    st.sidebar.caption(
        "Source: [CES 2024](https://dataverse.harvard.edu/dataset.xhtml?"
        "persistentId=doi:10.7910/DVN/X11EP6) — Harvard Dataverse"
    )

else:  # Synthetic
    pvi_lean = st.sidebar.slider(
        "PVI lean (neg=D, pos=R)", -30.0, 30.0,
        float(preset.get("pvi_lean", 0.0)), step=1.0,
        help=PARAM_HELP["pvi_lean"],
    )

n_agents = st.sidebar.slider(
    "Number of voters", 100, 10000,
    preset.get("n_agents", 200), step=100,
    help=PARAM_HELP["n_agents"],
)

st.sidebar.divider()
st.sidebar.subheader("Candidates")

_ALL_CANDIDATES = ["Democrat", "Republican", "Libertarian", "Green", "Independent"]
_CANDIDATE_IDEO_LABELS = {
    "Democrat": "center-left (-0.6)",
    "Republican": "center-right (+0.6)",
    "Libertarian": "right-libertarian (+0.4)",
    "Green": "far-left (-0.8)",
    "Independent": "centrist (0.0)",
}

candidates = st.sidebar.multiselect(
    "Candidates in race",
    _ALL_CANDIDATES,
    default=["Democrat", "Republican"],
    help=(
        "Choose which candidates appear on the ballot. Each has a fixed ideology "
        "position that voters compare against their own views:\n\n"
        + "\n".join(f"- **{c}**: {_CANDIDATE_IDEO_LABELS[c]}" for c in _ALL_CANDIDATES)
        + "\n\nMore candidates make RCV and TRS more interesting — spoiler effects, "
        "vote splitting, and elimination rounds all become visible. "
        "The delegation system is less affected since delegates vote for "
        "whichever candidate is closest to their accumulated ideology."
    ),
)
if len(candidates) < 2:
    st.sidebar.warning("At least 2 candidates are required.")
    candidates = ["Democrat", "Republican"]

st.sidebar.divider()
st.sidebar.subheader("Delegation")

delegation_prob = st.sidebar.slider(
    "Delegation probability", 0.0, 1.0,
    float(preset.get("delegation_probability_base", 0.15)), step=0.01,
    help=PARAM_HELP["delegation_prob"],
)
viscous_alpha = st.sidebar.slider(
    "Viscous decay (alpha)", 0.0, 1.0,
    float(preset.get("viscous_decay_alpha", 0.85)), step=0.05,
    help=PARAM_HELP["viscous_alpha"],
)
weight_cap_option = st.sidebar.selectbox(
    "Weight cap",
    ["None", "10", "25", "50", "100", "500"],
    index=["None", "10", "25", "50", "100", "500"].index(
        str(int(preset["weight_cap"])) if "weight_cap" in preset and preset["weight_cap"] is not None else "50"
    ),
    help=PARAM_HELP["weight_cap"],
)
weight_cap = None if weight_cap_option == "None" else float(weight_cap_option)

delegation_k = st.sidebar.slider(
    "Delegation options (k)", 1, 5,
    int(preset.get("delegation_options_k", 1)),
    help=PARAM_HELP["delegation_k"],
)

st.sidebar.divider()
st.sidebar.subheader("Influence Dynamics")

pref_attach_gamma = st.sidebar.slider(
    "Preferential attachment (gamma)", 0.0, 3.0,
    float(preset.get("preferential_attachment_gamma", 1.3)), step=0.1,
    help=PARAM_HELP["pref_gamma"],
)
bandwagon = st.sidebar.slider(
    "Bandwagon coefficient", 0.0, 0.5,
    float(preset.get("bandwagon_coefficient", 0.20)), step=0.05,
    help=PARAM_HELP["bandwagon"],
)
homophily = st.sidebar.slider(
    "Political homophily", 0.0, 1.0,
    float(preset.get("homophily", 0.30)), step=0.05,
    help=(
        "Fraction of social network ties rewired to same-party connections. "
        "At 0.30 (default), most ties are mixed — cross-party delegation is common. "
        "At 0.0, ties are fully random across parties. "
        "At 0.65+, most neighbors share your party — like real partisan bubbles. "
        "At 1.0, you only know people in your own party — zero cross-pollination."
    ),
)
bc_option = st.sidebar.selectbox(
    "Bounded confidence (epsilon)",
    ["None", "0.2", "0.4", "0.6", "0.8", "1.0"],
    index=2,  # default to 0.4
    help=PARAM_HELP["bounded_conf"],
)
bounded_confidence = None if bc_option == "None" else float(bc_option)

st.sidebar.divider()
st.sidebar.subheader("Agent Decision Mode")

use_llm = st.sidebar.toggle(
    "Enable free-thinking (Claude) voters",
    value=True,
    help=(
        "When enabled, a fraction of agents use Claude (via `claude -p`) to make "
        "voting decisions based on a persona prompt instead of rule-based logic. "
        "These agents receive their demographics, ideology, trust network opinions, "
        "and race context, and return a reasoned vote/delegate/abstain decision.\n\n"
        "**Warning:** This is slow — each LLM agent makes a subprocess call to Claude "
        "with a 30-second timeout. 5% of 1,000 agents = 50 Claude calls = ~2 min. "
        "Best used with smaller populations (200-500 agents)."
    ),
)

llm_fraction = 0.0
if use_llm:
    llm_fraction = st.sidebar.slider(
        "Free-thinking voter %", 1, 50, 10, step=1,
        help=(
            "Percentage of agents that use Claude for decisions instead of rule-based logic. "
            "These are selected from the top delegates (most-delegated-to agents), so the "
            "agents with the most influence are the ones reasoning about their choices.\n\n"
            "At 5%, the top ~50 delegates in a 1,000-agent sim use Claude. "
            "At 20%, ~200 agents call Claude — expect a 5-10 minute run."
        ),
    ) / 100.0

st.sidebar.subheader("Environment")

use_media = st.sidebar.toggle(
    "Enable media influence",
    value=False,
    help=(
        "When enabled, 10 simulated media outlets influence voter opinions each "
        "campaign tick. Each outlet has a fixed ideology, reach (fraction of "
        "population exposed per tick), and credibility (how much agents trust it). "
        "Outlets shift agents' ideology toward the outlet's position, modulated "
        "by confirmation bias and agent stubbornness.\n\n"
        "**The 10 outlets and their fixed parameters:**\n\n"
        "| Outlet | Lean | Reach | Credibility |\n"
        "|--------|------|-------|-------------|\n"
        "| MSNBC | -0.7 (left) | 25% | 0.55 |\n"
        "| CNN | -0.4 (center-left) | 28% | 0.50 |\n"
        "| NYT | -0.5 (center-left) | 15% | 0.70 |\n"
        "| Fox News | +0.7 (right) | 30% | 0.50 |\n"
        "| Daily Wire | +0.8 (far-right) | 12% | 0.40 |\n"
        "| WSJ | +0.4 (center-right) | 13% | 0.72 |\n"
        "| PBS | -0.1 (near-center) | 10% | 0.80 |\n"
        "| AP/Reuters | 0.0 (center) | 20% | 0.85 |\n"
        "| BBC | -0.15 (near-center) | 8% | 0.78 |\n"
        "| NPR | +0.05 (near-center) | 10% | 0.65 |\n\n"
        "**Reach** = what % of agents see this outlet each tick. "
        "**Credibility** = how much influence it has when seen (0-1). "
        "**Lean** = ideology position on [-1, +1] scale. "
        "These values are not adjustable individually — use the bias level "
        "toggle below to scale all leans uniformly."
    ),
)

media_bias_factor = 1.0

st.sidebar.divider()
seed = st.sidebar.number_input("Random seed", value=int(preset.get("seed", 42)), step=1)

ai_analysis = st.sidebar.toggle("AI Summary", value=True, help="Generate a Claude-powered analysis of the results")
_is_running = st.session_state.get("_running", False)
run_clicked = st.sidebar.button(
    "Running..." if _is_running else "RUN",
    type="primary", width="stretch", disabled=_is_running,
)

# ---------------------------------------------------------------------------
# Build config and run
# ---------------------------------------------------------------------------

config_dict = dict(
    n_agents=n_agents,
    pvi_lean=pvi_lean,
    seed=int(seed),
    district_id=selected_district,  # None = synthetic, str = real district
    use_ces=use_ces,
    delegation_probability_base=delegation_prob,
    viscous_decay_alpha=viscous_alpha,
    weight_cap=weight_cap,
    delegation_options_k=delegation_k,
    preferential_attachment_gamma=pref_attach_gamma,
    bandwagon_coefficient=bandwagon,
    bounded_confidence_epsilon=bounded_confidence,
    homophily=homophily,
    use_llm=use_llm,
    llm_agent_fraction=llm_fraction,
    use_media=use_media,
    media_bias_factor=media_bias_factor,
    races={"race": candidates},
)

# Clear stale results when config changes (before clicking RUN)
prev_config = st.session_state.get("config")
config_changed = prev_config is not None and prev_config != config_dict

if run_clicked:
    # Stage 1: update UI state and rerun so the guide closes and button disables
    st.session_state["seen_quick_start"] = False
    st.session_state["_running"] = True
    st.session_state["_pending_config"] = config_dict
    st.rerun()

if st.session_state.get("_running") and st.session_state.get("_pending_config") is not None:
    # Stage 2: actually run the simulation (page has already rerendered)
    _pending = st.session_state.pop("_pending_config")
    st.session_state["config"] = _pending
    st.session_state["results"] = run_simulation_with_progress(_pending)
    st.session_state["analysis"] = None
    st.session_state["graph_summaries"] = None
    st.session_state["_running"] = False
    st.rerun()

if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar and press **RUN** to start a simulation.")
    # Show quick start by default on first load
    if not st.session_state.get("_first_load_done"):
        st.session_state["seen_quick_start"] = True
        st.session_state["_first_load_done"] = True
        st.rerun()
    st.stop()

results = st.session_state["results"]
model: LiquidDemocracyModel | None = st.session_state.get("_model")
pvi_val = config_dict["pvi_lean"]
pvi_label = f"D+{abs(pvi_val):.0f}" if pvi_val < 0 else f"R+{pvi_val:.0f}" if pvi_val > 0 else "Even"
district_label = f" **{config_dict['district_id']}**" if config_dict.get("district_id") else ""
if config_dict.get("use_ces"):
    data_mode = "CES 2024 Survey"
elif config_dict.get("district_id"):
    data_mode = "2024 District Profile"
else:
    data_mode = "Synthetic"
st.caption(f"**{scenario}**{district_label} | {data_mode} | {n_agents:,} agents | {pvi_label} | seed {seed}")

# ---------------------------------------------------------------------------
# LLM error warning (if LLM agents abstained due to errors)
# ---------------------------------------------------------------------------

if model is not None and config_dict.get("use_llm"):
    _llm_agent_ids = {aid for aid, a in model.agents.items() if a.is_llm_agent}
    _llm_errors: list[str] = []
    # Error reasons start with known error prefixes; genuine abstentions don't
    _error_prefixes = ("llm_error", "batch timeout", "JSON parse", "claude exit",
                       "agent ", "empty response", "OS error", "exhausted")
    if _llm_agent_ids:
        for race_id, decs in model._llm_decisions.items():
            for entry in decs:
                aid, action = entry[0], entry[1]
                reason = entry[3] if len(entry) > 3 else ""
                if action == "abstain" and aid in _llm_agent_ids:
                    if any(reason.startswith(p) for p in _error_prefixes):
                        _llm_errors.append(f"Agent {aid}: {reason}")

    if _llm_errors:
        _total_llm = len(_llm_agent_ids)
        with st.expander(f"{len(_llm_errors)} LLM errors (click to see details)", expanded=False):
            for err in _llm_errors[:20]:
                st.caption(err)
            if len(_llm_errors) > 20:
                st.caption(f"... and {len(_llm_errors) - 20} more. Run `make dashboard-debug` for full logs.")

# ---------------------------------------------------------------------------
# AI Analysis (top of page)
# ---------------------------------------------------------------------------

if ai_analysis:
    from dashboard.llm_analysis import build_analysis_prompt

    if st.session_state.get("analysis") is None:
        _ai_status = st.empty()
        _ai_status.info("Waiting for Claude...")

        def _ai_progress(stderr_line: str | None, elapsed: float) -> None:
            if stderr_line and ("rate" in stderr_line.lower() or "limit" in stderr_line.lower()
                               or "retry" in stderr_line.lower() or "waiting" in stderr_line.lower()
                               or "throttl" in stderr_line.lower() or "429" in stderr_line):
                _ai_status.warning(f"Rate limited — {stderr_line}  ({elapsed:.0f}s)")
            elif stderr_line and ("error" in stderr_line.lower() or "fail" in stderr_line.lower()):
                _ai_status.error(f"{stderr_line}  ({elapsed:.0f}s)")
            elif elapsed < 10:
                _ai_status.info(f"Claude is analyzing the results...  ({elapsed:.0f}s)")
            elif elapsed < 30:
                _ai_status.info(f"Claude is writing the analysis...  ({elapsed:.0f}s)")
            elif elapsed < 60:
                _ai_status.info(f"Still generating...  ({elapsed:.0f}s)")
            else:
                _ai_status.info(f"Almost done...  ({elapsed:.0f}s)")

        analysis_text = analyze_results(results, config_dict, status_callback=_ai_progress)
        _ai_status.empty()
        st.session_state["analysis"] = analysis_text or "_Analysis unavailable._"

    analysis = st.session_state.get("analysis", "")
    if analysis:
        import re
        import json as _json

        # Split analysis into main body and "Try Next" suggestions
        _try_next_pattern = re.compile(r"###?\s*Try Next", re.IGNORECASE)
        _params_pattern = re.compile(r"<!--\s*PARAMS:\s*(\{.*?\})\s*-->")

        # Strip PARAMS comments from the raw analysis so they don't render as whitespace
        _clean_analysis = re.sub(r"\n*<!--\s*PARAMS:\s*\{.*?\}\s*-->\n*", "\n", analysis)
        # Collapse runs of 3+ newlines into 2
        _clean_analysis = re.sub(r"\n{3,}", "\n\n", _clean_analysis)

        _split = _try_next_pattern.split(_clean_analysis, maxsplit=1)
        _main_body = _split[0].strip()
        _try_next_raw = _split[1].strip() if len(_split) > 1 else ""

        # Re-split the original (with PARAMS) for button extraction
        _split_orig = _try_next_pattern.split(analysis, maxsplit=1)
        _try_next_section = _split_orig[1] if len(_split_orig) > 1 else ""

        with st.container(border=True):
            st.subheader("AI Analysis")
            st.markdown(_main_body)

            # Parse suggestions with their PARAMS
            if _try_next_section:
                st.markdown("### Try Next")

                # Extract params from original (with PARAMS comments)
                _parts = _params_pattern.split(_try_next_section)
                # _parts alternates: [text, json, text, json, ...]
                _suggestions: list[tuple[str, dict]] = []
                for i in range(0, len(_parts) - 1, 2):
                    text = _parts[i].strip()
                    try:
                        params = _json.loads(_parts[i + 1])
                    except (ValueError, IndexError):
                        params = {}
                    if text or params:
                        # Clean PARAMS comments from display text
                        clean_text = re.sub(r"<!--\s*PARAMS:.*?-->", "", text).strip()
                        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
                        _suggestions.append((clean_text, params))

                # Render any remaining text without params
                if len(_parts) % 2 == 1 and _parts[-1].strip():
                    trailing = re.sub(r"<!--\s*PARAMS:.*?-->", "", _parts[-1]).strip()
                    if not _suggestions:
                        if trailing:
                            st.markdown(trailing)
                    elif trailing:
                        _suggestions.append((trailing, {}))

                for idx, (text, params) in enumerate(_suggestions):
                    if text:
                        st.markdown(text)
                    if params:
                        # Build a readable label from the param changes
                        _label_parts = []
                        _PARAM_LABELS = {
                            "n_agents": "voters",
                            "pvi_lean": "PVI lean",
                            "delegation_probability_base": "delegation prob",
                            "viscous_decay_alpha": "viscous decay",
                            "weight_cap": "weight cap",
                            "delegation_options_k": "k",
                            "preferential_attachment_gamma": "pref. attachment",
                            "bandwagon_coefficient": "bandwagon",
                            "bounded_confidence_epsilon": "bounded conf.",
                            "homophily": "homophily",
                            "use_llm": "Claude voters",
                            "llm_agent_fraction": "Claude %",
                            "use_media": "media",
                            "media_bias_factor": "media bias",
                            "races": "candidates",
                        }
                        for k, v in params.items():
                            name = _PARAM_LABELS.get(k, k)
                            val = "off" if v is None else v
                            _label_parts.append(f"{name}={val}")
                        btn_label = "Try: " + ", ".join(_label_parts)
                        if st.button(btn_label, key=f"try_next_{idx}"):
                            new_config = {**config_dict, **params}
                            st.session_state["_running"] = True
                            st.session_state["_pending_config"] = new_config
                            st.session_state["analysis"] = None
                            st.session_state["graph_summaries"] = None
                            del st.session_state["results"]
                            st.rerun()

            with st.expander("Show prompt sent to Claude"):
                st.code(build_analysis_prompt(results, config_dict), language="text")

# ---------------------------------------------------------------------------
# Campaign trail summary
# ---------------------------------------------------------------------------

if model is not None:
    with st.expander("Campaign Trail Summary"):
        st.markdown(build_campaign_summary(model))

# ---------------------------------------------------------------------------
# Election outcomes
# ---------------------------------------------------------------------------

st.subheader("Election Outcomes")
_race_config = config_dict.get("races", {})
_n_candidates = sum(len(v) for v in _race_config.values())
_candidate_names = ", ".join(c for cands in _race_config.values() for c in cands)
st.caption(
    f"**{_n_candidates} candidates: {_candidate_names}.** "
    + GRAPH_HELP["outcomes"]
)

race_ids = list(results["fptp_results"].keys())
selected_race = race_ids[0] if race_ids else "race"

# Generate per-graph AI summaries (one Claude call for all charts)
if ai_analysis and model is not None and st.session_state.get("graph_summaries") is None:
    _gdata = _collect_graph_data(results, config_dict, model, selected_race)
    with st.spinner("Generating graph insights..."):
        st.session_state["graph_summaries"] = generate_graph_summaries(_gdata)

_graph_summaries = st.session_state.get("graph_summaries", {}) if ai_analysis else {}


def _ai_caption(key: str) -> None:
    """Display the AI-generated summary for *key* in italics, if available."""
    s = _graph_summaries.get(key)
    if s:
        st.markdown(f"*{s}*")


fptp_r = results["fptp_results"].get(selected_race, {})
rcv_r = results["rcv_results"].get(selected_race, {})
trs_r = results["trs_results"].get(selected_race, {})
deleg_r = results["delegation_results"].get(selected_race, {})

system_results = [
    ("US System (FPTP)", fptp_r),
    ("Ranked Choice (RCV)", rcv_r),
    ("Two-Round (TRS)", trs_r),
    ("Liquid Delegation", deleg_r),
]
winners = {label: _get_winner(r) for label, r in system_results}
unique_winners = set(w for w in winners.values() if w is not None)
has_divergence = len(unique_winners) > 1

import plotly.graph_objects as go

_PARTY_COLORS = {"Democrat": "#3b82f6", "Republican": "#ef4444", "Libertarian": "#f59e0b", "Green": "#22c55e", "Independent": "#a855f7"}

cols = st.columns(4)
for col, (system, winner) in zip(cols, winners.items()):
    result_data = dict(system_results)[system]
    counts = _get_counts(result_data)
    total = sum(counts.values()) if counts else 0
    winner_color = _PARTY_COLORS.get(winner, "#6b7280") if winner else "#6b7280"

    with col:
        fptp_winner = list(winners.values())[0]
        diverges_from_us = has_divergence and winner != fptp_winner and winner is not None

        # System label — red if divergent from FPTP
        if diverges_from_us:
            st.markdown(
                f":red[**{system}**] :red-background[different winner]",
                help=(
                    f"This system elected **{winner}** instead of "
                    f"**{fptp_winner}** (the US system result). "
                    "Same voters, same preferences — the counting method "
                    "changed the outcome."
                ),
            )
        else:
            st.markdown(f"**{system}**")

        # Winner name in party color
        st.markdown(
            f'<span style="color:{winner_color};font-size:1.3rem;font-weight:600">'
            f'{winner or "N/A"}</span>',
            unsafe_allow_html=True,
        )

        if counts and total > 0:
            sorted_items = sorted(counts.items(), key=lambda x: -x[1])
            labels = [c for c, _ in sorted_items]
            values = [v for _, v in sorted_items]
            colors = [_PARTY_COLORS.get(c, "#6b7280") for c in labels]
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
                sort=False,
            )])
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=5, b=5, l=5, r=5),
                height=150,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False}, key=f"pie_{system}")
            # System-specific round context
            decided_in = result_data.get("decided_in")
            rcv_rounds = result_data.get("rounds")
            eliminated = result_data.get("eliminated", [])
            if decided_in == 2:
                st.caption(f"{total:,.0f} votes (round 2 runoff)")
            elif decided_in == 1:
                st.caption(f"{total:,.0f} votes (won round 1)")
            elif rcv_rounds and len(rcv_rounds) > 1:
                st.caption(
                    f"{total:,.0f} votes (after {len(eliminated)} "
                    f"elimination{'s' if len(eliminated) != 1 else ''})"
                )
            else:
                st.caption(f"{total:,.0f} total votes")
        else:
            st.caption("No results")

_ai_caption("election_outcomes")

# ---------------------------------------------------------------------------
# Preference shifts (initial expected → final vote per system)
# ---------------------------------------------------------------------------

st.subheader("Voter Preference Shifts")
st.caption(
    "Shows where voters started (their closest candidate based on initial ideology) "
    "versus where they ended up voting. Gray flows = stayed with initial preference. "
    "Orange flows = shifted to a different candidate during the campaign.\n\n"
    "**Why voters abstain:** Abstention is checked *before* vote choice via four "
    "independent factors: **apathy** (low engagement, ~5% max), **futility** "
    "(minority party in a lopsided district, ~5% max), **efficacy** (low political "
    "knowledge, ~4% max), and **access barriers** (low income + rural, ~3% max). "
    "Knowledge is derived from education (0=no high school → 0.30, 1=high school → 0.38, "
    "2=some college → 0.46, 3=bachelor's → 0.54, 4=master's → 0.62, 5=PhD → 0.70). "
    "Agents with knowledge above 0.60 (master's+) vote on informed preference. "
    "Below-threshold agents check for a trusted delegate; if none, they fall through "
    "to a party-line heuristic — voting along party lines rather than on issue knowledge."
)

if model is not None:
    _shift_candidates = list(model.config.races.get(selected_race, ["Democrat", "Republican"]))
    _init_ideo = model._initial_ideologies

    # Build per-system vote choice dicts
    # FPTP: direct votes only
    _fptp_votes = dict(model._direct_votes.get(selected_race, {}))

    # Delegation: need to map through delegation graph to final candidate
    # Use direct votes + delegated votes resolved to candidates
    _deleg_votes: dict[int, str] = {}
    for aid, agent in model.agents.items():
        rs = agent.race_states.get(selected_race)
        if rs and rs.vote_choice:
            _deleg_votes[aid] = rs.vote_choice
        elif rs and rs.delegation_targets:
            # Follow delegation chain to find what candidate the delegate voted for
            delegate_id = rs.delegation_targets[0]
            delegate = model.agents.get(delegate_id)
            if delegate:
                drs = delegate.race_states.get(selected_race)
                if drs and drs.vote_choice:
                    _deleg_votes[aid] = drs.vote_choice

    shift_cols = st.columns(2)
    with shift_cols[0]:
        st.markdown("**US System (FPTP)**")
        fig_shift_fptp = plot_preference_shift(
            model.agents, _init_ideo, selected_race, _shift_candidates,
            vote_choices=_fptp_votes,
        )
        st.plotly_chart(fig_shift_fptp, width="stretch", config={"displayModeBar": False}, key="shift_fptp")

    with shift_cols[1]:
        st.markdown("**Liquid Delegation**")
        fig_shift_deleg = plot_preference_shift(
            model.agents, _init_ideo, selected_race, _shift_candidates,
            vote_choices=_deleg_votes,
        )
        st.plotly_chart(fig_shift_deleg, width="stretch", config={"displayModeBar": False}, key="shift_deleg")

    # RCV shows the interesting elimination-driven shifts
    shift_cols2 = st.columns(2)

    _rcv_votes: dict[int, str] = {}
    for aid, agent in model.agents.items():
        rs = agent.race_states.get(selected_race)
        if rs and rs.ranked_choices:
            _rcv_votes[aid] = rs.ranked_choices[0]

    with shift_cols2[0]:
        st.markdown("**Ranked Choice (1st pref)**")
        fig_shift_rcv = plot_preference_shift(
            model.agents, _init_ideo, selected_race, _shift_candidates,
            vote_choices=_rcv_votes,
        )
        st.plotly_chart(fig_shift_rcv, width="stretch", config={"displayModeBar": False}, key="shift_rcv")

    # TRS: same as FPTP for round 1
    with shift_cols2[1]:
        st.markdown("**Two-Round (TRS)**")
        fig_shift_trs = plot_preference_shift(
            model.agents, _init_ideo, selected_race, _shift_candidates,
            vote_choices=_fptp_votes,
        )
        st.plotly_chart(fig_shift_trs, width="stretch", config={"displayModeBar": False}, key="shift_trs")
else:
    st.info("Preference shift charts require a fresh simulation run.")

# ---------------------------------------------------------------------------
# Network + Lorenz
# ---------------------------------------------------------------------------

st.subheader("Network and Power Analysis")
col_net, col_power = st.columns(2)

with col_net:
    st.caption(GRAPH_HELP["network"])
    if model is not None:
        fig_net = plot_delegation_network(model.delegation_graph, model.agents, selected_race)
        st.plotly_chart(fig_net, width="stretch")
        _ai_caption("network")
    else:
        st.info("Network visualization requires a fresh simulation run.")

with col_power:
    st.caption(
        "Shows how much delegation deviates from equal voting power. "
        "In FPTP/RCV/TRS everyone has weight 1 (the zero line). In liquid delegation, "
        "delegators lose their weight and delegates gain it — the purple area IS the "
        "Gini coefficient. A tall peak means a few delegates hold outsized power."
    )
    # Get per-voter delegation weights
    deleg_w = [1.0] * max(config_dict["n_agents"], 1)
    if model is not None:
        weights = model.delegation_graph.resolve_all(
            selected_race, weight_cap=config_dict.get("weight_cap"),
        )
        if weights:
            all_weights = []
            graph_voter_ids = set(weights.keys())
            for aid in model.agents:
                aid_str = str(aid)
                if aid_str in graph_voter_ids:
                    all_weights.append(weights[aid_str])
                else:
                    all_weights.append(1.0)
            deleg_w = all_weights

    fig_dev = plot_deviation_from_equality(deleg_w)
    st.plotly_chart(fig_dev, width="stretch")
    _ai_caption("deviation")

# Wasted votes / Gallagher index
st.subheader("Representation Efficiency")
st.caption(
    "Compares how effectively each system translates votes into representation. "
    "**Wasted votes** = votes that didn't help elect the winner (loser votes in FPTP, "
    "redistributed-then-lost in RCV, eliminated in TRS round 1, weighted loser votes in delegation). "
    "The **Gallagher index** (white line) measures disproportionality between vote share and "
    "outcome — higher means the winner's seat share diverges more from their vote share."
)
fig_wasted = plot_wasted_votes(fptp_r, rcv_r, trs_r, deleg_r, config_dict["n_agents"])
st.plotly_chart(fig_wasted, width="stretch")
_ai_caption("wasted_votes")

# ---------------------------------------------------------------------------
# Sankey + distributions
# ---------------------------------------------------------------------------

st.subheader("Delegation Details")
col_sankey, col_dist = st.columns(2)

with col_sankey:
    st.caption(GRAPH_HELP["sankey"])
    if model is not None:
        candidates = list(model.config.races.get(selected_race, ["Democrat", "Republican"]))
        direct_votes = model._direct_votes.get(selected_race, {})
        fig_sankey = plot_delegation_sankey(
            model.delegation_graph, model.agents, direct_votes, selected_race, candidates,
        )
        st.plotly_chart(fig_sankey, width="stretch")
        _ai_caption("sankey")
    else:
        st.info("Sankey diagram requires a fresh simulation run.")

with col_dist:
    dist_choice = st.radio("Distribution", ["Chain lengths", "Vote weights"], horizontal=True)
    help_key = "chain_dist" if dist_choice == "Chain lengths" else "weight_dist"
    st.caption(GRAPH_HELP[help_key])
    if model is not None:
        if dist_choice == "Chain lengths":
            fig_dist = plot_chain_length_distribution(model.delegation_graph, selected_race)
        else:
            fig_dist = plot_weight_distribution(model.delegation_graph, selected_race)
        st.plotly_chart(fig_dist, width="stretch")
        _ai_caption("chain_lengths" if dist_choice == "Chain lengths" else "weight_dist")
    else:
        st.info("Distribution charts require a fresh simulation run.")

# ---------------------------------------------------------------------------
# Opinion timeline
# ---------------------------------------------------------------------------

st.subheader("Opinion Evolution")
st.caption(GRAPH_HELP["opinion"])
if model is not None:
    fig_timeline = plot_opinion_timeline(results["opinion_history"], model.agents)
    st.plotly_chart(fig_timeline, width="stretch")
    _ai_caption("opinion_timeline")
else:
    st.info("Opinion timeline requires a fresh simulation run.")

# ---------------------------------------------------------------------------
# Gini over time
# ---------------------------------------------------------------------------

gini_hist = results["delegation_stats"].get("gini_history", [])
if gini_hist:
    st.subheader("Power Concentration Over Time")
    st.caption(GRAPH_HELP["gini_time"])
    fig_gini = plot_gini_over_time(gini_hist)
    st.plotly_chart(fig_gini, width="stretch")
    _ai_caption("gini_time")

# ---------------------------------------------------------------------------
# Delegation stats
# ---------------------------------------------------------------------------

st.subheader("Delegation Statistics")
st.caption(
    "Summary metrics for the delegation system. Gini measures power concentration "
    "(0 = equal, 1 = total concentration). Max weight shows the most votes any single "
    "delegate carried."
)
stats = results["delegation_stats"]

m1, m2, m3 = st.columns(3)
m1.metric(
    "Total delegators",
    f"{stats.get('total_delegators', 0):,}",
    help="Agents who delegated their vote to someone else instead of voting directly.",
)
m2.metric(
    "Direct voters",
    f"{stats.get('total_direct_voters', 0):,}",
    help="Agents who cast their own vote without delegating.",
)
_abs_reasons = stats.get("abstention_reasons", {})
_abs_breakdown = ", ".join(f"{r}: {c}" for r, c in sorted(_abs_reasons.items(), key=lambda x: -x[1])) if _abs_reasons else "none"
m3.metric(
    "Abstentions",
    f"{stats.get('total_abstentions', 0):,}",
    help=(
        "Agents who sat out the election. Breakdown by reason: "
        f"{_abs_breakdown}. "
        "Apathy = low engagement, futility = lopsided district, "
        "efficacy = low knowledge, access = demographic barriers."
    ),
)

race_id = selected_race
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Gini",
    f"{stats.get('gini_per_race', {}).get(race_id, 0):.4f}",
    help="Gini coefficient of vote weight distribution. 0 = perfectly equal, 1 = one person holds all votes.",
)
c2.metric(
    "Max weight",
    f"{stats.get('max_weight_per_race', {}).get(race_id, 0):.2f}",
    help="The most votes any single delegate carried. Weight 1.0 = one person's vote.",
)
c3.metric(
    "Chain count",
    f"{stats.get('chain_count_per_race', {}).get(race_id, 0):,}",
    help="Total number of delegation chains registered for this race.",
)
c4.metric(
    "Avg chain length",
    f"{stats.get('avg_chain_length_per_race', {}).get(race_id, 0):.2f}",
    help="Average number of hops in delegation chains. Longer = more transitive trust.",
)

# ---------------------------------------------------------------------------
# Campaign Dynamics: convergence, stance shifts, cross-party movement
# ---------------------------------------------------------------------------

if model is not None:
    dynamics = model.get_dynamics_report()

    st.subheader("Campaign Dynamics")
    st.caption(
        "How opinions evolved during the 80-tick campaign phase. "
        "Convergence = when opinion changes dropped below 0.1% per tick. "
        "Cross-party shifts = agents whose ideology moved from D-leaning to R-leaning (or vice versa)."
    )

    # Top-line metrics
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Converged at tick",
        dynamics["convergence_tick"] if dynamics["converged"] else "Did not converge",
        help="The tick where mean opinion change per step dropped below 0.001. "
             "Lower = faster stabilization. With high stubbornness, this happens quickly.",
    )
    d2.metric(
        "Cross-party shifts",
        f"{dynamics['total_cross_party_shifts']} ({dynamics['cross_party_pct']}%)",
        help="Agents whose ideology crossed the D/R threshold during the campaign "
             "(moved from D-leaning to R-leaning or vice versa).",
    )
    d3.metric(
        "Stance changes",
        dynamics["total_stance_changes"],
        help="Agents whose actual vote differed from what their initial party lean predicted. "
             "An initially D-leaning agent voting Republican counts as a stance change.",
    )
    d4.metric(
        "Delegation turnout boost",
        f"{dynamics['delegation_turnout_boost']:+.0f} votes",
        help="Extra effective votes in delegation vs FPTP. "
             "Delegation can increase turnout because delegates vote on behalf of abstainers.",
    )

    # Per-group shift breakdown
    shift_data = dynamics.get("shift_by_group", {})
    if shift_data:
        st.markdown("**Opinion shifts by initial party**")
        group_order = ["strong_D", "lean_D", "independent", "lean_R", "strong_R"]
        visible_groups = [g for g in group_order if g in shift_data]

        cols = st.columns(len(visible_groups)) if visible_groups else []
        for col, group in zip(cols, visible_groups):
            data = shift_data[group]
            total = data["total"]
            with col:
                label = group.replace("_", " ").title()
                st.markdown(f"**{label}** ({total})")
                if total > 0:
                    st.caption(f"Shifted right: {data['shifted_right']} ({data['shifted_right']/total*100:.0f}%)")
                    st.caption(f"Shifted left: {data['shifted_left']} ({data['shifted_left']/total*100:.0f}%)")
                    st.caption(f"Crossed party: {data['crossed_party']} ({data['crossed_party']/total*100:.0f}%)")
                    st.caption(f"Mean shift: {data['mean_shift']:+.3f}")

    # Convergence curve — per-party breakdown with trust-rebuild annotations
    deltas = dynamics.get("opinion_deltas", [])
    if deltas and model is not None:
        import plotly.graph_objects as go
        from dashboard.distribution_panels import PARTY_COLORS

        snapshots = model._opinion_snapshots
        agent_ids = sorted(model.agents.keys())

        # Build party → index mapping
        party_indices: dict[str, list[int]] = {}
        for idx, aid in enumerate(agent_ids):
            party = model._initial_parties.get(aid, "independent")
            party_indices.setdefault(party, []).append(idx)

        fig_conv = go.Figure()
        ticks = list(range(1, len(snapshots)))

        # Per-party convergence lines (consistent D → Ind → R order)
        from dashboard.distribution_panels import PARTY_ORDER
        _conv_parties = [p for p in PARTY_ORDER if p in party_indices]
        _conv_parties += [p for p in sorted(party_indices) if p not in _conv_parties]
        for party in _conv_parties:
            indices = party_indices[party]
            party_deltas = []
            for t in range(1, len(snapshots)):
                prev = snapshots[t - 1][indices]
                curr = snapshots[t][indices]
                party_deltas.append(float(np.mean(np.abs(curr - prev))))
            label = party.replace("_", " ").title()
            fig_conv.add_trace(go.Scatter(
                x=ticks, y=party_deltas,
                mode="lines",
                name=label,
                line=dict(
                    color=PARTY_COLORS.get(party, "#999999"),
                    width=2,
                ),
            ))

        # Aggregate line (existing deltas) — dashed overlay
        fig_conv.add_trace(go.Scatter(
            x=ticks, y=deltas,
            mode="lines",
            name="All voters",
            line=dict(color="#AB63FA", width=2, dash="dash"),
        ))

        # Trust rebuild markers at every 10th tick
        max_delta = max(deltas) if deltas else 0.05
        for t in range(10, len(deltas) + 1, 10):
            fig_conv.add_vline(
                x=t, line_dash="dot", line_color="rgba(255,255,255,0.2)",
            )
        # Single annotation explaining the pattern
        first_rebuild = min(10, len(deltas))
        fig_conv.add_annotation(
            x=first_rebuild, y=max_delta * 0.85,
            text="Trust & influence<br>matrix rebuilt",
            showarrow=True, arrowhead=2,
            font=dict(size=10, color="rgba(255,255,255,0.6)"),
            arrowcolor="rgba(255,255,255,0.4)",
        )

        if dynamics["converged"]:
            fig_conv.add_vline(
                x=dynamics["convergence_tick"], line_dash="dash",
                line_color="gray",
                annotation_text=f"Converged (tick {dynamics['convergence_tick']})",
            )
        fig_conv.update_layout(
            title="Opinion Convergence Rate",
            xaxis_title="Tick", yaxis_title="Mean absolute opinion change",
            template="plotly_dark", height=350,
            margin=dict(l=50, r=30, t=50, b=50),
        )
        st.plotly_chart(fig_conv, width="stretch")
        st.caption(
            "Spikes at every 10th tick occur because the simulation rebuilds trust scores "
            "and the social influence matrix — agents suddenly 'hear' updated neighbor opinions, "
            "causing a brief jolt before settling again. Each party group converges at its own "
            "pace: independents (low stubbornness) shift fastest, strong partisans resist longest."
        )
        _ai_caption("convergence")

# ---------------------------------------------------------------------------
# LLM vs Rule-based comparison (only shown when LLM agents are enabled)
# ---------------------------------------------------------------------------

if model is not None and config_dict.get("use_llm") and hasattr(model, "get_llm_vs_rule_stats"):
    llm_stats = model.get_llm_vs_rule_stats()
    if llm_stats.get("llm_enabled") and llm_stats.get("n_llm", 0) > 0:
        st.subheader("Free-Thinking vs Rule-Based Agents")
        st.caption(
            "Compares how LLM-powered agents (who reason about their vote via Claude) "
            "behave differently from rule-based agents (who follow parametric decision logic). "
            "Differences suggest the LLM agents are making genuinely independent choices "
            "rather than following the same heuristics."
        )

        race_data = llm_stats.get(selected_race, {})
        llm_d = race_data.get("llm", {})
        rule_d = race_data.get("rule", {})

        # --- Helper: compute "voted against initial lean" stats for a group ---
        def _lean_vs_vote(decs: list) -> dict:
            """Count how many agents voted with vs against their initial party lean."""
            with_lean = 0
            against_lean = 0
            by_initial_party: dict[str, dict[str, int]] = {}
            for dec in decs:
                aid, action, choice = dec[0], dec[1], dec[2]
                if action != "vote" or not choice:
                    continue
                init_ideo = model._initial_ideologies.get(aid, 0.0)
                init_party = model._initial_parties.get(aid, "independent")
                expected = "Democrat" if init_ideo < 0 else "Republican" if init_ideo > 0 else choice
                if choice == expected:
                    with_lean += 1
                else:
                    against_lean += 1
                # Track by initial party group
                label = init_party.replace("_", " ").title()
                by_initial_party.setdefault(label, {"total": 0, "against": 0})
                by_initial_party[label]["total"] += 1
                if choice != expected:
                    by_initial_party[label]["against"] += 1
            total = with_lean + against_lean
            return {
                "with_lean": with_lean,
                "against_lean": against_lean,
                "against_pct": round(against_lean / total * 100, 1) if total else 0,
                "by_party": by_initial_party,
            }

        llm_decs = model._llm_decisions.get(selected_race, [])
        rule_decs = model._rule_decisions.get(selected_race, [])
        llm_lean = _lean_vs_vote(llm_decs)
        rule_lean = _lean_vs_vote(rule_decs)

        # --- Two-column comparison ---
        col_llm, col_rule = st.columns(2)

        for col, label, n_agents, d, lean_data, decs in [
            (col_llm, "Free-thinking (Claude)", llm_stats["n_llm"], llm_d, llm_lean, llm_decs),
            (col_rule, "Rule-based", llm_stats["n_rule"], rule_d, rule_lean, rule_decs),
        ]:
            with col:
                st.markdown(f"**{label}** — {n_agents} agents")

                if d.get("actions"):
                    action_parts = [f"{a} {pct}%" for a, pct in sorted(d["actions"].items())]
                    st.caption(" | ".join(action_parts))

                if d.get("choices"):
                    st.markdown("**Vote split:**")
                    for cand, pct in sorted(d["choices"].items(), key=lambda x: -x[1]):
                        st.progress(pct / 100, text=f"{cand}: {pct}%")

                # Voted against lean
                if lean_data["with_lean"] + lean_data["against_lean"] > 0:
                    st.markdown(
                        f"**Voted against initial lean:** {lean_data['against_lean']} "
                        f"({lean_data['against_pct']}%)"
                    )
                    if lean_data["by_party"]:
                        parts = []
                        for party, counts in sorted(lean_data["by_party"].items()):
                            if counts["against"] > 0:
                                parts.append(f"{counts['against']}/{counts['total']} {party}")
                        if parts:
                            st.caption("Defectors: " + ", ".join(parts))

        # --- Top reasons from LLM agents ---
        llm_reasons = llm_d.get("top_reasons", [])
        llm_total = llm_d.get("total", 0)
        if llm_reasons:
            st.markdown(f"**Top reasons from Claude agents** ({llm_total} total):")
            for reason, count in llm_reasons:
                pct = round(count / max(llm_total, 1) * 100)
                st.caption(f"- \"{reason}\" ({count} agents, {pct}%)")

        # --- Divergence callout ---
        llm_choices = llm_d.get("choices", {})
        rule_choices = rule_d.get("choices", {})
        if llm_choices and rule_choices:
            all_cands = set(llm_choices) | set(rule_choices)
            max_diff = 0
            diff_cand = ""
            for c in all_cands:
                diff = abs(llm_choices.get(c, 0) - rule_choices.get(c, 0))
                if diff > max_diff:
                    max_diff = diff
                    diff_cand = c

            lean_diff = abs(llm_lean["against_pct"] - rule_lean["against_pct"])
            if max_diff > 5:
                msg = (
                    f"Largest divergence: **{diff_cand}** — "
                    f"LLM agents voted {llm_choices.get(diff_cand, 0):.0f}% vs "
                    f"rule-based {rule_choices.get(diff_cand, 0):.0f}% "
                    f"(a {max_diff:.0f} point gap)."
                )
                if lean_diff > 3:
                    msg += (
                        f" Claude agents also broke from their initial lean at "
                        f"{llm_lean['against_pct']}% vs {rule_lean['against_pct']}% "
                        f"for rule-based — suggesting the LLM reasoning led to "
                        f"genuinely different conclusions."
                    )
                st.info(msg)
            else:
                st.caption("LLM and rule-based agents voted similarly (< 5 point gap).")
