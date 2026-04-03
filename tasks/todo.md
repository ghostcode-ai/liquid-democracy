# Liquid Democracy Simulation — Build Plan

## Phase 0: Foundation
- [x] Project scaffold (dirs, pyproject.toml, __init__.py)
- [x] VoterAgent dataclass with demographics, ideology, behavioral params
- [x] Commit Phase 0

## Phase 1: Voting Engine
- [x] DelegationGraph (add/remove/resolve/cycle-detect/gini/viscous-decay/weight-cap)
- [x] Tests for DelegationGraph (cycles, long chains, weight caps, Gini)
- [x] FPTP tally engine
- [x] RCV tally engine with elimination rounds + exhausted ballots
- [x] TRS two-round tally with withdrawal simulation
- [x] Delegation tally (resolve chains -> weighted vote)
- [x] Tests for all 4 tally engines
- [x] Unified election runner (runner.py)
- [x] Commit Phase 1

## Phase 2: Agent Behavior
- [x] Friedkin-Johnsen opinion dynamics + bounded confidence
- [x] Trust formation model (agreement + social proof + betrayal)
- [x] Agent decision engine integration
- [x] Claude Code LLM bridge (subprocess wrapper)
- [x] Social desirability gap (private != public)
- [x] Synthetic agent seeding engine
- [x] Mesa-style simulation controller with 3-phase tick loop
- [x] Social network generation (Watts-Strogatz + homophily)
- [x] Tests for opinion dynamics + seeding
- [x] Commit Phase 2

## Phase 3: Influence + Scenarios
- [x] Media agent injection
- [x] Bandwagon coefficient for delegation
- [x] 5 scenarios: baseline, celebrity, hub-attack, stale-decay, k=2
- [x] Parameter sweep script
- [x] Commit Phase 3

## Phase 4: Dashboard
- [x] Streamlit app with scenario selector
- [x] Network visualization (Plotly)
- [x] Distribution panels (Gini, chain lengths, Lorenz)
- [x] Opinion timeline chart
- [x] Parameter sliders with re-run
- [x] CLI entry point (scripts/run_simulation.py)
- [x] Integration tests (11 E2E tests, 93 total)
- [x] Commit Phase 4

## Review
- 93 tests passing across 5 test files
- E2E validated: 500 agents, Gini 0.55-0.70, system divergence observed
- D+20 and R+20 districts produce correct party winners
- Weight caps correctly limit delegation power
- All four voting systems produce results on same electorate
