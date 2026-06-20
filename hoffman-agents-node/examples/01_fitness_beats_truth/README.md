# Experiment 1: Fitness Beats Truth

## What This Tests

Donald Hoffman's core claim: perception is a user interface that hides objective reality to show fitness-relevant icons. The agent with *less information* (a compressed interface) should predict *better* than an agent with full information.

## How It Works

A `HiddenMarkovWorld` with 20 true states organized into 5 groups (high intra-group transition probability). Three conditions:

1. **Interface CA** — sees only the 5-group projection (a fitness-relevant interface)
2. **Truth CA** — sees all 20 true states (god's eye view)
3. **Random Projection CA** — sees a random 5-state projection (bad interface)

## Expected Result

Interface CA achieves significantly higher prediction improvement than Truth CA. The group projection carves nature at its joints by hiding irrelevant within-group variation.

## Interpreting Results

- Interface CA improvement > Truth CA improvement → **Hoffman confirmed**
- Truth CA improvement ≈ Interface CA improvement → **No evidence for or against** (world may be simple enough that compression doesn't matter)
- Truth CA improvement > Interface CA improvement → **Hoffman not supported** (compression loses too much information)
