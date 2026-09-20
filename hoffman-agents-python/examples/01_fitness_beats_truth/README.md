# Experiment 1: Fitness Beats Truth

## What This Tests

Donald Hoffman's interface theory holds that perception is a user interface that hides objective reality behind fitness-relevant icons. His *fitness-beats-truth* theorem (Prakash et al.) concerns evolutionary games: perceptual strategies tuned to fitness out-compete strategies tuned to truth.

This experiment tests a much weaker, predictive analogue: can an agent with *less information* (a compressed interface) predict *better* than an agent with full information?

## How It Works

A `HiddenMarkovWorld` with 20 true states organized into 5 groups (high intra-group transition probability). Three conditions:

1. **Interface CA** — sees only the 5-group projection (a fitness-relevant interface)
2. **Truth CA** — sees all 20 true states (god's eye view)
3. **Random Projection CA** — sees a random 5-state projection (bad interface)

## Expected Result

Interface CA achieves significantly higher prediction improvement than Truth CA. The group projection carves nature at its joints by hiding irrelevant within-group variation.

## Interpreting Results

- Interface CA improvement > Truth CA improvement → **consistent with the interface hypothesis** in this world
- Truth CA improvement ≈ Interface CA improvement → **no evidence either way** (the world may be simple enough that compression doesn't matter)
- Truth CA improvement > Interface CA improvement → **not supported** in this world (compression loses too much information)

The result depends on the world: a projection that matches the world's group structure helps, and a random projection (the control) does not. It does not test the evolutionary theorem itself, which is about selection between strategies, not prediction accuracy.
