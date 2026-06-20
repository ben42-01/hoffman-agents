# Experiment 2: Quantum Algebra Convergence

## What This Tests

Hoffman's theory predicts that the combination operator ⊗ should have specific algebraic properties, including tensor-product structure in the combined agent's dynamics. This experiment tests whether the implemented ⊗ (path union + averaging) produces eigenvalue spectra consistent with a tensor product.

## How It Works

1. Run Tree of Life to generate agents at multiple combination depths (L0→L1→L2→L3)
2. Extract Markov transition matrices from each agent's meta-trie
3. Compute spectral gap, detailed balance, entropy for each agent
4. Compare gap ratios across levels — do they converge to 1.0 (tensor product prediction)?

## Expected Result

Gap ratios converge toward 1.0 as combination depth increases, even though nobody programmed the tensor product into the code.

## Why This Matters (from FOR_DR_HOFFMAN.md)

> "This result I genuinely cannot explain. The combination operator was implemented as simple path union and averaging. It was not written as a tensor product. Yet the eigenvalue spectra converge toward what a tensor product would predict. This feels significant but I lack the mathematical training to understand why."
