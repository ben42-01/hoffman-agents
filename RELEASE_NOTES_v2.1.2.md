## v2.1.2 — "The Observer Effect"

A release about seeing clearly: the bugs we'd been living with, the structural limits we'd been guessing at, and the experiments that finally proved both.

### Bug Fixes (6 Critical)

**Meta-state ID mask collision** — `combine()`'s joint meta-trie tagged constituents with `0x10000000`/`0x20000000` bits that could collide with naturally-occurring high bits in the SHA-256 hash output, corrupting 40-45% of registry entries across every combine→fuse cycle in both languages. Fixed by reserving the top 4 bits via `& 0x0FFFFFFF`. Verified empirically: overlap went from ~45% to 0% (Python) / 4% (Node, within normal hash collision probability). The old `quantum_signature`/`markov_transition` example results were likely affected by this — if you saw weird spectral gaps and weren't sure whether to trust them, the answer was "no, don't."

**`lastWorldStateId` silently dropped on serialize/deserialize** — every time an agent was saved to `.soul` and reloaded, its `lastWorldStateId` reset to `null`. The very next `perceive()` call treated the agent as if it had just been born — no prediction made, no transition edge inserted, `predictionError` forced to 0.5. The agent's continuity of experience was silently severed on every reload. Now preserved correctly in both languages.

**Python `decide()` ignored `agent._rng`** — two agents seeded with the same `rng` stream could produce different output sequences because `decide()` never received or used the configured RNG; it always fell back to the global `random` module. Same for `invent_token()` calls inside `perceive()`. All randomness calls now respect the agent's configured RNG stream. Reproducibility works as documented.

**Node `perceive()` ran lexicon mutations in frozen mode** — `setMode('frozen')` documented "deterministic projection, no learning" and correctly froze trie updates and meta-observations, but silently continued decaying the lexicon, adopting tokens from other agents, and binding proto-words. Frozen now actually freezes everything, matching Python's correct behavior.

**Python `fuse()` returned wrong agent IDs** — `constituent_ids` was a `frozenset`, which iterates in hash order (not insertion order), so `fuse()` would consistently pair the wrong constituent's `agent_id` with the wrong split meta-trie's data. Changed to ordered `tuple` to match Node's `Set` behavior. Verified empirically: `f1.agent_id == 'ALPHA'` now correctly pairs with ALPHA's actual experience.

### New API Features

- **`Prediction` class** + `agent.predictNext()` / `agent.predict_next()` — returns the most likely next discrete state, a well-calibrated confidence score (verified: 80%-confident predictions are right ~80%), and a ranked list of alternatives via `topK(n)`. Both languages, fully export wired.
- **`ConsciousAgent.fromConfig()` / `from_config()`** — static factory for config-driven agent construction. Node only; Python already had it.
- **`buildWorldFromDataFrame()`** — build a Markov world from an array of row objects. Node only; Python already had it.
- **`Topology.maybeAddConnection()` / `getConnectionStrength()` / `getAgentObservers()`** — multi-agent network introspection. Node only; Python already had them.
- **Seeded RNG for `CoinTossWorld` and `World`** — both now accept an optional `seed` parameter (default 42) for reproducible experiments. Matches Python's `np.random.RandomState(42)`. Node only.
- **~17 new top-level exports** in Node's `src/index.js`: `TrieNode`, `MetaStateSnapshot`, `populationReferenceScore`, `populationLoopScore`, `firstDepthNGeneration`, `prune`, `traceDistance`, `mergeSimilarPaths`, `inventToken`, `isInventedToken`, `EnvironmentState`, `sequenceToStateId`, `Normalizer`, `FeatureSpec`, `Topology`, `InteractionCycle`, `Prediction`. Top-level exports in Python's `__init__.py` also expanded to parity.

- **All 3 CI/CD workflows** (`publish-pypi.yml`, `publish-npm.yml`, `deploy-pages.yml`) updated with `workflow_dispatch` trigger — you can now publish from the GitHub Actions UI with a button click.

### New Experiments

**experiments/07_exchange_symmetry/**
Tests whether `combine()` produces boson-like (symmetric) or fermion-like (antisymmetric) results under constituent swap. Result: **exactly exchange-symmetric, every time, in both languages** (all four observables showed |Δ| = 0.0000 across 20 trials). The fermion-like hypothesis is negated — if exclusion behavior is desired, it must be deliberately built into the kernel definition, not hoped to emerge from union operations.

**experiments/08_observer_gated_combination/**
What if, instead of merging blindly, each agent first PERCEIVES the other? The resulting observer-gated combination produces three clean regimes: symmetric fit (identical twins), symmetric confusion (totally alien agents), and **real principled asymmetry** (veteran/novice on the same world: ~10% surprise gap, 90-95% order-independent dominance). The asymmetry was always there — it's just not inside `combine()`. It's in how one agent's history relates to another's, which is exactly what an observer measures.

**experiments/09_double_slit_analogy/**
Can this library reproduce double-slit interference? **No** — and the README now rigorously explains why (square-then-sum vs sum-then-square is one line of mathematics that the Markov/trie machinery structurally cannot cross). But the diamond world's own transition matrix has 8/10 complex eigenvalues with real oscillation periods — the same mathematical family as wave interference, just applied to a different question. The README was updated mid-sprint to replace a too-poetic "different algebras for different observers" framing with a more precise "one rule, different marginal statistics depending on correlation structure" framing that connects directly to decoherence theory and to experiment 08's results.

**experiments/10_hierarchical_realizer/**
Directional prototype of the emergent pattern discovery architecture: a population of agents generates a prediction-error response matrix; spectral clustering detects geometric structure in that matrix (2-4 clusters from 20 agents, no labels); detected clusters are crystallized into higher-level agents via `combine()`. The proof-of-concept works, but the specific multi-channel world was flawed (all agents saw the same full token regardless of channel assignment). The README outlines what a proper implementation needs: genuinely independent sensory channels, larger populations (100-1000+), and per-step prediction signals (not running averages).

### Documentation

- Both `.context/SKILL.md` files completely rewritten with honest "What This Library IS / IS NOT" distinction, 6 best-fit use cases with pros/cons, a generic verdict section, and a hybrid trading architecture section (not as a primary prediction engine, but as a decorrelated risk overlay)
- Both `.context/SESSION_CONTEXT.md` files updated with all new exports, test counts (49 Node, 41 Python), and new experiment listings
- Both READMEs expanded with all new API surface
- Site (`docs/`, `index.html`, `app.js`) synced with latest docs and corrected version numbers
- Site's Node.js code snippet fixed from wrong `CoinTossWorld({ nCoins: 4 })` syntax to correct `CoinTossWorld(4)`
- README for each new experiment (07-09) written with honest caveats, no overclaiming

### Repository Health

```
Node:  49 tests, 0 failures
Python: 41 tests, 0 failures
Exports: 17 new top-level symbols wired in both languages
Bug fixes: 6 critical, all regression-tested
Experiments added: 4 (07-10)
```

### What's Next (tracked in experiments/10 README)

The hierarchical realizer concept — using spectral geometry of collective agent dynamics to detect and crystallize emergent structure — is directionally validated but needs a proper world model with genuinely independent sensory channels per agent, larger populations, and per-step response metrics. The experiment 10 README includes 4 open questions and a component requirements table for the next session.
