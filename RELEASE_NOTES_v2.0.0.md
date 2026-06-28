## v2.0.0 — The Integration Release

### What's New

**Agent Lifecycle**
- `setMode('frozen')` / `set_mode("frozen")` — deterministic projection mode (no learning, no self-model drift)
- `thaw()` / `refreeze()` — switch between frozen and learning modes at runtime
- `clearMemory()` / `clear_memory()` — reset trace buffer and step count while preserving learned trie structure

**AgentNetwork**
- `stepAll(worldState)` / `step_all(world_state)` — batch-step all agents with the same world state, bypassing topology
- `getMetrics()` / `get_metrics()` — aggregate metrics across all agents (mean PE, variance, lock rate, loop depth)
- `getAgentMetrics(id)` / `get_agent_metrics(id)` — per-agent metrics snapshot from the network
- `agentList` / `agent_list` — get agents as an ordered array

**Action Space**
- Full `actionDistribution` / `action_distribution` in every StepOutput — probability map over output tokens
- `allowableTokens` / `allowable_tokens` — constrain agent output to a permitted token set at construction or runtime

**Composition**
- N-ary `combine(...agents)` — combine 3+ agents into a balanced tree with proper cycle_level semantics
- Weighted combination — `combine(a, b, { weights: [0.7, 0.3] })` for asymmetric authority

**World & Perception**
- `injectObservation(worldState)` / `inject_observation(world_state)` — push new observations mid-run without reconstruction or reset
- `TraceBuffer.resize(newSize)` / `trace_buffer.resize(new_maxlen)` — dynamic window resizing, preserves newest entries

**Trie Introspection**
- `trie.getStats()` / `trie.get_stats()` — node count, max depth, mean visit count, depth distribution
- `trie.exportNodes(minVisits)` / `trie.export_nodes(min_visits)` — export all paths with visit stats
- `trie.getDominantPaths(topK)` / `trie.get_dominant_paths(top_k)` — top K most-visited paths

**Determinism**
- Seeded per-agent RNGs — agents created via `AgentNetwork` now get deterministic RNG derived from network seed
- Agent `decide()` accepts optional `rng` parameter for reproducible output sequences

### New Package

- `conscious-agent-react-native` — React Native / Expo adapter with pure-JS crypto shim and platform-adaptive IO

### Breaking Changes

- `StepOutput` now includes `actionDistribution` field (empty dict `{}` when not computed)
- Agents created without an explicit `agentId` now generate IDs using Math.random-based hex (was crypto.randomBytes)

### Full Changelog

- 16+ new features across both Node.js and Python libraries
- 33 tests per library, 10 RN compatibility tests
- Version bumps: npm 1.0.1 → 2.0.0, pip 1.0.0 → 2.0.0
- All examples verified working
