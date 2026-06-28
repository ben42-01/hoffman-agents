## v2.0.1 — Self-Awareness & Agent Skill Documentation

### New Features

**SelfWorld — Explicit Self-Perception (Node.js + Python)**
- New `SelfWorld` world wrapper that injects the agent's own internal metrics into its perception stream
- The agent's trie learns transitions over composite states of `(world_data + self_state)`
- Fully customizable via `getStateFn` callback
- Default metrics: stationary probability, prediction error, I-lock status, loop score, meta-trie size, vocabulary size, trie node count
- Documentation: `docs/SELF_AWARENESS.md`

### Documentation

- **`docs/SELF_AWARENESS.md`** — New feature document covering all four self-awareness mechanisms (MetaTrie, SelfTokenState, strangeLoopScore, SelfWorld)
- **SKILL.md updates** — Both Node and Python `.context/SKILL.md` files expanded with patterns for complex scenarios: multi-agent networks, WorldBuilder from real data, lifecycle management, crystal projection, debugging non-locking agents, live data feeding, confidence-based decisions
- **README updates** — Self-Awareness section added; SKILL.md discovery notice at top for AI coding tools
- **PyPI repo URL** — Fixed `pyproject.toml` from `anomalyco/conscious-agents` to `ben42-01/hoffman-agents`

### Packaging

- `.context/` directories included in npm package and PyPI source distribution (`MANIFEST.in`)
- `conscious-agent-react-native` updated with SelfWorld support

### Test Results

- Node.js: 36/36 passing
- Python: 36/36 passing
- React Native: 10/10 passing
