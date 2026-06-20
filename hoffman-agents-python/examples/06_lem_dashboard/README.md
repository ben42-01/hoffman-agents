# LEM Dashboard — Large Experience Model

5 **ConsciousAgents** (seer, sage, anchor, scout, mirror) living inside an emotional (VAD) world. Each has a different time horizon and learns the grammar of emotional transitions.

They observe the world **and** each other — forming a mesh network. When all 5 converge on the same prediction and get surprised at once, a **⊕ collapse** fires. That's the mesh achieving collective self-awareness.

```
http://localhost:8765
```

## What to look for

| Thing | What it means |
|-------|---------------|
| **3D scatter** | Agents' positions in Valence–Arousal–Dominance space. The active quadrant pulses. |
| **Topology graph** | Who watches whom. Thicker edge = lower prediction error = stronger connection. |
| **Quadrant bars** | What each agent predicts will happen next. All agree = consensus. |
| **⊕ collapse** | All 5 agents predicted X, but Y happened — unanimous surprise. Collective learning moment. |
| **Agent cards** | Per-agent: I-lock (identity formed?), prediction error, loop depth (self-awareness), vocabulary. |

## Run

```bash
uv run python examples/06_lem_dashboard/lem_engine.py
```

No external data needed — it generates synthetic emotional streams.
