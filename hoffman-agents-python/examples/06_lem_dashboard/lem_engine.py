"""
LEM Dashboard — Large Experience Model (lighter version)

5 ConsciousAgents process a synthetic emotional stream and observe
each other in a mesh network. No LLM, no SHA hashing in hot path,
no external data. Slower ticks to stay cool on laptops.
"""
from conscious_agent import ConsciousAgent, WorldState, ExperienceSpace, SelfTokenState
import http.server
import json
import os
import random
import socketserver
import time


QUADRANTS = ["fear", "desire", "fatigue", "joy"]

QUAD_COLORS = {
    "fear":    "#da3633",
    "desire":  "#d29922",
    "fatigue": "#6e7681",
    "joy":     "#3fb950",
}

# Pre-hashed state IDs for each quadrant (computed once, not per step)
_PRECHASHED = {q: WorldState.from_sequence("bluesky", [q]).get_state_id() for q in QUADRANTS}

TRANSITIONS = {
    "fear":    ["fear", "desire", "fatigue"],
    "desire":  ["desire", "joy", "fear"],
    "fatigue": ["fatigue", "fear", "joy"],
    "joy":     ["joy", "desire", "fatigue"],
}

AGENT_PROFILES = [
    ("seer",   5,  0.80, 0.08, 1.0),
    ("sage",   20, 0.80, 0.05, 1.0),
    ("anchor", 50, 0.95, 0.02, 0.5),
    ("scout",  10, 0.60, 0.20, 1.2),
    ("mirror", 15, 0.85, 0.05, 1.0),
]


class LEMEngine:
    def __init__(self):
        self._step = 0
        self._agents = {}
        self._quadrant = "joy"
        self._collapse_count = 0
        self._last_collapse_step = 0
        self._cooldown = 30
        self._collapse_history = []
        self._rng = random.Random(42)
        self._route_log = {q: 0 for q in QUADRANTS}
        self._ws_cache = {}  # avoid WorldState.from_sequence hot-loop hashing

        for aid, mi, ps, pe, et in AGENT_PROFILES:
            st = SelfTokenState(lock_threshold=0.20, lock_consecutive_required=2)
            self._agents[aid] = ConsciousAgent(
                agent_id=aid,
                experience=ExperienceSpace(self_token=st),
                meta_observation_interval=mi,
                p_stable=ps, p_explore=pe, expression_temp=et,
            )

    def _ws(self, key):
        """Cached WorldState builder — no re-hashing."""
        if key not in self._ws_cache:
            self._ws_cache[key] = WorldState.from_sequence("bluesky", [key])
        return self._ws_cache[key]

    def step(self):
        self._step += 1

        # Emotional state machine
        self._quadrant = self._rng.choice(TRANSITIONS[self._quadrant])
        self._route_log[self._quadrant] += 1
        world_id = _PRECHASHED[self._quadrant]

        # Phase 1: each agent observes the world (single step)
        for agent in self._agents.values():
            agent.step(self._ws(self._quadrant))

        # Phase 2: agents observe their neighbors' outputs (mesh topology)
        outputs = {aid: a.get_output() for aid, a in self._agents.items()}
        agent_names = list(self._agents.keys())
        topology = []
        for i, aid in enumerate(agent_names):
            agent = self._agents[aid]
            n1 = agent_names[(i + 1) % 5]
            n2 = agent_names[(i + 3) % 5]
            ws = WorldState(sequences={
                n1: outputs[n1],
                n2: outputs[n2],
            })
            agent.step(ws)
            # Connection strength based on prediction error (lower = stronger)
            pe = agent.experience.trace_buffer.prediction_error_mean(5)
            strength = round(max(0, 1.0 - pe), 3)
            topology.append({"source": aid, "target": n1, "strength": strength})
            topology.append({"source": aid, "target": n2, "strength": strength})

        # Agent states and predictions
        agent_states = {}
        predictions = {q: 0 for q in QUADRANTS}
        locked_count = 0
        for aid, agent in self._agents.items():
            pe = agent.experience.trace_buffer.prediction_error_mean(5)
            prev = agent.experience.last_world_state_id
            pred_id = agent.experience.trie.predict_next([prev]) if prev is not None else None
            for q, sid in _PRECHASHED.items():
                if pred_id == sid:
                    predictions[q] += 1
            if agent.experience.self_token.locked:
                locked_count += 1
            agent_states[aid] = {
                "prediction_error": round(pe, 3),
                "i_locked": agent.experience.self_token.locked,
                "loop_depth": round(agent.loop_score, 2),
                "last_output": agent._last_output,
            }

        locked_ratio = locked_count / 5
        dominant = max(predictions, key=predictions.get) if any(predictions.values()) else self._quadrant
        consensus = predictions.get(dominant, 0) / 5

        # ⊕ collapse detection
        collapse = None
        if (self._step > 30
            and self._step - self._last_collapse_step > self._cooldown
            and consensus >= 0.6
            and locked_ratio >= 0.4):
            pe_sum = sum(a.experience.trace_buffer.prediction_error_mean(1) for a in self._agents.values())
            if pe_sum / 5 >= 0.5:
                self._collapse_count += 1
                self._last_collapse_step = self._step
                collapse = {
                    "step": self._step,
                    "expected": dominant,
                    "actual": self._quadrant,
                    "consensus": consensus,
                    "locked_ratio": locked_ratio,
                }
                self._collapse_history.append(collapse)

        return {
            "step": self._step,
            "quadrant": self._quadrant,
            "color": QUAD_COLORS[self._quadrant],
            "agents": agent_states,
            "predictions": predictions,
            "consensus": consensus,
            "collapse_count": self._collapse_count,
            "collapse": collapse,
            "collapse_history": list(self._collapse_history[-5:]),
            "locked_ratio": locked_ratio,
            "locked_count": locked_count,
            "topology": topology,
            "route_log": dict(self._route_log),
        }


# ── Server ─────────────────────────────────────────────────────

engine = LEMEngine()
HTML = None


def warmup(engine, steps=200):
    """Fast warm-up: run N steps before anyone connects so agents have
    meta-tries, I-locks, and topology history ready to display."""
    for _ in range(steps):
        engine.step()
    print(f"  Warm-up complete: {engine._step} steps, {engine._collapse_count} ⊕ collapses, "
          f"{engine._step - engine._last_collapse_step} since last")


def load_html():
    global HTML
    p = os.path.join(os.path.dirname(__file__), "lem_dashboard.html")
    if os.path.exists(p):
        with open(p) as f:
            HTML = f.read()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    data = json.dumps(engine.step())
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(2)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def run():
    load_html()
    warmup(engine, steps=200)
    port = 8765
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"\n  LEM Dashboard: http://localhost:{port}")
        print(f"  Agents warmed up — I-locks, topology, and ⊕ collapses ready to view")
        print(f"  Updates every 2s — close browser tab to stop CPU")
        print(f"  No LLM calls, no external data\n")
        httpd.serve_forever()


if __name__ == "__main__":
    run()
