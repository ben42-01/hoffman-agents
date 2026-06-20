"""
Stop Lights — Hoffman's Traffic Light Analogy

"A traffic light is just a deterministic Markov chain:
  RED → GREEN → YELLOW → RED
But each driver experiences it differently — and that
experience is what matters, not the objective transitions."

This example creates a traffic light world with 10 agents.
Each agent observes the light, builds internal models,
and develops tokens for "stop", "go", "slow".

A web UI at http://localhost:8765 shows:
  - The traffic light changing colors
  - Each agent's current output and I-lock status
  - Shared vocabulary emergence
"""
from conscious_agent import ConsciousAgent, WorldState
from conscious_agent import combine
from conscious_agent.io import save_agent
import asyncio
import json
import random
import http.server
import socketserver
import threading
import os


# ── Traffic Light World ────────────────────────────────────────

LIGHT_STATES = [
    {"id": 0, "label": "RED → GREEN", "color": "red", "next": 1, "tokens": ["stop", "halt", "wait"]},
    {"id": 1, "label": "GREEN → YELLOW", "color": "green", "next": 2, "tokens": ["go", "drive", "move"]},
    {"id": 2, "label": "YELLOW → RED", "color": "yellow", "next": 0, "tokens": ["slow", "caution", "brake"]},
]


class StopLightWorld:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self._current = 0
        self._ticks = 0

    def step(self):
        state = LIGHT_STATES[self._current]
        self._ticks += 1
        if self._ticks % 3 == 0:
            self._current = state["next"]
            state = LIGHT_STATES[self._current]
        return WorldState.from_sequence("world", [state["color"]]), {
            "id": state["id"],
            "color": state["color"],
            "label": state["label"],
            "tick": self._ticks,
        }

    @property
    def current(self):
        return LIGHT_STATES[self._current]


# ── Simulation ─────────────────────────────────────────────────

class StopLightSimulation:
    def __init__(self, n_agents=10):
        self.world = StopLightWorld(seed=42)
        self.agents = []
        for i in range(n_agents):
            agent = ConsciousAgent(agent_id=f"driver_{i:02d}")
            self.agents.append(agent)

        self.generation = 0
        self.history = []

    def step(self):
        ws, light_info = self.world.step()
        outputs = {}
        for agent in self.agents:
            out = agent.step(ws)
            outputs[agent.agent_id] = {
                "output": out.sequence_str,
                "i_locked": out.i_locked,
                "loop_depth": round(out.loop_depth, 3),
                "pred_error": round(out.prediction_error, 3),
                "vocab_size": agent.experience.lexicon.vocabulary_size(),
            }

        # Track shared vocabulary
        all_tokens = set()
        for a in self.agents:
            for e in a.experience.lexicon._entries.values():
                if e.output_token:
                    all_tokens.add(e.output_token)

        self.generation += 1
        snapshot = {
            "generation": self.generation,
            "light": light_info,
            "agents": outputs,
            "shared_vocab": list(all_tokens)[:10],
            "avg_pred_error": sum(o["pred_error"] for o in outputs.values()) / len(outputs),
        }
        self.history.append(snapshot)
        return snapshot


# ── HTTP Server ────────────────────────────────────────────────

simulation = StopLightSimulation(n_agents=10)
HTML = None


def load_html():
    global HTML
    html_path = os.path.join(os.path.dirname(__file__), "stop_lights.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            HTML = f.read()
    else:
        HTML = "<html><body><h1>Stop Lights Dashboard</h1><p>HTML file not found</p></body></html>"


class SimulationHandler(http.server.SimpleHTTPRequestHandler):
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
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    snapshot = simulation.step()
                    data = json.dumps(snapshot)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    import time
                    time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass


def run_server():
    load_html()
    port = 8765
    with socketserver.TCPServer(("", port), SimulationHandler) as httpd:
        print(f"\n  Stop Lights Dashboard: http://localhost:{port}")
        print(f"  Press Ctrl+C to stop\n")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
