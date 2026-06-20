/**
 * Stop Lights — Hoffman's Traffic Light Analogy (Node.js + Web UI)
 *
 * A traffic light cycles RED → GREEN → YELLOW → RED.
 * 10 agents observe the light, build internal models, and develop
 * shared vocabulary for "stop", "go", "slow".
 *
 * Dashboard: http://localhost:8765
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const { ConsciousAgent, WorldState } = require('../../src/index');

// ── Traffic Light State Machine ────────────────────────────────

const LIGHT_STATES = [
  { id: 0, label: 'RED → GREEN', color: 'red', next: 1 },
  { id: 1, label: 'GREEN → YELLOW', color: 'green', next: 2 },
  { id: 2, label: 'YELLOW → RED', color: 'yellow', next: 0 },
];

class StopLightWorld {
  constructor(seed = 42) {
    this._current = 0;
    this._ticks = 0;
  }

  step() {
    const state = LIGHT_STATES[this._current];
    this._ticks++;
    if (this._ticks % 3 === 0) {
      this._current = state.next;
    }
    const s = LIGHT_STATES[this._current];
    return {
      ws: WorldState.fromSequence('world', [s.color]),
      info: { id: s.id, color: s.color, label: s.label, tick: this._ticks },
    };
  }
}

// ── Simulation ─────────────────────────────────────────────────

class Simulation {
  constructor(nAgents = 10) {
    this.world = new StopLightWorld();
    this.agents = Array.from({ length: nAgents }, (_, i) =>
      new ConsciousAgent({ agentId: `driver_${String(i).padStart(2, '0')}` })
    );
    this.generation = 0;
  }

  step() {
    const { ws, info } = this.world.step();
    const outputs = {};
    for (const agent of this.agents) {
      const out = agent.step(ws);
      outputs[agent.agentId] = {
        output: out.sequenceStr,
        i_locked: out.iLocked,
        loop_depth: out.loopDepth,
        pred_error: out.predictionError,
        vocab_size: agent.experience.lexicon.vocabularySize(),
      };
    }

    // Shared vocabulary
    const allTokens = new Set();
    for (const a of this.agents) {
      for (const entry of a.experience.lexicon._entries.values()) {
        if (entry.outputToken) allTokens.add(entry.outputToken);
      }
    }

    this.generation++;
    const avgErr = Object.values(outputs).reduce((s, a) => s + a.pred_error, 0) / Object.keys(outputs).length;
    return {
      generation: this.generation,
      light: info,
      agents: outputs,
      shared_vocab: [...allTokens].slice(0, 10),
      avg_pred_error: avgErr,
    };
  }
}

// ── Server ─────────────────────────────────────────────────────

const sim = new Simulation(10);
const htmlPath = path.join(__dirname, 'stop_lights.html');
let html = '';

try { html = fs.readFileSync(htmlPath, 'utf8'); } catch { html = '<html><body><h1>HTML not found</h1></body></html>'; }

const server = http.createServer((req, res) => {
  if (req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
  } else if (req.url === '/stream') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const interval = setInterval(() => {
      const data = JSON.stringify(sim.step());
      res.write(`data: ${data}\n\n`);
    }, 500);

    req.on('close', () => { clearInterval(interval); });
  } else {
    res.writeHead(404);
    res.end();
  }
});

const PORT = 8765;
server.listen(PORT, () => {
  console.log(`\n  Stop Lights Dashboard: http://localhost:${PORT}`);
  console.log(`  Press Ctrl+C to stop\n`);
});
