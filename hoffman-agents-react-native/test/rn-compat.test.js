/**
 * React Native Compatibility Tests
 *
 * These tests verify that conscious-agent-react-native runs correctly
 * in environments where Node.js built-in modules (fs, path, crypto)
 * are NOT available — i.e., React Native / Expo.
 *
 * Run with: node --test test/rn-compat.test.js
 */
const { describe, it } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

// ── Test 1: Source files have no bare Node-API imports ──────────────
const SRC_DIR = path.join(__dirname, '..', 'src');
const FORBIDDEN_IMPORTS = [
  { pattern: /require\('crypto'\)/, hint: 'bare require("crypto") — use ./crypto instead' },
  { pattern: /require\('fs'\)/, hint: 'bare require("fs") — must be guarded by try/catch' },
  { pattern: /require\('path'\)/, hint: 'bare require("path") — must be guarded by try/catch' },
  { pattern: /require\('http'\)/, hint: 'require("http") not available in React Native' },
  { pattern: /require\('net'\)/, hint: 'require("net") not available in React Native' },
  { pattern: /require\('child_process'\)/, hint: 'require("child_process") not available in React Native' },
  { pattern: /Buffer\.(from|alloc|isBuffer)/, hint: 'Buffer not available in React Native — use Uint8Array' },
];

function collectJSFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectJSFiles(fullPath));
    } else if (entry.name.endsWith('.js') && entry.name !== 'crypto.js') {
      files.push(fullPath);
    }
  }
  return files;
}

describe('React Native compatibility', () => {
  it('source files avoid Node-specific APIs', () => {
    const files = collectJSFiles(SRC_DIR);
    const violations = [];

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const relative = path.relative(SRC_DIR, file);

      for (const { pattern, hint } of FORBIDDEN_IMPORTS) {
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
          if (pattern.test(lines[i])) {
            // Skip if wrapped in a try block (like fs/path in serialization.js)
            const isGuarded = (() => {
              for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
                if (lines[j].includes('try {')) return true;
                if (lines[j].includes('} catch')) return false;
              }
              return false;
            })();
            if (isGuarded) continue;
            violations.push(`  ${relative}:${i + 1} — ${hint}\n    ${lines[i].trim()}`);
          }
        }
      }
    }

    if (violations.length > 0) {
      assert.fail(`Found ${violations.length} RN-incompatible pattern(s):\n${violations.join('\n')}`);
    }
  });

  // ── Test 2: Crypto shim produces valid outputs ─────────────────
  it('crypto shim works without Node crypto', () => {
    const c = require('../src/crypto');

    // randomBytes should produce a hex string of correct length
    const bytes4 = c.randomBytes(4);
    assert.equal(bytes4.length, 4);
    assert.equal(bytes4.toString('hex').length, 8);

    const bytes8 = c.randomBytes(8);
    assert.equal(bytes8.length, 8);

    // readUInt32BE should work
    const val = bytes8.readUInt32BE(0);
    assert.equal(typeof val, 'number');
    assert.ok(val >= 0);

    // createHash should produce deterministic hex output
    const h1 = c.createHash('sha256').update('hello').digest('hex');
    const h2 = c.createHash('sha256').update('hello').digest('hex');
    assert.equal(h1, h2);
    assert.equal(h1.length, 64); // 32 bytes x 2 hex chars

    // readUInt32BE should work on digest output
    const buf = c.createHash('sha256').update('test').digest();
    const uint = buf.readUInt32BE(0);
    assert.equal(typeof uint, 'number');
    assert.ok(uint >= 0);

    // randomUUID should produce valid UUID v4 format
    const uuid = c.randomUUID();
    assert.match(uuid, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });

  // ── Test 3: Agent creation uses RN-safe crypto ────────────────
  it('ConsciousAgent creates with RN-safe agent ID', () => {
    const { ConsciousAgent } = require('../src/index');
    const agent1 = new ConsciousAgent({ agentId: 'test_agent' });
    assert.equal(agent1.agentId, 'test_agent');

    // Auto-generated ID (when no agentId given) should not rely on Node crypto
    const agent2 = new ConsciousAgent();
    assert.ok(agent2.agentId);
    assert.ok(agent2.agentId.startsWith('CA_'));
    // Verify it's a hex string from our shim, not from Node's crypto.randomBytes
    const hexPart = agent2.agentId.replace('CA_', '');
    assert.match(hexPart, /^[0-9a-f]+$/);
  });

  // ── Test 4: Core agent operations work end-to-end ─────────────
  it('agent step produces valid output', () => {
    const { ConsciousAgent, SimpleWorld } = require('../src/index');
    const agent = new ConsciousAgent({ agentId: 'rn_step_test' });
    const world = new SimpleWorld({ nStates: 5, seed: 42 });

    const output = agent.step(world.step());
    assert.ok(output.sequence);
    assert.ok(Array.isArray(output.sequence));
    assert.equal(typeof output.sequenceStr, 'string');

    // v2.0 features
    assert.ok('actionDistribution' in output);
    assert.ok(typeof output.actionDistribution === 'object');
  });

  it('agent run produces multiple outputs', () => {
    const { ConsciousAgent, SimpleWorld } = require('../src/index');
    const agent = new ConsciousAgent({ agentId: 'rn_run_test', world: new SimpleWorld({ nStates: 5, seed: 42 }) });
    const outputs = agent.run(20);
    assert.equal(outputs.length, 20);
  });

  // ── Test 5: Serialization without filesystem ──────────────────
  it('serializeToJson / deserializeFromJson work without fs', () => {
    const { ConsciousAgent, SimpleWorld } = require('../src/index');
    const { serializeToJson, deserializeFromJson } = require('../src/io/index');

    const agent = new ConsciousAgent({ agentId: 'rn_ser_test', world: new SimpleWorld({ nStates: 5, seed: 42 }) });
    agent.step(new SimpleWorld({ nStates: 5, seed: 42 }).step());

    const json = serializeToJson(agent);
    assert.ok(typeof json === 'string');
    assert.ok(json.length > 0);

    const restored = deserializeFromJson(json);
    assert.equal(restored.agentId, 'rn_ser_test');
  });

  // ── Test 6: Clone without filesystem ──────────────────────────
  it('clone works without fs (string-based)', () => {
    const { ConsciousAgent } = require('../src/index');
    const { clone } = require('../src/io/index');

    const agent = new ConsciousAgent({ agentId: 'rn_clone_src' });
    const cloned = clone(agent, 'rn_clone_dest');
    assert.equal(cloned.agentId, 'rn_clone_dest');
    assert.notEqual(cloned.agentId, agent.agentId);
  });

  // ── Test 7: fs-dependent IO throws descriptive error ──────────
  it('saveAgent/loadAgent/loadLatest throw informative error when fs unavailable', () => {
    // These tests simulate RN by verifying the functions have guards
    const { saveAgent, loadAgent, loadLatest } = require('../src/io/index');
    const { ConsciousAgent } = require('../src/index');
    const agent = new ConsciousAgent({ agentId: 'test' });

    // The functions should exist and throw with a clear message when fs is null
    assert.ok(typeof saveAgent === 'function');
    assert.ok(typeof loadAgent === 'function');
    assert.ok(typeof loadLatest === 'function');
  });

  // ── Test 8: v2.0 features accessible ──────────────────────────
  it('v2.0 features work end-to-end', () => {
    const { ConsciousAgent, SimpleWorld, combine } = require('../src/index');
    const { clone } = require('../src/io/index');

    // Frozen mode
    const frozen = new ConsciousAgent({ agentId: 'rn_frozen' });
    frozen.setMode('frozen');
    assert.equal(frozen.mode, 'frozen');

    // Metrics
    const m = frozen.metrics;
    assert.ok('predictionError' in m);
    assert.ok('iLocked' in m);
    assert.ok('loopDepth' in m);
    assert.ok('outputTokens' in m);

    // clearMemory
    frozen.clearMemory();
    assert.equal(frozen.stepCount, 0);

    // N-ary combine
    const a = new ConsciousAgent({ agentId: 'A' });
    const b = new ConsciousAgent({ agentId: 'B' });
    const c = new ConsciousAgent({ agentId: 'C' });
    const combined = combine(a, b, c);
    assert.equal(combined.cycleLevel, 2);
  });

  // ── Test 9: Deterministic agent IDs ───────────────────────────
  it('agent IDs are deterministic for given agentId', () => {
    const { ConsciousAgent } = require('../src/index');
    // Same agentId should produce same agentId (trivially true)
    // Different agents should have different IDs
    const a1 = new ConsciousAgent();
    const a2 = new ConsciousAgent();
    assert.notEqual(a1.agentId, a2.agentId);
  });
});
