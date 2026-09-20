const { describe, it } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { compare } = require('./parity/compare');

const pyRunner = path.join(__dirname, '..', '..', 'hoffman-agents-python', 'tests', 'parity', 'run.py');
const venvPython = path.join(__dirname, '..', '..', 'hoffman-agents-python', '.venv', 'bin', 'python');
const python = process.env.PYTHON || (fs.existsSync(venvPython) ? venvPython : null);

describe('Node <-> Python parity (v3)', () => {
  it('produces identical agent dynamics, lock events, combination and fusion', { skip: !python || !fs.existsSync(pyRunner) ? 'python library not available (set PYTHON)' : false, timeout: 300000 }, () => {
    const node = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, 'parity', 'run.js')], { maxBuffer: 1 << 28 }));
    const py = JSON.parse(execFileSync(python, [pyRunner], { maxBuffer: 1 << 28 }));
    const diffs = compare(node, py);
    assert.deepEqual(diffs, []);
    assert.ok(node.home_1.lockHistory.length > 0, 'scenario exercises the lock');
  });
});
