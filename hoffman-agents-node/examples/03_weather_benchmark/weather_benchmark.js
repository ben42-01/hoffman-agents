/**
 * Weather Benchmark — Markov World vs Classical Baselines
 *
 * Build a Markov world from weather CSV data and compare prediction
 * accuracy against persistence, climatology, and AR(1).
 *
 * Usage:
 *   node weather_benchmark.js [path/to/weather_berlin.csv]
 */
const fs = require('fs');
const path = require('path');
const { WorldBuilder } = require('../../src/index');

function loadCSV(filePath) {
  const text = fs.readFileSync(filePath, 'utf8').trim();
  const lines = text.split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(',').map(v => v.trim());
    if (vals.length === headers.length) {
      const row = {};
      headers.forEach((h, j) => { row[h] = vals[j]; });
      rows.push(row);
    }
  }
  return rows;
}

function floatOrNull(v) { return v === '' || v === 'None' ? null : parseFloat(v); }

function computeTempDirection(temp, horizon = 3, threshold = 0.5) {
  return temp.map((t, i) => {
    const target = i + horizon;
    if (target >= temp.length || t === null || temp[target] === null) return null;
    const change = temp[target] - t;
    if (Math.abs(change) <= threshold) return 'same';
    return change > 0 ? 'up' : 'down';
  });
}

function run(csvPath) {
  console.log('='.repeat(60));
  console.log('Weather Benchmark: Markov World vs Classical Baselines');
  console.log('='.repeat(60));

  if (!fs.existsSync(csvPath)) {
    console.error(`File not found: ${csvPath}`);
    console.error('Download from: https://github.com/anomalyco/conscious-agents/blob/main/data/weather_berlin.csv');
    process.exit(1);
  }

  const rows = loadCSV(csvPath);
  console.log(`\nLoaded ${rows.length} weather records`);

  const temp = rows.map(r => floatOrNull(r.temperature_2m));
  const pressure = rows.map(r => floatOrNull(r.surface_pressure));
  const humidity = rows.map(r => floatOrNull(r.relative_humidity_2m));
  const wind = rows.map(r => floatOrNull(r.wind_speed_10m));

  // Build data array (skip rows with nulls)
  const valid = [];
  for (let i = 0; i < rows.length; i++) {
    if (temp[i] !== null && pressure[i] !== null && humidity[i] !== null && wind[i] !== null) {
      valid.push([temp[i], pressure[i], humidity[i], wind[i]]);
    }
  }
  const n = valid.length;

  if (n < 100) {
    console.error('Too few valid rows after filtering');
    process.exit(1);
  }

  console.log(`Building Markov world...`);
  const world = new WorldBuilder()
    .addFeature('temp', 'minmax', 4)
    .addFeature('pressure', 'minmax', 4)
    .addFeature('humidity', 'minmax', 4)
    .addFeature('wind', 'minmax', 3)
    .build(valid);
  console.log(`  ${world.nStates} states`);

  const stateIds = world.stateIds;
  const dirs = computeTempDirection(temp);

  // Filter valid indices
  const validIdx = [];
  for (let i = 0; i < n; i++) {
    if (dirs[i] !== null) validIdx.push(i);
  }

  const split = Math.floor(validIdx.length * 0.8);
  const trainIdx = validIdx.slice(0, split);
  const testIdx = validIdx.slice(split);

  // Build direction map from training data
  const dirMap = {};
  for (const i of trainIdx) {
    const sid = stateIds[i], d = dirs[i];
    if (d === null) continue;
    if (!dirMap[sid]) dirMap[sid] = { up: 0, down: 0, same: 0 };
    dirMap[sid][d]++;
  }

  function predictDir(sid) {
    const counts = dirMap[sid];
    if (!counts) return null;
    let best = 'up', bc = 0;
    for (const [d, c] of Object.entries(counts)) { if (c > bc) { best = d; bc = c; } }
    return best;
  }

  // Markov evaluation
  let mCor = 0, mTot = 0;
  for (const i of testIdx) {
    const pred = predictDir(stateIds[i]);
    if (pred) { mTot++; if (pred === dirs[i]) mCor++; }
  }
  const markovAcc = mTot > 0 ? mCor / mTot : 0;

  // Majority class
  const trainCounts = { up: 0, down: 0, same: 0 };
  for (const i of trainIdx) { if (dirs[i]) trainCounts[dirs[i]]++; }
  const majority = Object.entries(trainCounts).sort((a, b) => b[1] - a[1])[0][0];
  const majorityAcc = trainCounts[majority] / Object.values(trainCounts).reduce((a, b) => a + b, 0);

  // Persistence
  let pCor = 0, pTot = 0;
  for (const i of testIdx) {
    const past = i - 6;
    if (past < 0) continue;
    const change = temp[i] - temp[past];
    const pred = Math.abs(change) <= 0.5 ? 'same' : change > 0 ? 'up' : 'down';
    pTot++; if (pred === dirs[i]) pCor++;
  }
  const persAcc = pTot > 0 ? pCor / pTot : 0;

  // AR(1)
  const arPairs = [];
  for (const i of trainIdx) {
    if (temp[i] !== null && temp[i + 3] !== null) arPairs.push([temp[i], temp[i + 3]]);
  }
  let arAcc = 0;
  if (arPairs.length > 10) {
    const xs = arPairs.map(p => p[0]), ys = arPairs.map(p => p[1]);
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    let num = 0, den = 0;
    for (let k = 0; k < xs.length; k++) { num += (xs[k] - mx) * (ys[k] - my); den += (xs[k] - mx) ** 2; }
    const a = den > 1e-12 ? num / den : 0;
    const b = my - a * mx;
    let arCor = 0, arTot = 0;
    for (const i of testIdx) {
      const predicted = a * temp[i] + b;
      const change = predicted - temp[i];
      const pred = Math.abs(change) <= 0.5 ? 'same' : change > 0 ? 'up' : 'down';
      arTot++; if (pred === dirs[i]) arCor++;
    }
    arAcc = arTot > 0 ? arCor / arTot : 0;
  }

  console.log(`\nBaselines:`);
  console.log(`  Majority class ('${majority}'):       ${(majorityAcc*100).toFixed(1)}%`);
  console.log(`  Persistence (6h):            ${(persAcc*100).toFixed(1)}%`);
  console.log(`  AR(1):                       ${(arAcc*100).toFixed(1)}%`);
  console.log(`  Markov World:                ${(markovAcc*100).toFixed(1)}%`);
  console.log(`  Improvement over majority:   ${(markovAcc/majorityAcc*100-100).toFixed(1)}%`);

  if (markovAcc > Math.max(majorityAcc, persAcc, arAcc)) {
    console.log(`\n  ✓ Markov world beats classical baselines`);
  }
}

const csvPath = process.argv[2] || path.join(__dirname, 'weather_berlin.csv');
run(csvPath);
