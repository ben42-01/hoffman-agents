#!/usr/bin/env node
/**
 * Color Fusion — Full-RGB Perceptual Domain Model
 *
 * The world spans the ENTIRE RGB cube. Each agent has a perceptual domain —
 * a region of RGB space it can perceive. Colors outside that domain are
 * outside its interface: the agent disregards them (high prediction error,
 * no learning).
 *
 * combine() merges domains. fuse() splits them back.
 * The dashboard shows each agent's perceived color changing as the world
 * moves through the spectrum and the agent's prediction error fluctuates.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { ConsciousAgent, WorldState, combine, fuse } = require('../../src/index');

const PORT = 5000;
const ROOT = __dirname;

/* ─── Perceptual domains — each agent accepts a region of RGB space ─── */

// Fractional RGB scores: a channel's contribution relative to total brightness
function scores(r, g, b) {
  const t = r + g + b + 1;
  return { rs: r / t, gs: g / t, bs: b / t };
}

const DOMAINS = {
  Red:   { label: 'Red',   hex: '#FF0000', test: (r,g,b) => { const s=scores(r,g,b); return s.rs>0.45 && s.gs<0.3 && s.bs<0.3; }},
  Green: { label: 'Green', hex: '#00FF00', test: (r,g,b) => { const s=scores(r,g,b); return s.gs>0.45 && s.rs<0.3 && s.bs<0.3; }},
  Blue:  { label: 'Blue',  hex: '#0000FF', test: (r,g,b) => { const s=scores(r,g,b); return s.bs>0.45 && s.rs<0.3 && s.gs<0.3; }},
  UV:    { label: 'UV',    hex: '#8800FF', test: (r,g,b) => { const s=scores(r,g,b); return s.bs>0.35 && s.rs>s.gs && s.bs>s.rs; }},
  IR:    { label: 'IR',    hex: '#FF0088', test: (r,g,b) => { const s=scores(r,g,b); return s.rs>0.35 && s.bs>s.gs && s.rs>s.bs; }},
};

const DOMAIN_KEYS = Object.keys(DOMAINS);

/* ─── Generate a random RGB color that falls within a domain ─── */

function randomInDomain(domainKey, rng) {
  const d = DOMAINS[domainKey];
  for (let attempt = 0; attempt < 200; attempt++) {
    // Bias the distribution toward the domain's dominant channel
    let r, g, b;
    if (domainKey === 'Red')   { r = 100 + rng() * 155; g = rng() * 80;  b = rng() * 80; }
    else if (domainKey === 'Green') { g = 100 + rng() * 155; r = rng() * 80;  b = rng() * 80; }
    else if (domainKey === 'Blue')  { b = 100 + rng() * 155; r = rng() * 80;  g = rng() * 80; }
    else if (domainKey === 'UV')    { b = 120 + rng() * 135; r = 40 + rng() * 100; g = rng() * 60; }
    else if (domainKey === 'IR')    { r = 120 + rng() * 135; b = 40 + rng() * 100; g = rng() * 60; }
    if (d.test(r, g, b)) return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
  }
  return { r: 128, g: 128, b: 128 };
}

function rgbToHex(r, g, b) {
  return `#${Math.min(255,Math.max(0,Math.round(r))).toString(16).padStart(2,'0')}${Math.min(255,Math.max(0,Math.round(g))).toString(16).padStart(2,'0')}${Math.min(255,Math.max(0,Math.round(b))).toString(16).padStart(2,'0')}`;
}

/* ─── ColorWorld: Markov chain across the full RGB space ─── */

class ColorWorld {
  constructor(seed = 42) {
    this._rng = this._seedRandom(seed);
    this._currentDomain = 0; // index into DOMAIN_KEYS
    this._currentRGB = randomInDomain(DOMAIN_KEYS[this._currentDomain], this._rng);
  }
  _seedRandom(s) { let seed = s; return () => { seed = (seed * 1664525 + 1013904223) & 0x7FFFFFFF; return seed / 0x7FFFFFFF; }; }

  step() {
    // Markov transition between domains (spectral adjacency)
    if (this._rng() < 0.3) {
      // Shift to adjacent domain
      const delta = this._rng() < 0.5 ? -1 : 1;
      this._currentDomain = (this._currentDomain + delta + DOMAIN_KEYS.length) % DOMAIN_KEYS.length;
    }
    // Generate a new RGB color within the current domain
    this._currentRGB = randomInDomain(DOMAIN_KEYS[this._currentDomain], this._rng);
    const hex = rgbToHex(this._currentRGB.r, this._currentRGB.g, this._currentRGB.b);
    return {
      ws: WorldState.fromSequence('world', [hex]),
      rgb: this._currentRGB,
      hex,
      domain: DOMAIN_KEYS[this._currentDomain],
    };
  }
}

/* ─── Agent definitions ─── */

const AGENTS = [
  { id: 'Red',     domain: 'Red',   level: 0 },
  { id: 'Green',   domain: 'Green', level: 0 },
  { id: 'Blue',    domain: 'Blue',  level: 0 },
  { id: 'UV',      domain: 'UV',    level: 0 },
  { id: 'IR',      domain: 'IR',    level: 0 },
];

const FUSIONS = [
  { id: 'Yellow',   level: 1, parents: ['Red','Green'] },
  { id: 'Magenta',  level: 1, parents: ['Red','Blue'] },
  { id: 'Cyan',     level: 1, parents: ['Green','Blue'] },
  { id: 'UV+Red',   level: 1, parents: ['UV','Red'] },
  { id: 'IR+Green', level: 1, parents: ['IR','Green'] },
  { id: 'White',    level: 2, parents: ['Yellow','Blue'] },
  { id: 'Omni',     level: 3, parents: ['White','UV','IR'] },
];

/* ─── State ─── */

const world = new ColorWorld(42);
const agents = {};  // { id: ConsciousAgent }
const fused = {};   // { id: ConsciousAgent }
const reclaimed = {};
let step = 0;

/* ─── Train primaries ─── */

function trainPrimaries() {
  for (const def of AGENTS) {
    const d = DOMAINS[def.domain];
    const a = new ConsciousAgent({ agentId: def.id, metaObservationInterval: 10 });
    for (let i = 0; i < 600; i++) {
      const { ws, rgb } = world.step();
      // Only learn from colors within the agent's perceptual domain
      if (d.test(rgb.r, rgb.g, rgb.b)) a.step(ws);
    }
    agents[def.id] = a;
  }
}

/* ─── Build fusions ─── */

function buildFusions() {
  const resolve = name => fused[name] || agents[name];
  for (const def of FUSIONS) {
    const parents = def.parents.map(resolve).filter(Boolean);
    if (parents.length < 2) continue;
    let c = parents[0];
    for (let i = 1; i < parents.length; i++) c = combine(c, parents[i]);
    if (c) fused[def.id] = c;
  }
  // Fuse Yellow — reclaimants carry forward shared experience
  if (fused.Yellow) {
    const parts = fuse(fused.Yellow);
    reclaimed.Red_reclaimed = { agent: parts[0], parentage: 'from Yellow (fused)' };
    reclaimed.Green_reclaimed = { agent: parts[1], parentage: 'from Yellow (fused)' };
  }
}

/* ─── Tick — track each agent's last in-domain color for display ─── */

const lastDomainColor = {}; // agent id -> hex (only updated when world color is in the agent's domain)
const domainLocks = {};     // agent id -> bool (is the current world color in this agent's domain?)

function tickAll() {
  step++;
  const { ws, rgb, hex, domain } = world.step();

  for (const [id, a] of Object.entries(agents)) {
    a.step(ws);
    const d = DOMAINS[AGENTS.find(x => x.id === id).domain];
    const inDomain = d.test(rgb.r, rgb.g, rgb.b);
    if (inDomain) lastDomainColor[id] = hex;
    domainLocks[id] = inDomain;
  }

  for (const [id, f] of Object.entries(fused)) {
    f.step(ws);
    const fdef = FUSIONS.find(x => x.id === id);
    if (!fdef) continue;
    const inDomain = fdef.parents.some(p => domainLocks[p] || false);
    // Fused color = WEIGHTED blend of parents' domain colors, where weight
    // depends on which parents are currently active (perceiving their domain).
    const activeParents = fdef.parents.filter(p => domainLocks[p]);
    if (activeParents.length > 0) {
      const parentHexes = fdef.parents.map(p => domainHexFor(p));
      // Weight: active parents contribute more, but inactive ones still
      // contribute enough that the result is always recognizably a blend —
      // never collapses into the active parent's pure color.
      const weights = fdef.parents.map(p => domainLocks[p] ? 1.0 : 0.55);
      const totalW = weights.reduce((s, w) => s + w, 0);
      let r = 0, g = 0, b = 0;
      fdef.parents.forEach((p, i) => {
        const hex = parentHexes[i];
        const w = weights[i] / totalW;
        r += parseInt(hex.slice(1,3), 16) * w;
        g += parseInt(hex.slice(3,5), 16) * w;
        b += parseInt(hex.slice(5,7), 16) * w;
      });
      lastDomainColor[id] = rgbToHex(r, g, b);
    } else if (fdef.parents.some(p => lastDomainColor[p])) {
      // No parent active but some have history — weighted blend from stored colors
      const storedColors = fdef.parents.map(p => lastDomainColor[p] || domainHexFor(p));
      let r = 0, g = 0, b = 0;
      storedColors.forEach(hex => {
        r += parseInt(hex.slice(1,3), 16);
        g += parseInt(hex.slice(3,5), 16);
        b += parseInt(hex.slice(5,7), 16);
      });
      const n = storedColors.length;
      lastDomainColor[id] = rgbToHex(r/n, g/n, b/n);
    }
    domainLocks[id] = inDomain;
  }

  for (const [key, r] of Object.entries(reclaimed)) {
    r.agent.step(ws);
    const base = key.replace('_reclaimed', '');
    const inDomain = domainLocks[base] || false;
    // Reclaimed agents show their own base domain color, not the world color
    if (inDomain) lastDomainColor[key] = domainHexFor(base);
    domainLocks[key] = inDomain;
  }
}

/* ─── Perceived color: use the last in-domain world color, fall back to domain identity ─── */

function domainHexFor(id) {
  const def = AGENTS.find(a => a.id === id);
  if (def) return DOMAINS[def.domain].hex;
  const base = id.replace('_reclaimed', '');
  const bdef = AGENTS.find(a => a.id === base);
  if (bdef) return DOMAINS[bdef.domain].hex;
  const fdef = FUSIONS.find(f => f.id === id);
  if (fdef) {
    const parentHexes = fdef.parents.map(p => domainHexFor(p));
    return avgHex(parentHexes);
  }
  return '#888888';
}

function perceivedColor(id) {
  // If we have a recent in-domain color, use it directly — vibrant and live
  if (lastDomainColor[id]) return lastDomainColor[id];

  // Fallback: domain identity color, gently washed
  const hex = domainHexFor(id);
  const r0 = parseInt(hex.slice(1,3), 16);
  const g0 = parseInt(hex.slice(3,5), 16);
  const b0 = parseInt(hex.slice(5,7), 16);
  const breath = Math.sin(step * 0.025) * 12;
  return rgbToHex(r0 + breath, g0 + breath, b0 + breath);
}

function avgHex(hexes) {
  let r = 0, g = 0, b = 0;
  hexes.forEach(h => { r += parseInt(h.slice(1,3),16); g += parseInt(h.slice(3,5),16); b += parseInt(h.slice(5,7),16); });
  const n = hexes.length;
  return rgbToHex(r / n, g / n, b / n);
}

/* ─── Perceptual band label ─── */

function bandFor(id) {
  if (id.includes('reclaimed')) return 'visible';
  const def = AGENTS.find(a => a.id === id);
  if (!def) {
    // Fused — check parents for invisible
    const fdef = FUSIONS.find(f => f.id === id);
    if (!fdef) return 'visible';
    const hasInv = fdef.parents.some(p => bandFor(p) === 'invisible' || bandFor(p) === 'hybrid');
    const allInv = fdef.parents.every(p => bandFor(p) === 'invisible' || bandFor(p) === 'hybrid');
    return allInv ? 'invisible' : (hasInv ? 'hybrid' : 'visible');
  }
  const d = DOMAINS[def.domain];
  return d.label === 'UV' || d.label === 'IR' ? 'invisible' : 'visible';
}

/* ─── Snapshot ─── */

function getSnapshot() {
  const all = [];

  const push = (id, level, parentage) => {
    const agent = agents[id] || fused[id] || (reclaimed[id] && reclaimed[id].agent);
    if (!agent) return;
    all.push({
      id, level, parentage, band: bandFor(id),
      hex: perceivedColor(id),
      active: domainLocks[id] || false,
      locked: agent.isILocked,
      pe: +(agent.meanPredictionError || 0.5).toFixed(2),
      trie: agent.experience.trie.size(),
      meta: agent.experience.metaTrie.registrySize,
      learning: +Math.min(1, agent.experience.trie.size() / 80).toFixed(2),
    });
  };

  AGENTS.forEach(d => push(d.id, d.level, d.id));
  FUSIONS.forEach(d => push(d.id, d.level, d.parents.join('+')));
  Object.entries(reclaimed).forEach(([key, r]) => push(key, 0, r.parentage));

  return {
    step,
    agents: all,
    spectrum: DOMAIN_KEYS.map(k => ({ name: k, hex: DOMAINS[k].hex })),
    currentColor: rgbToHex(world._currentRGB.r, world._currentRGB.g, world._currentRGB.b),
    currentDomain: DOMAIN_KEYS[world._currentDomain],
  };
}

/* ─── HTTP server ─── */

const MIME = { '.html':'text/html','.js':'application/javascript','.json':'application/json','.md':'text/markdown' };
const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];
  if (url === '/state') {
    res.writeHead(200, { 'Content-Type':'application/json', 'Access-Control-Allow-Origin':'*', 'Cache-Control':'no-cache' });
    return res.end(JSON.stringify(getSnapshot()));
  }
  const file = url === '/' ? '/dashboard.html' : url;
  const fp = path.join(ROOT, file);
  if (!fp.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not Found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(fp)] || 'application/octet-stream' });
    res.end(data);
  });
});

trainPrimaries();
buildFusions();
setInterval(tickAll, 330);
server.listen(PORT, () => {
  console.log(`\n  Color Fusion — Full-RGB Perceptual Domains\n`);
  console.log(`  World:    full RGB cube, Markov chain over 5 perceptual domains`);
  console.log(`  Agents:   each has a domain filter — colors outside = outside interface`);
  console.log(`  API:      /state for live agent data\n`);
  console.log(`  →  http://localhost:${PORT}\n`);
});
