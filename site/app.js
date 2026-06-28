(function() {
'use strict';

/* ==============================
   1. SPA ROUTER
   ============================== */
const navLinks = document.querySelectorAll('.nav-links li');
const pages = document.querySelectorAll('.page');
const sidebar = document.getElementById('sidebar');
const mobileToggle = document.getElementById('mobileToggle');
const mobileOverlay = document.getElementById('mobileOverlay');
let currentRoute = 'home';

function navigate(route) {
  currentRoute = route;
  navLinks.forEach(el => el.classList.toggle('active', el.dataset.route === route));
  pages.forEach(el => el.classList.toggle('active', el.id === `page-${route}`));
  sidebar.classList.remove('open');
  mobileOverlay.classList.remove('open');
  if (route === 'home') initHeroBrain();
  if (route === 'monitor') initMonitor();
  if (route === 'network') initNetwork();
  if (route === 'docs') initDocs();
}

document.querySelectorAll('[data-nav]').forEach(el => {
  el.addEventListener('click', () => navigate(el.dataset.nav));
});

navLinks.forEach(el => {
  el.addEventListener('click', () => navigate(el.dataset.route));
});

mobileToggle.addEventListener('click', () => {
  sidebar.classList.toggle('open');
  mobileOverlay.classList.toggle('open');
});
mobileOverlay.addEventListener('click', () => {
  sidebar.classList.remove('open');
  mobileOverlay.classList.remove('open');
});

/* ==============================
   2. HERO BRAIN CANVAS
   ============================== */
let heroAnimId = null;
let heroParticles = [];
let heroHealth = 0;
let heroCanvas, heroCtx;

function initHeroBrain() {
  const canvas = document.getElementById('heroBrain');
  if (!canvas || heroAnimId) return;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  heroCanvas = canvas;
  heroCtx = canvas.getContext('2d');
  heroParticles = createBrainParticles(canvas.width, canvas.height, 1500);
  heroHealth = 0;
  simulateHeroHealth();
  if (heroAnimId) cancelAnimationFrame(heroAnimId);
  heroAnimId = requestAnimationFrame(renderHeroBrain);
}

function simulateHeroHealth() {
  if (currentRoute !== 'home') return;
  heroHealth = Math.max(0, heroHealth - 0.003);
  if (Math.random() < 0.005) heroHealth = Math.min(1, heroHealth + 0.3 + Math.random() * 0.5);
  setTimeout(simulateHeroHealth, 100);
}

function renderHeroBrain() {
  if (currentRoute !== 'home') { heroAnimId = null; return; }
  const ctx = heroCtx, w = heroCanvas.width, h = heroCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const intensity = heroHealth;
  heroParticles.forEach(p => {
    const targetX = p.bx + (Math.random() - 0.5) * 2;
    const targetY = p.by + (Math.random() - 0.5) * 2;
    const scatter = intensity * 200;
    p.x += (targetX + (Math.random() - 0.5) * scatter * 2 - p.x) * 0.02;
    p.y += (targetY + (Math.random() - 0.5) * scatter * 2 - p.y) * 0.02;

    const alpha = p.baseAlpha * (1 - intensity * 0.6);
    const sz = p.baseSize * (1 + intensity);
    ctx.beginPath();
    ctx.arc(p.x, p.y, sz, 0, Math.PI * 2);
    const r = Math.round(96 + (1 - intensity) * 48);
    const g = Math.round(165 + intensity * (-100));
    const b = Math.round(250 + intensity * (-100));
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.fill();
  });

  // Connections
  ctx.strokeStyle = `rgba(59,130,246,${0.04 * (1 - intensity)})`;
  ctx.lineWidth = 0.5;
  for (let i = 0; i < heroParticles.length; i += 10) {
    for (let j = i + 1; j < Math.min(i + 6, heroParticles.length); j += 3) {
      const dx = heroParticles[i].x - heroParticles[j].x;
      const dy = heroParticles[i].y - heroParticles[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 80) {
        ctx.globalAlpha = (1 - dist / 80) * 0.3 * (1 - intensity * 0.7);
        ctx.beginPath();
        ctx.moveTo(heroParticles[i].x, heroParticles[i].y);
        ctx.lineTo(heroParticles[j].x, heroParticles[j].y);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;

  heroAnimId = requestAnimationFrame(renderHeroBrain);
}

/* ==============================
   3. PARTICLE CLOUD SYSTEM
   ============================== */
let disturbances = [];

function createBrainParticles(w, h, count) {
  const particles = [];
  const cx = w / 2, cy = h / 2;
  const scale = Math.min(w, h) * 0.35;
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * scale * 0.6;
    const hemi = Math.random() < 0.5 ? -1 : 1;
    const hx = hemi * (Math.random() * 2 - 1) * scale * 0.3;
    const hy = (Math.random() * 2 - 1) * scale * 0.4;
    const bx = cx + hx;
    const by = cy + hy;
    particles.push({
      bx, by, x: bx, y: by, vx: 0, vy: 0,
      baseAlpha: 0.2 + Math.random() * 0.4, alpha: 0.2 + Math.random() * 0.4,
      baseSize: 1 + Math.random() * 1.5, size: 1 + Math.random() * 1.5,
      phase: Math.random() * Math.PI * 2, connectedTo: []
    });
  }
  for (let i = 0; i < particles.length; i += 3) {
    for (let j = i + 1; j < Math.min(i + 8, particles.length); j++) {
      const dx = particles[i].bx - particles[j].bx;
      const dy = particles[i].by - particles[j].by;
      if (dx * dx + dy * dy < 10000) {
        particles[i].connectedTo.push(j);
        particles[j].connectedTo.push(i);
      }
    }
  }
  return particles;
}

function createParticleCloud(w, h, count, spread) {
  const particles = [];
  const cx = w / 2, cy = h / 2;
  const maxR = Math.min(w, h) * (spread || 0.38);
  const minR = maxR * 0.1;
  for (let i = 0; i < count; i++) {
    const r = minR + (maxR - minR) * Math.pow(Math.random(), 0.6);
    const angle = Math.random() * Math.PI * 2;
    const bx = cx + Math.cos(angle) * r;
    const by = cy + Math.sin(angle) * r;
    particles.push({
      bx, by, x: bx, y: by, vx: 0, vy: 0,
      baseAlpha: 0.12 + Math.random() * 0.35, alpha: 0.12 + Math.random() * 0.35,
      baseSize: 0.8 + Math.random() * 1.8, size: 0.8 + Math.random() * 1.8,
      phase: Math.random() * Math.PI * 2, connectedTo: []
    });
  }
  for (let i = 0; i < particles.length; i += 2) {
    for (let j = i + 1; j < Math.min(i + 5, particles.length); j++) {
      const dx = particles[i].bx - particles[j].bx;
      const dy = particles[i].by - particles[j].by;
      if (dx * dx + dy * dy < 12000) {
        particles[i].connectedTo.push(j);
        particles[j].connectedTo.push(i);
      }
    }
  }
  return particles;
}

function addDisturbance(x, y, strength, radius) {
  disturbances.push({ x, y, strength, radius, age: 0, maxAge: 80 + Math.random() * 60 });
}

function updateDisturbances(particles, time) {
  for (let d = disturbances.length - 1; d >= 0; d--) {
    const dist = disturbances[d];
    dist.age++;
    const decay = 1 - dist.age / dist.maxAge;
    if (decay <= 0) { disturbances.splice(d, 1); continue; }
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const dx = p.x - dist.x;
      const dy = p.y - dist.y;
      const d2 = Math.sqrt(dx * dx + dy * dy);
      if (d2 < dist.radius && d2 > 1) {
        const influence = (1 - d2 / dist.radius) * decay;
        const radForce = influence * dist.strength * 20;
        p.vx += (dx / d2) * radForce;
        p.vy += (dy / d2) * radForce;
        const swirlForce = influence * dist.strength * 12;
        p.vx += (-dy / d2) * swirlForce;
        p.vy += (dx / d2) * swirlForce;
      }
    }
    if (dist.age < dist.maxAge * 0.6) {
      const a = Math.sin(dist.age * 0.05) * 0.8 + Math.cos(dist.age * 0.07) * 0.6;
      dist.x += Math.cos(a) * dist.strength * 2;
      dist.y += Math.sin(a * 0.7) * dist.strength * 2;
    }
  }
}

function updateSmokeParticles(particles, w, h, time, turbulence) {
  const t = turbulence || 0;
  const maxV = 6 + t * 20;
  const spring = 0.003 + t * 0.005;
  const damp = 0.96 - t * 0.01;
  const brown = 0.2 + t * 1.5;
  const oscMul = 1 + t * 3;
  for (let i = 0, n = particles.length; i < n; i++) {
    const p = particles[i];
    const oscX = Math.sin(time * 0.0008 + p.phase) * oscMul;
    const oscY = Math.cos(time * 0.0011 + p.phase * 1.3) * oscMul;
    p.vx += (p.bx + oscX - p.x) * spring;
    p.vy += (p.by + oscY - p.y) * spring;
    p.vx *= damp;
    p.vy *= damp;
    p.vx += (Math.random() - 0.5) * brown;
    p.vy += (Math.random() - 0.5) * brown;
    const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
    if (speed > maxV) { p.vx = (p.vx / speed) * maxV; p.vy = (p.vy / speed) * maxV; }
    p.x += p.vx;
    p.y += p.vy;
    if (p.x < -60) p.x = -60; else if (p.x > w + 60) p.x = w + 60;
    if (p.y < -60) p.y = -60; else if (p.y > h + 60) p.y = h + 60;
    p.alpha = p.baseAlpha * (1 - t * 0.3);
    p.size = p.baseSize * (1 + t * 0.6);
  }
}

/* ==============================
   4. SERVER MONITOR
   ============================== */
const AGENT_DEFS = [
  { id: 'ram', name: 'RAM', unit: '%', normal: [40, 65], warning: [75, 88], critical: [90, 98] },
  { id: 'cpu', name: 'CPU', unit: '%', normal: [15, 40], warning: [60, 80], critical: [85, 100] },
  { id: 'temp', name: 'CPU Temp', unit: '°C', normal: [38, 55], warning: [65, 78], critical: [80, 95] },
  { id: 'log', name: 'Log Anomalies', unit: '/min', normal: [0, 2], warning: [5, 12], critical: [15, 30] },
  { id: 'net', name: 'Latency', unit: 'ms', normal: [5, 20], warning: [50, 120], critical: [200, 500] },
  { id: 'disk', name: 'Disk I/O', unit: '% wait', normal: [0, 5], warning: [15, 30], critical: [40, 80] },
  { id: 'proc', name: 'Process Health', unit: 'issues', normal: [0, 0], warning: [1, 2], critical: [3, 8] }
];

const TOKENS = ['stable', 'nominal', 'normal', 'steady', 'monitoring', 'observing',
  'anomaly', 'drift', 'strain', 'warning', 'alert', 'unstable',
  'critical', 'failure', 'overload', 'breach', 'panic', 'meltdown'];

let monitorAgents = [];
let monitorPaused = false;
let monitorInterval = null;
let particleAnimId = null;
let brainParticles = [];
let particleCanvas, particleCtx;

function initMonitor() {
  if (!monitorGridBuilt()) {
    buildMonitorGrid();
    resetMonitorAgents();
  }
  startMonitorLoop();
  initParticleBrain();
}

function monitorGridBuilt() {
  return document.getElementById('monitorGrid').children.length > 0;
}

function buildMonitorGrid() {
  const grid = document.getElementById('monitorGrid');
  grid.innerHTML = '';
  AGENT_DEFS.forEach(def => {
    const card = document.createElement('div');
    card.className = 'monitor-card status-normal';
    card.id = `mcard-${def.id}`;
    card.innerHTML = `
      <div class="monitor-card-header">
        <span class="monitor-card-name">${def.name}</span>
        <span class="monitor-card-status">normal</span>
      </div>
      <div class="monitor-card-value" id="mval-${def.id}">--</div>
      <div class="monitor-card-bar"><div class="monitor-card-fill" id="mbar-${def.id}" style="width:0%"></div></div>
      <div class="monitor-card-detail">
        <span>pred_err: <span id="merr-${def.id}">0.000</span></span>
        <span>token: <span id="mtok-${def.id}">--</span></span>
      </div>
      <div class="monitor-card-agent">
        <span class="monitor-card-lock"><span class="dot unlocked" id="mlock-${def.id}"></span> I-lock</span>
        <span id="mstep-${def.id}">step 0</span>
      </div>`;
    grid.appendChild(card);
  });
}

function resetMonitorAgents() {
  monitorAgents = AGENT_DEFS.map(def => {
    const n = def.normal;
    const baseVal = n[0] + Math.random() * (n[1] - n[0]);
    return {
      def,
      value: baseVal,
      baseValue: baseVal,
      predError: 0.05 + Math.random() * 0.1,
      isLocked: Math.random() < 0.3,
      token: TOKENS[0],
      step: 0,
      status: 'normal',
      anomalyTimer: 0,
      anomalyDuration: 0,
      anomalyType: null
    };
  });
}

function startMonitorLoop() {
  if (monitorInterval) clearInterval(monitorInterval);
  monitorInterval = setInterval(tickMonitor, 800);
}

function tickMonitor() {
  if (monitorPaused) return;

  monitorAgents.forEach(a => {
    a.step++;

    // Anomaly management
    if (a.anomalyTimer > 0) {
      a.anomalyTimer--;
      if (a.anomalyTimer <= 0) {
        const n = a.def.normal;
        a.baseValue = n[0] + Math.random() * (n[1] - n[0]);
        a.predError = 0.05 + Math.random() * 0.15;
      }
    }

    // Random anomaly injection
    if (a.anomalyTimer <= 0 && Math.random() < 0.008) {
      a.anomalyType = Math.random() < 0.5 ? 'warning' : 'critical';
      const target = a.def[a.anomalyType];
      a.baseValue = target[0] + Math.random() * (target[1] - target[0]);
      a.predError = a.anomalyType === 'critical' ? 0.6 + Math.random() * 0.35 : 0.3 + Math.random() * 0.25;
      a.anomalyTimer = Math.floor(5 + Math.random() * 12);
      if (a.anomalyType === 'critical') triggerMonitorDisturbance();
    }

    // Random walk
    const drift = (Math.random() - 0.5) * 3;
    a.value += drift;
    a.value = Math.max(0, Math.min(100, a.value));

    // Mean reversion toward base value
    a.value += (a.baseValue - a.value) * 0.08;

    // Prediction error decay
    a.predError = Math.max(0.02, a.predError * (1 - 0.02));

    // Determine status
    const v = a.value;
    const def = a.def;
    if (v >= def.critical[0]) a.status = 'critical';
    else if (v >= def.warning[0]) a.status = 'warning';
    else a.status = 'normal';

    // I-lock: unlock on high error
    if (a.predError > 0.5) {
      if (Math.random() < 0.05) a.isLocked = false;
    } else if (a.predError < 0.2 && a.status === 'normal') {
      if (Math.random() < 0.02) a.isLocked = true;
    }

    // Generate token
    const errIdx = a.predError < 0.2 ? 0 : a.predError < 0.5 ? 1 : 2;
    const tokensByErr = [
      ['stable', 'nominal', 'normal', 'steady', 'monitoring', 'observing'],
      ['anomaly', 'drift', 'strain', 'warning', 'alert', 'unstable'],
      ['critical', 'failure', 'overload', 'breach', 'panic', 'meltdown']
    ];
    const pool = tokensByErr[errIdx];
    a.token = pool[Math.floor(Math.random() * pool.length)];

    // Update UI
    const cid = a.def.id;
    const card = document.getElementById(`mcard-${cid}`);
    card.className = `monitor-card status-${a.status}`;
    document.getElementById(`mval-${cid}`).textContent = formatValue(a.value, a.def.unit);
    document.querySelector(`#mcard-${cid} .monitor-card-status`).textContent = a.status;
    document.getElementById(`mbar-${cid}`).style.width = `${a.value}%`;
    document.getElementById(`merr-${cid}`).textContent = a.predError.toFixed(3);
    document.getElementById(`mtok-${cid}`).textContent = a.token;
    const dot = document.getElementById(`mlock-${cid}`);
    dot.className = `dot ${a.isLocked ? 'locked' : 'unlocked'}`;
    document.getElementById(`mstep-${cid}`).textContent = `step ${a.step}`;
  });

  // Update summary
  updateMonitorSummary();
}

function formatValue(val, unit) {
  if (unit === '°C') return `${Math.round(val)}°C`;
  if (unit === '%' || unit === '% wait') return `${Math.round(val)}${unit}`;
  if (unit === '/min') return `${Math.round(val)}/min`;
  if (unit === 'ms') return `${Math.round(val)}ms`;
  if (unit === 'issues') return `${Math.round(val)}`;
  return `${val.toFixed(1)}${unit}`;
}

function updateMonitorSummary() {
  const total = monitorAgents.length;
  const locked = monitorAgents.filter(a => a.isLocked).length;
  const critical = monitorAgents.filter(a => a.status === 'critical').length;
  const warning = monitorAgents.filter(a => a.status === 'warning').length;
  const avgErr = monitorAgents.reduce((s, a) => s + a.predError, 0) / total;

  const el = document.getElementById('monitorSummary');
  el.innerHTML = `
    <div class="summary-item"><div class="label">System Health</div><div class="value" style="color:${critical > 0 ? '#ef4444' : warning > 0 ? '#eab308' : '#22c55e'}">${critical > 0 ? 'Critical' : warning > 0 ? 'Warning' : 'Normal'}</div></div>
    <div class="summary-item"><div class="label">I-Locked</div><div class="value" style="color:#22c55e">${locked}/${total}</div></div>
    <div class="summary-item"><div class="label">Avg Pred Error</div><div class="value">${avgErr.toFixed(3)}</div></div>
    <div class="summary-item"><div class="label">Anomalies</div><div class="value" style="color:#ef4444">${critical + warning}</div></div>
  `;
}

/* ==============================
   5. MONITOR SMOKE CLOUD
   ============================== */
let monitorCloudTime = 0;

function initParticleBrain() {
  const canvas = document.getElementById('particleBrain');
  if (!canvas || particleAnimId) return;
  canvas.width = Math.max(canvas.clientWidth, 400) * devicePixelRatio;
  canvas.height = Math.max(canvas.clientHeight, 280) * devicePixelRatio;
  particleCanvas = canvas;
  particleCtx = canvas.getContext('2d');
  brainParticles = createParticleCloud(canvas.width, canvas.height, 2000, 0.42);
  disturbances = [];
  if (particleAnimId) cancelAnimationFrame(particleAnimId);
  particleAnimId = requestAnimationFrame(renderMonitorCloud);
}

function triggerMonitorDisturbance() {
  const w = particleCanvas.width, h = particleCanvas.height;
  const cx = w / 2, cy = h / 2;
  const angle = Math.random() * Math.PI * 2;
  const r = Math.random() * Math.min(w, h) * 0.25;
  const x = cx + Math.cos(angle) * r;
  const y = cy + Math.sin(angle) * r;
  const strength = 0.5 + Math.random() * 1.0;
  const radius = Math.min(w, h) * (0.1 + Math.random() * 0.15);
  addDisturbance(x, y, strength, radius);
}

function getSystemHealth() {
  const critical = monitorAgents.filter(a => a.status === 'critical').length;
  const warning = monitorAgents.filter(a => a.status === 'warning').length;
  const total = monitorAgents.length;
  return (critical * 1.0 + warning * 0.5) / total;
}

function renderMonitorCloud() {
  if (currentRoute !== 'monitor' || !particleCanvas) { particleAnimId = null; return; }
  const ctx = particleCtx, w = particleCanvas.width, h = particleCanvas.height;
  monitorCloudTime++;
  const health = getSystemHealth();

  ctx.clearRect(0, 0, w, h);

  updateSmokeParticles(brainParticles, w, h, monitorCloudTime, health);
  updateDisturbances(brainParticles, monitorCloudTime);

  // Background glow on critical
  if (health > 0.3) {
    const grad = ctx.createRadialGradient(w/2, h/2, 0, w/2, h/2, Math.max(w,h) * 0.55);
    const clr = health > 0.5 ? '239,68,68' : '234,179,8';
    grad.addColorStop(0, `rgba(${clr},${(health - 0.3) * 0.06})`);
    grad.addColorStop(1, `rgba(${clr},0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }

  // Draw connections (sparse)
  for (let i = 0; i < brainParticles.length; i += 12) {
    const p = brainParticles[i];
    for (let k = 0; k < p.connectedTo.length && k < 4; k++) {
      const other = brainParticles[p.connectedTo[k]];
      if (!other) continue;
      const dx = p.x - other.x;
      const dy = p.y - other.y;
      if (dx > 120 || dy > 120) continue;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 120 && d > 0) {
        ctx.globalAlpha = (1 - d / 120) * 0.12 * (1 - health * 0.3);
        ctx.strokeStyle = `rgba(148,163,184,${ctx.globalAlpha})`;
        ctx.lineWidth = 0.3;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(other.x, other.y);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;

  // Draw particles (fast fillRect)
  const colorBase = health > 0.5 ? '239,68,68' : health > 0.2 ? '234,179,8' : '148,163,184';
  for (let i = 0, n = brainParticles.length; i < n; i++) {
    const p = brainParticles[i];
    ctx.fillStyle = `rgba(${colorBase},${p.alpha})`;
    ctx.fillRect(p.x - p.size, p.y - p.size, p.size * 2, p.size * 2);
  }

  particleAnimId = requestAnimationFrame(renderMonitorCloud);
}

/* ==============================
   6. NETWORK SMOKE CLOUD
   ============================== */
let networkParticles = [];
let networkPaused = false;
let networkAnimId = null;
let networkCanvas, networkCtx;
let networkCloudTime = 0;
let networkAgents = [];
let networkAgentsN = 400;
let networkStatsTicker = 0;

const COMBO_LEVELS = [
  { name: 'Fused',    rgba: '167,139,250', sizeMul: 1.5, pulseAmp: 0.04, alphaAdd: 0.2, label: 'L1' },
  { name: 'United',   rgba: '250,204,21',  sizeMul: 1.8, pulseAmp: 0.06, alphaAdd: 0.3, label: 'L2' },
  { name: 'Coherent', rgba: '34,211,238',  sizeMul: 2.2, pulseAmp: 0.08, alphaAdd: 0.4, label: 'L3' },
  { name: 'Planck',   rgba: '255,255,255', sizeMul: 2.8, pulseAmp: 0.10, alphaAdd: 0.6, label: 'L4' },
];

function initNetwork() {
  const canvas = document.getElementById('networkCanvas');
  if (!canvas || networkAnimId) return;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  networkCanvas = canvas;
  networkCtx = canvas.getContext('2d');
  resetNetworkCloud();
  networkAnimId = requestAnimationFrame(renderNetworkCloud);
}

function resetNetworkCloud() {
  const w = networkCanvas.width, h = networkCanvas.height;
  disturbances = [];
  networkCloudTime = 0;
  networkStatsTicker = 0;
  networkAgentsN = 400;
  networkParticles = createParticleCloud(w, h, networkAgentsN, 0.38);
  networkAgents = [];
  for (let i = 0; i < networkAgentsN; i++) {
    networkAgents.push({ level: 0, locked: Math.random() < 0.3, predError: 0.05 + Math.random() * 0.2 });
  }
  updateNetworkStats();
}

function tickNetworkCloud() {
  if (networkPaused) return;
  networkCloudTime++;

  // Count agents per level for cascade speed multiplier
  let levelCounts = [0,0,0,0,0];
  for (let i = 0; i < networkAgentsN; i++) {
    const a = networkAgents[i];
    levelCounts[a.level]++;
    if (a.level === 0) {
      if (a.predError > 0.5) { if (Math.random() < 0.03) a.locked = false; }
      else if (a.predError < 0.15) { if (Math.random() < 0.015) a.locked = true; }
      a.predError += (Math.random() - 0.5) * 0.04;
      if (a.predError > 0.95) a.predError = 0.95;
      if (a.predError < 0.01) a.predError = 0.01;
    }
  }

  const comboTotal = networkAgentsN - levelCounts[0];
  const cascadeMul = 1 + comboTotal / networkAgentsN * 3;

  // Level-0 locked agents combine → Level 1
  for (let i = 0; i < networkAgentsN; i++) {
    const a = networkAgents[i];
    if (a.level !== 0 || !a.locked) continue;
    if (Math.random() > 0.003 * cascadeMul) continue;
    for (let j = i + 1; j < Math.min(i + 40, networkAgentsN); j++) {
      const b = networkAgents[j];
      if (b.level !== 0 || !b.locked) continue;
      a.level = 1; b.level = 1;
      triggerCombine(i, j, 1);
      break;
    }
  }

  // Higher-level combinations: L1→L2, L2→L3, L3→L4
  for (let level = 1; level < 4; level++) {
    const count = levelCounts[level];
    if (count < 2) continue;
    const prob = 0.002 * cascadeMul * (1 + level * 2) / Math.max(1, count / 10);
    for (let i = 0; i < networkAgentsN; i++) {
      const a = networkAgents[i];
      if (a.level !== level) continue;
      if (Math.random() > prob) continue;
      for (let j = i + 1; j < Math.min(i + 60, networkAgentsN); j++) {
        const b = networkAgents[j];
        if (b.level !== level) continue;
        a.level = level + 1; b.level = level + 1;
        triggerCombine(i, j, level + 1);
        break;
      }
    }
  }

  networkStatsTicker++;
  if (networkStatsTicker % 6 === 0) updateNetworkStats();
}

function triggerCombine(i, j, level) {
  const pi = networkParticles[i], pj = networkParticles[j];
  if (!pi || !pj) return;
  const distRadius = Math.min(networkCanvas.width, networkCanvas.height) * (0.08 + level * 0.04);
  const strength = 0.4 + level * 0.3;
  addDisturbance((pi.x + pj.x) / 2, (pi.y + pj.y) / 2, strength, distRadius);
}

function renderNetworkCloud() {
  if (currentRoute !== 'network' || !networkCanvas) { networkAnimId = null; return; }
  const ctx = networkCtx, w = networkCanvas.width, h = networkCanvas.height;
  ctx.fillStyle = '#080c16';
  ctx.fillRect(0, 0, w, h);

  tickNetworkCloud();
  updateSmokeParticles(networkParticles, w, h, networkCloudTime, getNetworkTurbulence());
  updateDisturbances(networkParticles, networkCloudTime);

  const time = networkCloudTime;

  // Edges between nearby combined agents (L1+)
  let edgeCount = 0;
  for (let i = 0; i < networkAgentsN && edgeCount < 600; i += 2) {
    const a = networkAgents[i];
    if (a.level === 0) continue;
    const pi = networkParticles[i];
    if (!pi) continue;
    for (let j = i + 2; j < networkAgentsN && edgeCount < 600; j += 2) {
      const b = networkAgents[j];
      if (b.level === 0) continue;
      const pj = networkParticles[j];
      if (!pj) continue;
      const dx = pi.x - pj.x, dy = pi.y - pj.y;
      if (dx > 140 || dy > 140) continue;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 140 && d > 5) {
        const alpha = (1 - d / 140) * 0.25 * Math.min(1, (a.level + b.level) / 4);
        const lvl = COMBO_LEVELS[Math.max(a.level, b.level) - 1];
        ctx.strokeStyle = `rgba(${lvl.rgba},${alpha})`;
        ctx.lineWidth = 0.3 + Math.max(a.level, b.level) * 0.15;
        ctx.beginPath();
        ctx.moveTo(pi.x, pi.y);
        ctx.lineTo(pj.x, pj.y);
        ctx.stroke();
        edgeCount++;
      }
    }
  }

  for (let i = 0; i < networkAgentsN; i++) {
    const p = networkParticles[i];
    const a = networkAgents[i];
    let s = p.size;
    let r, g, b, al = p.alpha;

    if (a.level === 0) {
      if (a.locked) { r = 34; g = 197; b = 94; al += 0.25; s *= 1.3; }
      else if (a.predError > 0.5) { r = 239; g = 68; b = 68; al += 0.1; }
      else { r = 71; g = 85; b = 105; }
      ctx.fillStyle = `rgba(${r},${g},${b},${al})`;
    } else {
      const lvl = COMBO_LEVELS[a.level - 1];
      const pulse = 1 + Math.sin(time * 0.04 + i * 0.15) * lvl.pulseAmp;
      s *= lvl.sizeMul * pulse;
      ctx.fillStyle = `rgba(${lvl.rgba},${Math.min(1, al + lvl.alphaAdd)})`;
    }
    ctx.fillRect(p.x - s, p.y - s, s * 2, s * 2);
  }

  networkAnimId = requestAnimationFrame(renderNetworkCloud);
}

function updateNetworkStats() {
  const counts = [0,0,0,0,0];
  let locked = 0, sumErr = 0;
  for (let i = 0; i < networkAgentsN; i++) {
    const a = networkAgents[i];
    counts[a.level]++;
    if (a.level === 0 && a.locked) locked++;
    sumErr += a.predError;
  }
  const el = document.getElementById('networkStats');
  if (!el) return;
  el.innerHTML =
    `<div class="stat-row"><span class="stat-label">Agents</span><span class="stat-value">${networkAgentsN}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Locked (L0)</span><span class="stat-value" style="color:#22c55e">${locked}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Fused (L1)</span><span class="stat-value" style="color:#a78bfa">${counts[1]}</span></div>` +
    `<div class="stat-row"><span class="stat-label">United (L2)</span><span class="stat-value" style="color:#eab308">${counts[2]}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Coherent (L3)</span><span class="stat-value" style="color:#22d3ee">${counts[3]}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Planck (L4)</span><span class="stat-value" style="color:#f8fafc">${counts[4]}</span></div>` +
    `<div class="stat-row"><span class="stat-label">Avg Pred Error</span><span class="stat-value">${(sumErr / networkAgentsN).toFixed(3)}</span></div>`;
}

function getNetworkTurbulence() {
  let locked = 0, sum = 0;
  for (let i = 0; i < networkAgentsN; i++) {
    const a = networkAgents[i];
    if (a.level === 0 && a.locked) locked++;
    sum += a.predError;
  }
  return (networkAgentsN - locked) / networkAgentsN * 0.2 + sum / networkAgentsN * 0.3;
}

/* ==============================
   7. DOCUMENTATION VIEWER
   ============================== */
const DOC_FILES = [
  { id: 'theory', label: 'Theory', file: 'CONSCIOUS_AGENTS_THEORY.md' },
  { id: 'api', label: 'Runtime API', file: 'CA_RUNTIME_API.md' },
  { id: 'components', label: 'Components', file: 'COMPONENT_DEFINITIONS.md' },
  { id: 'self-awareness', label: 'Self-Awareness', file: 'SELF_AWARENESS.md' },
  { id: 'visual-guide', label: 'Visual Guide', file: 'CONSCIOUS_AGENTS_VISUAL_GUIDE.md' },
  { id: 'glossary', label: 'Glossary', file: 'GLOSSARY.md' },
  { id: 'qa', label: 'Q&A', file: 'Q_AND_A.md' }
];

let docsLoaded = {};

function initDocs() {
  const list = document.getElementById('docList');
  if (list.children.length > 0) return;

  DOC_FILES.forEach((doc, idx) => {
    const li = document.createElement('li');
    li.textContent = doc.label;
    li.dataset.doc = doc.id;
    li.addEventListener('click', () => loadDoc(doc));
    list.appendChild(li);
    if (idx === 0) loadDoc(doc, true);
  });
}

async function loadDoc(doc, initial = false) {
  document.querySelectorAll('#docList li').forEach(el => el.classList.remove('active'));
  document.querySelector(`#docList li[data-doc="${doc.id}"]`)?.classList.add('active');

  const content = document.getElementById('docContent');
  content.innerHTML = '<div style="padding:40px;text-align:center;color:#64748b">Loading...</div>';

  try {
    const resp = await fetch(`docs/${doc.file}`);
    const md = await resp.text();
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        breaks: true,
        gfm: true,
        langPrefix: 'language-'
      });
      let html = marked.parse(md);
      // Fix mermaid code blocks - wrap them in a container
      html = html.replace(/<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g, (_, code) => {
        return `<div class="mermaid-block"><pre><code class="language-mermaid">${code}</code></pre></div>`;
      });
      content.innerHTML = `<div class="doc-body">${html}</div>`;
      if (typeof hljs !== 'undefined') {
        content.querySelectorAll('pre code:not(.language-mermaid)').forEach(block => {
          hljs.highlightElement(block);
        });
      }
    } else {
      content.innerHTML = `<div class="doc-body"><pre>${md}</pre></div>`;
    }
  } catch (e) {
    content.innerHTML = `<div style="padding:40px;text-align:center;color:#ef4444">Error loading document: ${e.message}</div>`;
  }
}

/* ==============================
   8. CODE TAB SWITCHING (Home)
   ============================== */
document.querySelectorAll('.code-tab').forEach(tab => {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
    this.classList.add('active');
    const lang = this.dataset.lang;
    const codeEl = document.getElementById('heroCode');
    if (lang === 'python') {
      codeEl.textContent = `from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld

world = CoinTossWorld(n_coins=4)
agent = ConsciousAgent(world=world, agent_id="my_agent")
outputs = agent.run(n_steps=1000)
print(f'"I" locked: {agent.is_i_locked}')`;
    } else {
      codeEl.textContent = `const { ConsciousAgent } = require('conscious-agent');
const { CoinTossWorld } = require('conscious-agent/worlds');

const world = new CoinTossWorld({ nCoins: 4 });
const agent = new ConsciousAgent({ agentId: 'my_agent', world });
const outputs = agent.run({ nSteps: 1000 });
console.log('I locked:', agent.isILocked);`;
    }
    codeEl.className = `language-${lang}`;
    if (typeof hljs !== 'undefined') hljs.highlightElement(codeEl);
  });
});

/* ==============================
   9. CONTROLS
   ============================== */
document.getElementById('monitorPause')?.addEventListener('click', function() {
  monitorPaused = !monitorPaused;
  this.textContent = monitorPaused ? 'Resume' : 'Pause';
});
document.getElementById('monitorReset')?.addEventListener('click', () => {
  resetMonitorAgents();
});
document.getElementById('networkPause')?.addEventListener('click', function() {
  networkPaused = !networkPaused;
  this.textContent = networkPaused ? 'Resume' : 'Pause';
});
document.getElementById('networkReset')?.addEventListener('click', () => {
  resetNetworkCloud();
});

/* ==============================
   10. INIT
   ============================== */
navigate('home');

// Handle window resize for canvases
let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (currentRoute === 'home') {
      if (heroAnimId) { cancelAnimationFrame(heroAnimId); heroAnimId = null; }
      initHeroBrain();
    }
    if (currentRoute === 'monitor') {
      if (particleAnimId) { cancelAnimationFrame(particleAnimId); particleAnimId = null; }
      initParticleBrain();
    }
    if (currentRoute === 'network') {
      if (networkAnimId) { cancelAnimationFrame(networkAnimId); networkAnimId = null; }
      initNetwork();
    }
  }, 300);
});

// Clean up intervals/animation on page unload
window.addEventListener('beforeunload', () => {
  if (monitorInterval) clearInterval(monitorInterval);
  if (heroAnimId) cancelAnimationFrame(heroAnimId);
  if (particleAnimId) cancelAnimationFrame(particleAnimId);
  if (networkAnimId) cancelAnimationFrame(networkAnimId);
});

})();
