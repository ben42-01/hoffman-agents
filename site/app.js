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

const ROUTES = ['home', 'how', 'docs'];

function navigate(route) {
  if (!ROUTES.includes(route)) route = 'home';
  currentRoute = route;
  if (location.hash.slice(1) !== route) history.replaceState(null, '', route === 'home' ? location.pathname : `#${route}`);
  navLinks.forEach(el => el.classList.toggle('active', el.dataset.route === route));
  pages.forEach(el => el.classList.toggle('active', el.id === `page-${route}`));
  sidebar.classList.remove('open');
  mobileOverlay.classList.remove('open');
  if (route !== 'how' && window.SixTuple) window.SixTuple.stop();
  if (route === 'home') initHeroBrain();
  if (route === 'how' && window.SixTuple) window.SixTuple.init();
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
   3. HERO PARTICLES (decorative)
   ============================== */
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

/* ==============================
   7. DOCUMENTATION VIEWER
   ============================== */
const DOC_FILES = [
  { id: 'math', label: 'Mathematical Model', file: 'MATHEMATICAL_MODEL.md' },
  { id: 'qa', label: 'Results, Q&A & Corrections', file: 'Q_AND_A.md' },
  { id: 'components', label: 'Components', file: 'COMPONENT_DEFINITIONS.md' },
  { id: 'visual-guide', label: 'Visual Guide', file: 'CONSCIOUS_AGENTS_VISUAL_GUIDE.md' },
  { id: 'self-modelling', label: 'Self-Modelling', file: 'SELF_AWARENESS.md' },
  { id: 'glossary', label: 'Glossary', file: 'GLOSSARY.md' },
  { id: 'theory', label: 'Design Document (hypotheses)', file: 'CONSCIOUS_AGENTS_THEORY.md' },
  { id: 'api', label: 'API Design Spec (partly unimplemented)', file: 'CA_RUNTIME_API.md' }
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
agent = ConsciousAgent(agent_id="my_agent", seed=1, world=world)
agent.run(2000)
print(agent.ergodic_stats()["lock"])  # lock state and criteria`;
    } else {
      codeEl.textContent = `const { ConsciousAgent } = require('conscious-agent');
const { CoinTossWorld } = require('conscious-agent/worlds');

const world = new CoinTossWorld(4);
const agent = new ConsciousAgent({ agentId: 'my_agent', seed: 1, world });
agent.run(2000);
console.log(agent.ergodicStats().lock); // lock state and criteria`;
    }
    codeEl.className = `language-${lang}`;
    if (typeof hljs !== 'undefined') hljs.highlightElement(codeEl);
  });
});

/* ==============================
   9. INIT
   ============================== */
navigate(location.hash.slice(1) || 'home');
window.addEventListener('hashchange', () => navigate(location.hash.slice(1) || 'home'));

// Handle window resize for the hero canvas
let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (currentRoute === 'home') {
      if (heroAnimId) { cancelAnimationFrame(heroAnimId); heroAnimId = null; }
      initHeroBrain();
    }
  }, 300);
});

window.addEventListener('beforeunload', () => {
  if (heroAnimId) cancelAnimationFrame(heroAnimId);
});

})();
