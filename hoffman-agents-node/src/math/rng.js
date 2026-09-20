// Deterministic, cross-language (Node <-> Python) primitives.
// Both functions are ported bit-exactly to conscious_agent/math/rng.py.

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// FNV-1a 32-bit over the UTF-8 bytes of a string.
function fnv1a32(str) {
  const bytes = Buffer.from(String(str), 'utf8');
  let h = 0x811C9DC5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

module.exports = { mulberry32, fnv1a32 };
