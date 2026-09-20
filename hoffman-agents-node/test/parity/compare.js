// Structural comparison of two parity traces. Floats compare with a tolerance
// (the languages sum in different orders); everything else must be identical.
function compare(a, b, pathStr = '$', tol = 1e-9, diffs = []) {
  if (diffs.length > 20) return diffs;
  if (typeof a === 'number' && typeof b === 'number') {
    if (!(a === b || Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)))) diffs.push(`${pathStr}: ${a} != ${b}`);
  } else if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) diffs.push(`${pathStr}: length ${a.length} != ${b.length}`);
    for (let i = 0; i < Math.min(a.length, b.length); i++) compare(a[i], b[i], `${pathStr}[${i}]`, tol, diffs);
  } else if (a && b && typeof a === 'object' && typeof b === 'object') {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) {
      if (!(k in a) || !(k in b)) diffs.push(`${pathStr}.${k}: missing on one side`);
      else compare(a[k], b[k], `${pathStr}.${k}`, tol, diffs);
    }
  } else if (a !== b) {
    diffs.push(`${pathStr}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
  }
  return diffs;
}

module.exports = { compare };
