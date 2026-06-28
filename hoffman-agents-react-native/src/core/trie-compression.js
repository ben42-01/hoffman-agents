const { ExperienceTrie } = require('./experience-trie');

function norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

function cosineSimilarity(a, b) {
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot / (na * nb);
}

function prune(trie, minVisits, protectedNodes) {
  protectedNodes = protectedNodes || new Set();

  const canPrune = (stateId) => !protectedNodes.has(stateId);

  const pruneRecursive = (parentPath) => {
    const node = parentPath.length === 0 ? trie.root : trie.lookup(parentPath);
    if (!node) return;

    for (const cs of Object.keys(node.children)) {
      pruneRecursive([...parentPath, parseInt(cs)]);
    }

    const toDelete = [];
    for (const [cs, child] of Object.entries(node.children)) {
      if (child.visitCount < minVisits && Object.keys(child.children).length === 0 && canPrune(child.stateId)) {
        toDelete.push(parseInt(cs));
      }
    }
    for (const cs of toDelete) delete node.children[cs];
  };

  pruneRecursive([]);
}

function traceDistance(pathA, pathB, transitionMatrix) {
  const n = pathA.length, m = pathB.length;

  const dp = Array.from({ length: n + 1 }, () => new Float64Array(m + 1));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      let cost = 1;
      if (pathA[i - 1] === pathB[j - 1]) {
        cost = 0;
      } else if (transitionMatrix) {
        const a = pathA[i - 1], b = pathB[j - 1];
        if (a < transitionMatrix.length && b < transitionMatrix[0].length) {
          cost = 1 - cosineSimilarity(transitionMatrix[a], transitionMatrix[b]);
        }
      }

      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }

  const maxLen = Math.max(n, m);
  return maxLen === 0 ? 0 : dp[n][m] / maxLen;
}

function mergeSimilarPaths(trie, transitionMatrix, threshold = 0.15, protectedNodes) {
  protectedNodes = protectedNodes || new Set();
  const leafPaths = trie.getLeafPaths(1);
  let mergedCount = 0;
  const mergedPaths = new Set();

  for (let i = 0; i < leafPaths.length; i++) {
    const pathI = leafPaths[i];
    const keyI = pathI.join(',');

    if (mergedPaths.has(keyI)) continue;

    const nodeI = trie.lookup(pathI);
    if (!nodeI) continue;
    if (protectedNodes.has(nodeI.stateId)) continue;

    for (let j = i + 1; j < leafPaths.length; j++) {
      const pathJ = leafPaths[j];
      const keyJ = pathJ.join(',');

      if (mergedPaths.has(keyJ)) continue;

      const nodeJ = trie.lookup(pathJ);
      if (!nodeJ) continue;
      if (protectedNodes.has(nodeJ.stateId)) continue;

      const dist = traceDistance(pathI, pathJ, transitionMatrix);
      if (dist < threshold) {
        if (nodeI.visitCount >= nodeJ.visitCount) {
          nodeI.visitCount += nodeJ.visitCount;
          _removePath(trie, pathJ);
          mergedPaths.add(keyJ);
        } else {
          nodeJ.visitCount += nodeI.visitCount;
          _removePath(trie, pathI);
          mergedPaths.add(keyI);
          break;
        }
        mergedCount++;
      }
    }
  }
  return mergedCount;
}

function _removePath(trie, path) {
  if (path.length === 0) return;
  for (let depth = path.length; depth >= 1; depth--) {
    const prefix = path.slice(0, depth);
    const node = trie.lookup(prefix);
    if (!node) continue;

    if (depth < path.length) {
      const childState = path[depth];
      if (node.children[childState]) {
        if (node.children[childState].visitCount <= 0 && Object.keys(node.children[childState].children).length === 0) {
          delete node.children[childState];
        }
      }
    } else {
      node.visitCount = 0;
      node.predictionErrors = [];
      node.meanPredictionError = 0;
    }
  }
}

module.exports = { prune, traceDistance, mergeSimilarPaths, cosineSimilarity, norm };
