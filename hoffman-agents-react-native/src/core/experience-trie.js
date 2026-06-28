class TrieNode {
  constructor(stateId, depth = 0) {
    this.stateId = stateId;
    this.children = {};
    this.visitCount = 0;
    this.predictionErrors = [];
    this.meanPredictionError = 0;
    this.wordBinding = null;
    this.depth = depth;
  }

  addPredictionError(error) {
    const n = this.predictionErrors.length;
    this.meanPredictionError = (this.meanPredictionError * n + error) / (n + 1);
    this.predictionErrors.push(error);
  }
}

class ExperienceTrie {
  constructor(maxDepth = 10) {
    if (maxDepth < 1) throw new Error(`max_depth must be >= 1, got ${maxDepth}`);
    this._root = new TrieNode(-1);
    this._maxDepth = maxDepth;
  }

  insert(path, predictionError = 0) {
    if (!path || path.length === 0) return;
    if (path.length > this._maxDepth) path = path.slice(0, this._maxDepth);
    let node = this._root;
    for (let depth = 0; depth < path.length; depth++) {
      const stateId = path[depth];
      if (!node.children[stateId]) {
        node.children[stateId] = new TrieNode(stateId, depth + 1);
      }
      node = node.children[stateId];
      node.visitCount++;
    }
    node.addPredictionError(predictionError);
  }

  lookup(path) {
    let node = this._root;
    for (const stateId of path) {
      if (!node.children[stateId]) return null;
      node = node.children[stateId];
    }
    return node;
  }

  predictNext(path) {
    const node = this.lookup(path);
    if (!node || Object.keys(node.children).length === 0) return null;
    let bestState = null;
    let bestCount = -1;
    for (const [stateId, child] of Object.entries(node.children)) {
      if (child.visitCount > bestCount) {
        bestCount = child.visitCount;
        bestState = parseInt(stateId);
      }
    }
    return bestState;
  }

  _dfsPaths(node, current, minVisits, results, leavesOnly) {
    if (node !== this._root && node.visitCount >= minVisits) {
      if (!leavesOnly || Object.keys(node.children).length === 0) {
        results.push([...current]);
      }
    }
    for (const [childState, child] of Object.entries(node.children)) {
      current.push(parseInt(childState));
      this._dfsPaths(child, current, minVisits, results, leavesOnly);
      current.pop();
    }
  }

  getAllPaths(minVisits = 1) {
    const results = [];
    this._dfsPaths(this._root, [], minVisits, results, false);
    return results;
  }

  getLeafPaths(minVisits = 1) {
    const results = [];
    this._dfsPaths(this._root, [], minVisits, results, true);
    return results;
  }

  compress(minVisits) {
    const _prune = (node) => {
      const toDelete = [];
      for (const [cs, child] of Object.entries(node.children)) {
        if (_prune(child)) toDelete.push(parseInt(cs));
      }
      for (const cs of toDelete) delete node.children[cs];
      if (node !== this._root && node.visitCount < minVisits && Object.keys(node.children).length === 0) {
        return true;
      }
      return false;
    };
    _prune(this._root);
  }

  merge(other) {
    const merged = new ExperienceTrie(Math.max(this._maxDepth, other._maxDepth));
    const pathData = new Map();

    const collect = (node, path) => {
      for (const [cs, child] of Object.entries(node.children)) {
        path.push(parseInt(cs));
        const key = path.join(',');
        const existing = pathData.get(key) || [0, []];
        existing[0] += child.visitCount;
        existing[1] = existing[1].concat(child.predictionErrors);
        pathData.set(key, existing);
        collect(child, path);
        path.pop();
      }
    };

    collect(this._root, []);
    collect(other._root, []);

    for (const [keyStr, [visitCount, predErrors]] of pathData) {
      const path = keyStr.split(',').map(Number);
      merged.insert(path);
      const node = merged.lookup(path);
      if (node) {
        node.visitCount = visitCount;
        node.predictionErrors = predErrors;
        node.meanPredictionError = predErrors.length > 0
          ? predErrors.reduce((a, b) => a + b, 0) / predErrors.length
          : 0;
      }
    }
    return merged;
  }

  _collectPaths(node, currentPath, minVisits, results) {
    if (node !== this._root && node.visitCount >= minVisits) {
      results.push({ path: [...currentPath], visitCount: node.visitCount, meanPredictionError: node.meanPredictionError, wordBinding: node.wordBinding });
    }
    for (const [stateId, child] of Object.entries(node.children)) {
      currentPath.push(parseInt(stateId));
      this._collectPaths(child, currentPath, minVisits, results);
      currentPath.pop();
    }
  }

  getStats() {
    let nodeCount = 0, maxDepth = 0, totalVisits = 0, visitSum = 0;
    const depthDist = {};
    const walk = (node, depth) => {
      nodeCount++;
      maxDepth = Math.max(maxDepth, depth);
      totalVisits += node.visitCount;
      visitSum += node.visitCount;
      depthDist[depth] = (depthDist[depth] || 0) + 1;
      for (const child of Object.values(node.children)) walk(child, depth + 1);
    };
    walk(this._root, 0);
    return { nodeCount, maxDepth, totalVisits, meanVisitCount: nodeCount > 0 ? totalVisits / nodeCount : 0, depthDistribution: depthDist };
  }

  exportNodes(minVisits = 1) {
    const results = [];
    this._collectPaths(this._root, [], minVisits, results);
    return results;
  }

  getDominantPaths(topK = 10) {
    const all = this.exportNodes(1);
    all.sort((a, b) => b.visitCount - a.visitCount);
    return all.slice(0, topK);
  }

  size() {
    let count = 0;
    const countNodes = (node) => {
      count++;
      for (const child of Object.values(node.children)) countNodes(child);
    };
    countNodes(this._root);
    return count;
  }

  compressionRatio(totalSteps) {
    if (totalSteps === 0) return 0;
    return this.size() / totalSteps;
  }

  get root() { return this._root; }
  get maxDepth() { return this._maxDepth; }

  clear() {
    this._root = new TrieNode(-1);
  }
}

module.exports = { ExperienceTrie, TrieNode };
