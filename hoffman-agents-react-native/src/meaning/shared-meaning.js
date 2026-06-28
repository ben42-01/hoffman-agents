class SharedMeaningTracker {
  constructor() {
    this._epochs = [];
  }

  snapshot(agents, generation) {
    const allTokens = {};
    const allLabels = {};
    for (const [aid, lex] of Object.entries(agents)) {
      const tokens = new Set();
      for (const entry of lex._entries.values()) {
        if (entry.outputToken) tokens.add(entry.outputToken);
      }
      allTokens[aid] = tokens;
      allLabels[aid] = new Set(lex._entries.keys());
    }

    const sharedness = this._computeSharedness(allTokens);
    const epoch = { generation, sharedness, tokenOverlap: this._computeOverlap(allTokens) };
    this._epochs.push(epoch);
    return epoch;
  }

  _computeSharedness(allTokens) {
    const agentIds = Object.keys(allTokens);
    if (agentIds.length < 2) return 0;

    let totalPairs = 0, sharedPairs = 0;
    for (let i = 0; i < agentIds.length; i++) {
      for (let j = i + 1; j < agentIds.length; j++) {
        const setI = allTokens[agentIds[i]];
        const setJ = allTokens[agentIds[j]];
        const union = new Set([...setI, ...setJ]);
        if (union.size === 0) continue;
        totalPairs++;
        const intersection = new Set([...setI].filter(x => setJ.has(x)));
        sharedPairs += intersection.size / union.size;
      }
    }
    return totalPairs > 0 ? sharedPairs / totalPairs : 0;
  }

  _computeOverlap(allTokens) {
    const counts = {};
    for (const tokens of Object.values(allTokens)) {
      for (const token of tokens) counts[token] = (counts[token] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
  }
}

module.exports = { SharedMeaningTracker };
