class LexiconEntry {
  constructor(label, outputToken, traceSignature, {
    predictionErrorPeak = 0,
    integrationDepth = 0.5,
    encounterCount = 0,
    generationBound = 0,
    stepBound = 0,
    labelingSource = 'proto',
    associatedLabels = [],
  } = {}) {
    this.label = label;
    this.outputToken = outputToken;
    this.traceSignature = traceSignature;
    this.predictionErrorPeak = predictionErrorPeak;
    this.integrationDepth = integrationDepth;
    this.encounterCount = encounterCount;
    this.generationBound = generationBound;
    this.stepBound = stepBound;
    this.labelingSource = labelingSource;
    this.associatedLabels = associatedLabels;
  }
}

class ExperienceLexicon {
  constructor(embeddingDim = 64, associationThreshold = 0.15) {
    this._embeddingDim = embeddingDim;
    this._associationThreshold = associationThreshold;
    this._entries = new Map();
    this._signatures = [];
  }

  _normalizeSignature(sig) {
    let arr;
    if (typeof sig === 'number') {
      arr = new Float64Array(this._embeddingDim).fill(sig);
    } else if (sig.length < this._embeddingDim) {
      arr = new Float64Array(this._embeddingDim);
      arr.set(sig);
    } else {
      arr = sig.slice(0, this._embeddingDim);
    }

    let norm = 0;
    for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < arr.length; i++) arr[i] /= norm;
    return arr;
  }

  _computeAssociations(traceSignature) {
    const dists = [];
    for (const [sig, label] of this._signatures) {
      let d = 0;
      for (let i = 0; i < sig.length; i++) d += (sig[i] - traceSignature[i]) ** 2;
      d = Math.sqrt(d);
      if (d < this._associationThreshold) dists.push([d, label]);
    }
    dists.sort((a, b) => a[0] - b[0]);
    return dists.slice(0, 10).map(([, l]) => l);
  }

  bind(label, traceSignature, {
    predictionErrorPeak = 0,
    source = 'proto',
    generation = 0,
    step = 0,
    outputToken = null,
  } = {}) {
    traceSignature = this._normalizeSignature(traceSignature);
    const associations = this._computeAssociations(traceSignature);
    if (!outputToken) outputToken = label;

    const entry = new LexiconEntry(label, outputToken, traceSignature, {
      predictionErrorPeak,
      generationBound: generation,
      stepBound: step,
      labelingSource: source,
      associatedLabels: associations,
    });

    this._entries.set(label, entry);
    this._signatures.push([traceSignature, label]);
    return entry;
  }

  lookupByLabel(label) { return this._entries.get(label) || null; }

  lookupBySignature(traceSignature, threshold = 0.2) {
    traceSignature = this._normalizeSignature(traceSignature);
    let bestDist = Infinity, bestLabel = null;
    for (const [sig, label] of this._signatures) {
      let d = 0;
      for (let i = 0; i < sig.length; i++) d += (sig[i] - traceSignature[i]) ** 2;
      d = Math.sqrt(d);
      if (d < bestDist) { bestDist = d; bestLabel = label; }
    }
    if (bestLabel && bestDist < threshold) return this._entries.get(bestLabel);
    return null;
  }

  updateIntegration(label, encountered) {
    const entry = this._entries.get(label);
    if (!entry) return;
    if (encountered) {
      entry.integrationDepth = Math.min(entry.integrationDepth + 0.05, 1);
      entry.encounterCount++;
    } else {
      entry.integrationDepth = Math.max(entry.integrationDepth - 0.01, 0);
    }
  }

  nearestLabel(traceSignature) {
    traceSignature = this._normalizeSignature(traceSignature);
    if (this._signatures.length === 0) return ['', 1];
    let bestLabel = '', bestDist = Infinity;
    for (const [sig, label] of this._signatures) {
      let d = 0;
      for (let i = 0; i < sig.length; i++) d += (sig[i] - traceSignature[i]) ** 2;
      d = Math.sqrt(d);
      if (d < bestDist) { bestDist = d; bestLabel = label; }
    }
    return [bestLabel, bestDist];
  }

  vocabularySize() {
    let count = 0;
    for (const entry of this._entries.values()) {
      if (entry.integrationDepth > 0.1) count++;
    }
    return count;
  }

  sortedByIntegration() {
    return [...this._entries.values()].sort((a, b) => b.integrationDepth - a.integrationDepth);
  }

  relabel(oldLabel, newLabel, newOutputToken) {
    const entry = this._entries.get(oldLabel);
    if (!entry) return null;
    if (!newOutputToken) newOutputToken = newLabel;
    this._signatures = this._signatures.filter(([, l]) => l !== oldLabel);
    this._entries.delete(oldLabel);
    entry.label = newLabel;
    entry.outputToken = newOutputToken;
    entry.labelingSource = 'parent';
    this._entries.set(newLabel, entry);
    this._signatures.push([entry.traceSignature, newLabel]);
    return entry;
  }

  remove(label) {
    if (!this._entries.has(label)) return false;
    this._signatures = this._signatures.filter(([, l]) => l !== label);
    this._entries.delete(label);
    return true;
  }

  clear() {
    this._entries.clear();
    this._signatures = [];
  }

  toDict() {
    const entries = [];
    for (const entry of this._entries.values()) {
      entries.push({
        label: entry.label,
        outputToken: entry.outputToken,
        traceSignature: Array.from(entry.traceSignature),
        predictionErrorPeak: entry.predictionErrorPeak,
        integrationDepth: entry.integrationDepth,
        encounterCount: entry.encounterCount,
        generationBound: entry.generationBound,
        stepBound: entry.stepBound,
        labelingSource: entry.labelingSource,
        associatedLabels: [...entry.associatedLabels],
      });
    }
    return { embeddingDim: this._embeddingDim, associationThreshold: this._associationThreshold, entries };
  }

  static fromDict(data) {
    const lex = new ExperienceLexicon(data.embeddingDim || 64, data.associationThreshold || 0.15);
    for (const ed of data.entries || []) {
      const sig = new Float64Array(ed.traceSignature);
      lex.bind(ed.label, sig, {
        predictionErrorPeak: ed.predictionErrorPeak || 0,
        source: ed.labelingSource || 'proto',
        generation: ed.generationBound || 0,
        step: ed.stepBound || 0,
        outputToken: ed.outputToken,
      });
      const entry = lex._entries.get(ed.label);
      if (entry) {
        entry.integrationDepth = ed.integrationDepth || 0.5;
        entry.encounterCount = ed.encounterCount || 0;
      }
    }
    return lex;
  }

  get embeddingDim() { return this._embeddingDim; }
  get entryCount() { return this._entries.size; }
}

module.exports = { ExperienceLexicon, LexiconEntry };
