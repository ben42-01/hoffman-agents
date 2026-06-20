const SELF_TOKEN = 'I';

const PREDICATE_TOKENS = new Set([
  'remember', 'notice', 'think', 'see', 'feel', 'know',
  'cross', 'arrive', 'return', 'enter', 'leave',
  'expect', 'predict', 'observe', 'experience', 'am',
]);

function computeSelfReferenceScore(sequence) {
  const nI = sequence.filter(t => t === SELF_TOKEN).length;

  if (nI === 0) return 0;
  if (nI === 1) return 0.5;

  const positions = sequence.map((t, i) => t === SELF_TOKEN ? i : -1).filter(i => i >= 0);
  let validPairs = 0;

  for (let idx = 0; idx < positions.length; idx++) {
    for (let jdx = idx + 1; jdx < positions.length; jdx++) {
      const i = positions[idx], j = positions[jdx];
      const between = sequence.slice(i + 1, j);
      const hasPredicate = between.some(t => PREDICATE_TOKENS.has(t));
      if (hasPredicate) validPairs++;
    }
  }

  if (validPairs === 0) return 0.5;
  const maxDepth = positions.length;
  const score = 0.5 * maxDepth * (validPairs / (nI * (nI - 1) / 2));
  return Math.min(score, 2.0);
}

function populationReferenceScore(sequences) {
  if (!sequences || sequences.length === 0) return 0;
  return sequences.reduce((s, seq) => s + computeSelfReferenceScore(seq), 0) / sequences.length;
}

const strangeLoopScore = computeSelfReferenceScore;
const populationLoopScore = populationReferenceScore;

function firstDepthNGeneration(loopHistory, threshold) {
  for (let gen = 0; gen < loopHistory.length; gen++) {
    if (loopHistory[gen] >= threshold) return gen;
  }
  return null;
}

module.exports = {
  computeSelfReferenceScore,
  populationReferenceScore,
  strangeLoopScore,
  populationLoopScore,
  firstDepthNGeneration,
  SELF_TOKEN,
};
