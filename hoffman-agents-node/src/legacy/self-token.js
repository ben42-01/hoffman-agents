// 2.x I-lock rule, preserved verbatim for mathVersion: 'legacy'.
// Locks when max stationary probability exceeds a fixed threshold for a few
// observations; never unlocks.
function update(st, metaTrie, generation) {
  const dist = metaTrie.stationaryDistribution();
  if (dist.size === 0) return;

  let dominant = null, prob = 0;
  for (const [id, p] of dist) {
    if (p > prob) { prob = p; dominant = id; }
  }
  prob = Math.min(prob, 1.0);

  st.stationaryProb = prob;
  st.stabilityHistory.push(prob);
  if (st.stabilityHistory.length > 20) st.stabilityHistory.shift();

  if (!st.locked) {
    if (prob > st.lockThreshold) {
      st.consecutiveAboveThreshold++;
      if (st.consecutiveAboveThreshold >= st.lockConsecutiveRequired) {
        st.locked = true;
        st.referentMetaStateId = dominant;
        st.lockGeneration = generation;
      }
    } else {
      st.consecutiveAboveThreshold = 0;
    }
  } else {
    st.referentMetaStateId = dominant;
  }
}

module.exports = { update };
