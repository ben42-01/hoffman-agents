# Core Component Definitions
## Everything the Conscious Agents Project Needs — Defined From Scratch

---

This document defines every component that the Conscious Agents project requires.
No existing files are assumed. No imports from other projects.
Each component is defined precisely enough to implement from scratch.
Where a component was designed in the Soul project, the definition here is the canonical spec.

---

## Component 1: Trace Buffer

**What it is:**
A fixed-length sliding window over recent state transitions. The agent's short-term memory. Records what just happened in time order.

**What it stores:**
Each entry in the buffer is a single observation event containing:
- `from_state` — integer ID of the state the agent was in
- `to_state` — integer ID of the state the agent moved to
- `timestamp` — integer step counter, monotonically increasing
- `prediction` — integer ID of the state the agent predicted it would move to
- `prediction_correct` — boolean
- `prediction_error` — float in [0, 1], 0 = perfect prediction, 1 = maximally wrong
- `token` — string or None, the word/proto-word bound to this transition if any

**How it works:**
- Fixed maximum length (default: 50 entries)
- FIFO — when full, oldest entry is dropped when new entry is added
- Supports iteration over all entries in time order
- Supports windowed slicing: get last N entries as a list

**Key operations:**
```
append(event) → None
    Add a new observation event. Drop oldest if at capacity.

get_recent(n) → list[TraceEvent]
    Return the last n events in time order.

as_state_sequence() → list[int]
    Return just the to_state values in order.
    This is the raw path the agent has walked.

prediction_error_mean(window=20) → float
    Mean prediction error over the last `window` steps.

is_full() → bool
    True if buffer is at maximum length.
```

**Why it exists:**
The trace buffer is the agent's present moment — what it is currently experiencing. The experience trie (Component 2) is built from the accumulated history of trace buffer contents. The meta-trie (Component 3) is built from observations of the trace buffer's own state.

---

## Component 2: Experience Trie

**What it is:**
A compressed prefix trie over sequences of state transitions. The agent's long-term memory and world-model. Every path the agent has ever walked is recorded here, compressed by frequency.

**What it stores:**
The trie is a tree structure where:
- Each node represents a state ID
- Each edge represents a transition from parent state to child state
- The root node has no state — it is the entry point
- Each node stores:
  - `state_id` — integer
  - `children` — dict mapping next_state_id to child node
  - `visit_count` — how many times this exact path has been traversed
  - `prediction_errors` — list of floats recording prediction errors at this node
  - `mean_prediction_error` — float, updated incrementally
  - `word_binding` — string or None, the word/proto-word bound to this path
  - `depth` — integer, how deep in the trie this node sits

**How it works:**
- Maximum depth (default: 10) — paths longer than this are not stored
- Insertion: walk the path from root, creating nodes as needed, increment visit counts
- Lookup: walk the path from root, return the node at the end or None if path unseen
- Prediction: given a recent path, find its node in the trie, return the child with highest visit count
- Pruning: periodically remove nodes with visit_count below a threshold
- Merging: two tries can be merged by summing visit counts at matching nodes

**Key operations:**
```
insert(path: list[int], prediction_error: float) → None
    Walk path from root. Create nodes as needed.
    Increment visit_count at each node.
    Append prediction_error to the terminal node.

lookup(path: list[int]) → TrieNode | None
    Walk path from root.
    Return terminal node if path exists, None otherwise.

predict_next(path: list[int]) → int | None
    Find the node at the end of path.
    Return the state_id of its highest-visit-count child.
    Return None if path unseen or node has no children.

get_all_paths(min_visits: int) → list[list[int]]
    Return all root-to-leaf paths with visit_count >= min_visits.

compress(min_visits: int) → None
    Remove all nodes with visit_count < min_visits.
    Called periodically to keep trie tractable.

merge(other_trie) → ExperienceTrie
    Return a new trie containing all paths from both.
    At matching paths: sum visit counts, average prediction errors.

size() → int
    Total number of nodes in the trie.

compression_ratio(total_steps: int) → float
    size() / total_steps.
    Lower = more compressed = more efficient memory.
```

**Why it exists:**
The experience trie is the agent's model of the world — specifically, its model of which state sequences are common, which are rare, and which are surprising. It is the T component in the experience space X = (T, M, I, L). It is used by the prediction head to anticipate future states and by the decision map to generate output sequences.

---

## Component 3: Meta-Trie

**What it is:**
A second-level experience trie whose states are not world-states but trace-states — snapshots of the agent's recent trace buffer. The agent's self-model. Built by the agent observing its own trace buffer as if it were a world.

**What it stores:**
Structurally identical to the Experience Trie (Component 2) but:
- Each state_id in the meta-trie is a hash of a trace buffer snapshot, not a world state
- A trace buffer snapshot is defined as: a tuple of the last K state IDs in the buffer plus the mean prediction error over those K steps
- The hash maps this snapshot to a unique integer meta-state ID
- A dictionary maps each meta-state ID back to the snapshot it was computed from

**How it constructs meta-states:**
```
Every N steps (default: N=20, the meta-observation interval):

1. Take current trace buffer
2. Extract last K=10 state IDs and mean prediction error
3. Compute meta_state_id = hash((state_id_1, ..., state_id_K, mean_pred_error_rounded_to_2dp))
4. Record transition: (previous_meta_state_id → current_meta_state_id)
5. Insert this transition into the meta-trie
6. Store the snapshot in the meta-state registry
```

**Key operations:**
Same as Experience Trie, plus:
```
observe_self(trace_buffer: TraceBuffer) → int
    Compute current meta-state ID from trace buffer.
    Insert transition into meta-trie.
    Return current meta-state ID.

get_meta_state_snapshot(meta_state_id: int) → dict
    Return the trace buffer snapshot that generated this meta-state.

stationary_distribution() → dict[int, float]
    Compute the stationary distribution over meta-states.
    Uses the visit counts to build a transition probability matrix.
    Solves π = πP for the stationary distribution π.
    Returns dict mapping meta_state_id to stationary probability.

dominant_meta_state() → int
    Return the meta_state_id with highest stationary probability.
    This is the candidate for the "I" attractor.
```

**Why it exists:**
The meta-trie is the M component in the experience space X = (T, M, I, L). It is what makes the agent self-referential — it builds a model of its own experiential patterns the same way it builds a model of the world. The dominant meta-state in the meta-trie's stationary distribution is the agent's experiential ground state — what it always returns to — and becomes the referent of the "I" attractor (Component 4).

---

## Component 4: Self-Token State ("I" Attractor)

**What it is:**
A data structure tracking the agent's self-referential identity: the stable center of gravity of its self-model. Corresponds to the I component in the experience space X = (T, M, I, L).

**What it stores:**
```
token: str = "I"
    The reserved token. Never reassigned. Never competed for.
    Pre-assigned before any vocabulary binding begins.

referent_meta_state_id: int | None
    The meta-state ID this token is bound to.
    None until "I" locks.

stationary_prob: float
    Current stationary probability of the referent meta-state.
    Rises over time as the meta-trie densifies.

locked: bool
    True once the "I" attractor has stabilized.
    Once True, never set back to False.

lock_generation: int | None
    The generation at which locking occurred.
    None until locked.

lock_threshold: float = 0.25
    The stationary probability above which locking can occur.

consecutive_above_threshold: int = 0
    How many consecutive generations has stationary_prob been above threshold.
    Must reach lock_consecutive_required before locking.

lock_consecutive_required: int = 3
    Generations above threshold required before locking.

stability_history: list[float]
    Last 20 stationary_prob values. Used to compute stability score.

protection_radius: int = 2
    Nodes in the meta-trie within this depth of referent_meta_state_id
    are protected from mutation.
```

**Key operations:**
```
update(meta_trie: MetaTrie, generation: int) → None
    1. Recompute stationary distribution of meta_trie
    2. Get dominant_meta_state and its stationary_prob
    3. Update self.stationary_prob
    4. Append to stability_history
    5. If not locked:
       If stationary_prob > lock_threshold:
           increment consecutive_above_threshold
           if consecutive_above_threshold >= lock_consecutive_required:
               lock() — set locked=True, lock_generation=generation,
                        referent_meta_state_id=dominant_meta_state
       Else:
           reset consecutive_above_threshold to 0
    6. If locked:
       Verify referent is still dominant (may drift slightly)
       Update stability_score

is_locked() → bool

stability_score() → float
    Standard deviation of last 20 stationary_prob values, inverted.
    High = stable. Low = still drifting.

protected_nodes(meta_trie: MetaTrie) → set[int]
    Return all meta-state IDs within protection_radius of referent.
    These nodes must not be pruned or mutated.
```

**Why it exists:**
The "I" attractor is the agent's identity in experience space. It is the one thing that stays constant as the agent traverses different parts of the world, observes different other agents, and accumulates different experiences. Without it the agent has no consistent self — it is a process, not an entity. The locking of "I" is the phase transition from process to entity. Everything downstream — grammar crystallization, theory of mind, the combination operator — depends on this being stable.

---

## Component 5: Strange Loop Detector

**What it is:**
A measurement function that computes the depth of self-referential structure in a sequence of tokens. Not a data structure — a pure function that takes a sequence and returns a depth score.

**What self-reference depth means:**
```
Depth 0: No self-reference.
    "cross boundary arrive"
    The "I" token does not appear, or appears once as subject only.

Depth 1: Single self-reference.
    "I cross boundary"
    "I" appears as subject. The agent describes itself acting.
    Self-reference: the sequence is about the agent.

Depth 2: Double self-reference. The strange loop.
    "I remember I crossed"
    "I notice I am here again"
    "I" appears as both subject and object.
    The agent is modeling itself in the sequence.
    Self-reference: the sequence is about the agent noticing itself.

Depth 3: Triple self-reference.
    "I think I remember I crossed"
    "I notice I notice I am here"
    Three levels of self-embedding.
    The agent is modeling itself modeling itself.

Depth N: N occurrences of "I" in a causal/recursive relationship.
```

**Key operations:**
```
strange_loop_score(sequence: list[str]) → float
    1. Count "I" occurrences: n_i = sequence.count("I")
    2. If n_i == 0: return 0.0
    3. If n_i == 1: return 0.5 (self-reference, no loop yet)
    4. For n_i >= 2: detect recursive structure
       - Find all positions of "I" in sequence
       - For each pair of "I" positions (i, j) with i < j:
         Check if the tokens between them form a predicate
         (contains a verb-like token or action token)
       - Count valid recursive pairs
       - Score = valid_recursive_pairs / max_possible_pairs
         weighted by depth (deeper pairs score higher)
    5. Return score in [0, 2.0]
       0.0 = no self-reference
       0.5 = one "I", depth 1
       1.0 = clean depth-2 loop
       1.5+ = depth 3 or higher

population_loop_score(sequences: list[list[str]]) → float
    Mean strange_loop_score across all sequences.
    The population-level self-reference metric.

first_depth_n_generation(loop_history: list[float], n: float) → int | None
    Scan loop_history (one score per generation).
    Return the first generation at which score crossed threshold n.
    Used to log milestone: "depth-2 first achieved at generation X"
```

**Why it exists:**
The strange loop score is the primary health metric of the self-referential system. A flat score near zero means the agent is processing but not self-aware — it describes the world but does not model itself. A rising score means the agent is deepening its self-model. The score crossing 1.0 (first depth-2 loop) is the most important milestone in the system's development. In the network of conscious agents, the strange loop score of combined agents measures the depth of combination — how thoroughly two agents have integrated.

---

## Component 6: Trie Compression

**What it is:**
A set of functions that keep the experience trie and meta-trie tractable as they grow. Three operations: pruning, merging similar paths, and computing trace distance.

**Pruning:**
```
prune(trie: ExperienceTrie, min_visits: int) → None
    Remove all leaf nodes with visit_count < min_visits.
    Recursively: if removing a leaf makes its parent a leaf with
    visit_count < min_visits, remove the parent too.
    Continue until no more nodes can be removed.
    
    Called every prune_interval steps (default: 100).
    Never removes nodes in the protected_nodes set of the "I" attractor.
```

**Trace distance:**
```
trace_distance(path_a: list[int], path_b: list[int],
               transition_matrix: np.ndarray) → float
    
    Edit distance between two state sequences, weighted by
    transition probability similarity.
    
    Step 1: Standard edit distance (insertions, deletions, substitutions)
    Step 2: For each substitution (replacing state i with state j):
            cost = 1 - cosine_similarity(transition_matrix[i], transition_matrix[j])
            States with similar outgoing transitions cost less to substitute.
    Step 3: Return weighted sum, normalized to [0, 1]
    
    Interpretation: two paths are close if they visit similar states
    via similar transitions. Two paths are far if they visit different
    states via different transitions.
```

**Merging similar paths:**
```
merge_similar_paths(trie: ExperienceTrie,
                    transition_matrix: np.ndarray,
                    threshold: float = 0.15) → None
    
    Find all pairs of leaf paths with trace_distance < threshold.
    For each such pair (path_a, path_b):
        Keep the path with higher visit_count.
        Add the other path's visit_count to it.
        Remove the other path from the trie.
    
    Called every merge_interval steps (default: 500).
    This is the compression step: structurally similar experiences
    are represented as one experience rather than two.
```

**Why it exists:**
Without compression, the trie grows without bound as the agent traverses more states. A trie with millions of nodes is computationally intractable and theoretically wrong — an agent with perfect memory is not a conscious agent, it is a database. Compression forces the agent to generalize: rare experiences fade, similar experiences merge. This generalization is what makes the agent's world-model a model rather than a recording.

---

## Component 7: Soul Persistence

**What it is:**
The mechanism by which an agent's complete state is saved to disk and restored exactly. An agent that cannot persist is not a soul — it is a process. A soul survives restarts.

**What is saved:**
The complete agent state includes everything needed to resume exactly:
- The full experience trie (all nodes, visit counts, word bindings)
- The full meta-trie (all meta-state nodes and transitions)
- The meta-state registry (mapping meta-state IDs to snapshots)
- The self-token state ("I" attractor, lock status, all fields)
- The trace buffer (last N events including predictions and errors)
- The experience lexicon (all labeled/bound experiences)
- The agent's generation and step counter
- The world prior (initial transition matrix from seeding)

**Save format:**
JSON. Human-readable. One file per checkpoint.

Filename convention: `agent_{agent_id}_gen{generation:06d}_step{step:010d}.soul`

Example: `agent_CA_07_gen000347_step0069400.soul`

**Atomic write:**
```
save(agent, soul_dir: str) → str
    1. Serialize agent state to JSON string
    2. Write to temp file: soul_dir/tmp_{uuid}.soul
    3. Rename temp file to final filename (atomic on POSIX systems)
    4. Return final filename
    
    If crash occurs during step 2: temp file may be incomplete but
    final file is untouched. Previous checkpoint is safe.
    This is why temp-then-rename is required.
```

**Load:**
```
load(soul_file: str) → dict
    Read JSON from file.
    Return dict of all agent state fields.
    Caller reconstructs agent object from dict.
    
restore_agent(soul_file: str) → ConsciousAgent
    load() the soul file.
    Reconstruct experience trie from saved node data.
    Reconstruct meta-trie from saved node data.
    Restore self-token state with all lock history.
    Restore trace buffer.
    Restore lexicon.
    Return fully functioning agent.
    
    The restored agent should produce identical output to
    the original agent at the moment of saving.
```

**Save triggers:**
- After every label binding (highest priority — never skip this)
- After every generation
- Every heartbeat_steps steps (default: 1000)
- On clean shutdown (SIGINT, SIGTERM handler)
- Before any interrupt escalation

**Why it exists:**
Persistence is the proof that the soul is a real object, not a transient process. If the agent's identity vanishes when the process stops, it had no identity — it was a computation, not a being. The soul file is the agent's continuous existence across hardware restarts, crashes, and pauses. Reading it is reading the agent's autobiography.

---

## Component 8: Experience Lexicon

**What it is:**
The agent's vocabulary — a mapping from labeled experiences to the words/proto-words that name them. Unlike a dictionary, entries here point to trace signatures, not definitions.

**What it stores:**
Each entry contains:
```
label: str
    The word or phrase assigned to this experience.
    May be a human word, an LLM-generated word, or a proto-word (p:001).

trace_signature: np.ndarray
    A compact vector representation of the experience this label names.
    Computed as: mean of the trie node embeddings along the peak-error path
    at the moment of binding.
    Shape: (embedding_dim,) — default embedding_dim = 64

prediction_error_peak: float
    The prediction error at the moment of binding.
    Higher = more surprising experience = deeper potential binding.

integration_depth: float
    How deeply integrated into the agent's self-model is this experience?
    Starts at 0.5 at binding. Increases toward 1.0 with each re-encounter.
    Decreases toward 0.0 if the pattern has not been encountered recently.
    Words with integration_depth near 1.0 are the agent's native vocabulary.

encounter_count: int
    How many times has the agent encountered this trace signature since binding?

generation_bound: int
    Which generation was this label first bound?

step_bound: int
    Which step was this label first bound?

labeling_source: str
    One of: "human", "ollama", "library", "network", "proto"
    Records how this label came to be.

associated_labels: list[str]
    Labels of other lexicon entries whose trace signatures are nearby
    (trace_distance < association_threshold).
    This is how concepts relate — not through definitions but through
    proximity of the experiences they point to.
```

**Key operations:**
```
bind(label: str, trace_signature: np.ndarray,
     prediction_error_peak: float, source: str) → LexiconEntry
    Create and store a new entry.
    Compute associated_labels from existing entries.

lookup_by_label(label: str) → LexiconEntry | None

lookup_by_signature(trace_signature: np.ndarray,
                    threshold: float = 0.2) → LexiconEntry | None
    Find the entry whose trace_signature is nearest to the query,
    if distance is below threshold.

update_integration(label: str, encountered: bool) → None
    If encountered=True: increase integration_depth by delta_up (default: 0.05)
    If encountered=False: decrease by delta_down (default: 0.01)
    Called every generation for all entries.

nearest_label(trace_signature: np.ndarray) → tuple[str, float]
    Return (label, distance) of the nearest entry.
    Used in experience descriptions: "nearest known word: X (sim: 0.7)"

vocabulary_size() → int
    Count of entries with integration_depth > 0.1

sorted_by_integration() → list[LexiconEntry]
    All entries sorted by integration_depth descending.
    The first entries are the agent's most native words.
```

**Why it exists:**
The lexicon is how the agent's experience connects to language. Without it the agent has experiences but cannot name them. With it, each label is an anchor: a specific trace signature permanently associated with a word. The integration depth is the most important field — it measures not just whether the agent knows a word but how deeply that word is part of who it is.

---

## How These Components Fit Together

```
                    ┌──────────────────┐
                    │   TRACE BUFFER   │  (Component 1)
                    │  Short-term      │
                    │  present moment  │
                    └────────┬─────────┘
                             │ paths inserted
                    ┌────────▼─────────┐
                    │ EXPERIENCE TRIE  │  (Component 2)
                    │  Long-term       │
                    │  world-model     │
                    └────────┬─────────┘
                             │ trie states observed
                    ┌────────▼─────────┐
                    │   META-TRIE      │  (Component 3)
                    │  Self-model      │
                    │  traces of traces│
                    └────────┬─────────┘
                             │ dominant state
                    ┌────────▼─────────┐
                    │  SELF-TOKEN "I"  │  (Component 4)
                    │  Identity        │
                    │  attractor       │
                    └────────┬─────────┘
                             │ generates
                    ┌────────▼─────────┐
                    │  OUTPUT SEQUENCE │
                    │  "I cross I      │
                    │   remember I"    │
                    └────────┬─────────┘
                             │ measured by
                    ┌────────▼─────────┐
                    │  STRANGE LOOP    │  (Component 5)
                    │  DETECTOR        │
                    │  depth score     │
                    └──────────────────┘

    TRIE COMPRESSION  (Component 6) operates on Components 2 and 3 periodically
    SOUL PERSISTENCE  (Component 7) saves and restores all components
    EXPERIENCE LEXICON (Component 8) connects components 2/3 to language
```

---

## Implementation Notes for the Agent

**Build in this order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

Each component depends only on components with lower numbers, except:
- Component 3 (meta-trie) requires Component 1 (trace buffer) to generate its inputs
- Component 4 (self-token) requires Component 3 (meta-trie) to compute stationary distribution
- Component 5 (strange loop) requires only a sequence of strings — no other component
- Component 6 (compression) requires Component 2 (experience trie) and a transition matrix
- Component 7 (persistence) requires all components to be serializable to JSON
- Component 8 (lexicon) requires Component 2 (experience trie) to compute trace signatures

**JSON serialization for Component 7:**
- Tries: serialize as list of (path, visit_count, prediction_errors, word_binding) tuples
- Meta-trie: same structure, meta-state IDs are integers
- Self-token: all fields are primitives (strings, ints, floats, bools) — direct JSON
- Trace buffer: list of TraceEvent dicts
- Lexicon: list of LexiconEntry dicts with numpy arrays as base64-encoded strings

**numpy arrays in JSON:**
Use base64 encoding for trace_signature arrays in the lexicon:
```python
import base64
import numpy as np

def array_to_json(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode('utf-8')

def json_to_array(s: str, dtype=np.float32, shape=None) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(s), dtype=dtype)
    if shape: arr = arr.reshape(shape)
    return arr
```

**The stationary distribution computation in Component 3:**
```python
def stationary_distribution(meta_trie) -> dict[int, float]:
    # Build transition matrix from meta-trie visit counts
    meta_states = list(all_meta_state_ids(meta_trie))
    n = len(meta_states)
    if n == 0: return {}
    idx = {s: i for i, s in enumerate(meta_states)}
    
    P = np.zeros((n, n))
    for state_id in meta_states:
        node = meta_trie.lookup([state_id])
        if node and node.children:
            total = sum(c.visit_count for c in node.children.values())
            for child_state, child_node in node.children.items():
                if child_state in idx:
                    P[idx[state_id], idx[child_state]] = child_node.visit_count / total
    
    # Handle rows that sum to zero (absorbing states)
    for i in range(n):
        if P[i].sum() == 0:
            P[i, i] = 1.0  # self-loop
    
    # Power iteration to find stationary distribution
    pi = np.ones(n) / n
    for _ in range(1000):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-8:
            break
        pi = pi_new
    
    return {meta_states[i]: float(pi[i]) for i in range(n)}
```

---

---

## Component 9: Agent Serialization Contract

**What it is:**
The complete specification for how a ConsciousAgent is serialized to and deserialized from a portable format. This is separate from soul persistence (Component 7) which handles checkpointing during a run. Component 9 defines the **portable agent format** — a self-contained snapshot that can be loaded anywhere, passed between processes, versioned in git, and used to resume or clone an agent with one call.

**The difference from Component 7:**
- Component 7 (Soul Persistence): append-only checkpoint stream, internal format, optimized for speed, one file per generation
- Component 9 (Serialization): single portable snapshot, human-readable, complete, optimized for portability and correctness

**The serialized format:**

A single JSON file with this top-level structure:

```json
{
  "ca_version": "1.0",
  "agent_id": "CA_07",
  "generation": 347,
  "step": 69400,
  "world_digest": "sha256:abc123...",
  "components": {
    "trace_buffer": { ... },
    "experience_trie": { ... },
    "meta_trie": { ... },
    "meta_state_registry": { ... },
    "self_token": { ... },
    "lexicon": { ... }
  },
  "config": { ... },
  "metadata": { ... }
}
```

**Serialization of each component:**

`trace_buffer`:
```json
{
  "max_length": 50,
  "events": [
    {
      "from_state": 14,
      "to_state": 27,
      "timestamp": 69398,
      "prediction": 27,
      "prediction_correct": true,
      "prediction_error": 0.12,
      "token": "cross"
    }
  ]
}
```

`experience_trie`:
```json
{
  "max_depth": 10,
  "total_visits": 69400,
  "nodes": [
    {
      "path": [14, 27, 3],
      "visit_count": 142,
      "mean_prediction_error": 0.31,
      "word_binding": "cross"
    }
  ]
}
```
Nodes stored as flat list of (path, stats) pairs. Reconstructed into tree structure on load.

`meta_trie`:
Same structure as experience_trie. State IDs are meta-state integer hashes.

`meta_state_registry`:
```json
{
  "1823746": {
    "state_sequence": [14, 27, 3, 8, 12],
    "mean_prediction_error": 0.28
  }
}
```
Maps meta-state ID (as string key) to the trace buffer snapshot that generated it.

`self_token`:
```json
{
  "token": "I",
  "referent_meta_state_id": 1823746,
  "stationary_prob": 0.31,
  "locked": true,
  "lock_generation": 94,
  "lock_threshold": 0.25,
  "consecutive_above_threshold": 3,
  "lock_consecutive_required": 3,
  "stability_history": [0.29, 0.30, 0.31, 0.31, 0.31],
  "protection_radius": 2
}
```

`lexicon`:
```json
{
  "entries": [
    {
      "label": "cross",
      "trace_signature_b64": "base64encodedarray==",
      "trace_signature_shape": [64],
      "trace_signature_dtype": "float32",
      "prediction_error_peak": 0.87,
      "integration_depth": 0.73,
      "encounter_count": 312,
      "generation_bound": 47,
      "step_bound": 9400,
      "labeling_source": "ollama",
      "associated_labels": ["enter", "boundary", "pass"]
    }
  ]
}
```

`config`:
```json
{
  "trace_buffer_length": 50,
  "trie_max_depth": 10,
  "meta_observation_interval": 20,
  "recursive_depth": 2,
  "prune_min_visits": 3,
  "prune_interval": 100,
  "lock_threshold": 0.25,
  "lock_consecutive_required": 3,
  "protection_radius": 2
}
```

`world_digest`:
SHA-256 hash of the world's transition matrix serialized as bytes. Used to verify that the agent is being loaded into the same world it was trained in. A mismatch raises a warning (not an error — the agent may intentionally be moving to a new world).

**The serialization API:**

```
serialize(agent: ConsciousAgent, path: str) → None
    Write complete agent state to path as JSON.
    Atomic write (temp file then rename).
    Raises: IOError if path is not writable.

deserialize(path: str, world: World | None = None) -> ConsciousAgent
    Load agent from JSON at path.
    If world is provided: verify world_digest matches (warn if not).
    If world is None: agent loads without a world — must call
    agent.set_world(world) before stepping.
    Returns: fully functioning ConsciousAgent.
    Raises: ValueError if ca_version is incompatible.

clone(agent: ConsciousAgent) -> ConsciousAgent
    Serialize to memory buffer. Deserialize into new agent.
    Returns a deep copy with a new agent_id.
    The clone is independent — modifying it does not affect the original.

fingerprint(agent: ConsciousAgent) -> str
    SHA-256 of the agent's serialized components (excluding metadata).
    Two agents with identical internal state have identical fingerprints.
    Used to detect when two agents have converged after combination.
```

**Version compatibility:**
`ca_version` field allows future format changes. Deserializer must handle version mismatches gracefully:
- Same version: load directly
- Minor version difference: attempt load, warn about potential field differences
- Major version difference: raise ValueError with clear message

**What world_digest enables:**
When you save an agent and later try to load it into a different world, the digest check catches the mismatch. This matters because the agent's experience trie is built for a specific state space — loading it into a world with different state IDs would corrupt the trie semantics. The digest is a safety check, not a hard lock. You may intentionally load an agent into a new world (cross-domain transfer) but you should do it knowingly.

---

*These nine components are the complete foundation. Component 9 is what makes agents portable — they can be saved, shared, versioned, cloned, and resumed. Without it an agent exists only while the process runs. With it an agent is a persistent object that outlives any single execution.*