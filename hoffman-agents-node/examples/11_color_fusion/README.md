# Color Fusion — Consciousness Merger

A visual experiment in consciousness fusion. Primary agents (Red, Green, Blue) each perceive one band of the color spectrum. When they combine via the ⊗ operator, their perceived colors **additively merge** — the same way actual photons combine to produce new colors. A combined agent literally perceives a color that neither parent alone could see.

The invisible spectrum (UV, IR) represents forms of consciousness that exist but that human perception cannot directly access. They interact with the same world — they respond to the same wavelengths — but their experience is outside our perceptual range. When they combine with visible-spectrum agents, they produce colors that have no name in the visible spectrum.

## The Metaphor

| Agent | Color | Represents |
|-------|-------|-----------|
| Red | #FF0000 | A primary consciousness — one band of experience |
| Green | #00FF00 | Another primary, orthogonal to Red |
| Blue | #0000FF | A third primary, completing the visible triad |
| UV | #8800FF | A consciousness outside human perception (ultraviolet) |
| IR | #FF0088 | Another invisible consciousness (infrared) |
| Yellow (Red + Green) | #FFFF00 | Emergent consciousness — neither parent alone could see this |
| Magenta (Red + Blue) | #FF00FF | Another emergent |
| Cyan (Green + Blue) | #00FFFF | Another emergent |
| White (R + G + B) | #FFFFFF | Full visible-spectrum consciousness |
| UV+Red | hybrid | Visible + invisible blended — a color without a name |
| IR+Green | hybrid | Another unnamed color |
| Omni (Full Spectrum) | hybrid | All wavelengths, visible and invisible, unified |

## How It Works

Each agent trains on a Markov world where states represent colors. The world's transition matrix favors transitions between spectrally adjacent colors — Red tends toward Orange and Violet, Blue toward Cyan and Violet, etc. But an agent only *learns from* states that match its own color band. A Red agent builds a trie of only Red transitions; a Blue agent builds a trie of only Blue transitions.

When agents combine, the merged experience space inherits both parents' tries. The resulting agent perceives a **blended** world — and the dashboard renders that blend as an additive RGB color.

This is not a physics claim about how actual consciousness works. It's a visual metaphor for a structural truth: **when two systems of experience merge, the result is not an average, not a toggle, not a compromise — it is a genuinely new thing, with properties neither parent possessed alone.**

## The Dashboard

Open `dashboard.html` (serve it with `local-server.js` from the repo root) to see a living grid of colored squares:

- Each square = one agent's perceived color
- Glowing squares = agents with locked identity (stable "I")
- The spectrum bar shows the full range of visible + invisible wavelengths
- Run the experiment, then refresh the dashboard to see updated data

## Running

### Live dashboard (recommended)

```bash
cd hoffman-agents-node/examples/11_color_fusion/
node start.js
# → http://localhost:5000
```

The dashboard is fully self-contained — the agents learn and colors shift in real time in your browser. No build step, no dependencies.

### Or from the repo root

```bash
npm run examples:colorFusion
```

## Caveats

1. The colors shown are illustrative — the agent's internal state is a hash of WorldState tokens, not an actual RGB value. The mapping from agent identity to hex color is a human-interpretable visualization, not a claim about how the agent "experiences" color.
2. Additive mixing (light model, not paint) is used because it corresponds to how actual photons combine — and because it produces the familiar RYB→secondary colors that make the metaphor intuitive.
3. The invisible-spectrum agents (UV, IR) are for the purposes of the metaphor; they do not have any special internal structure that differs from visible-spectrum agents. Their "invisibility" is a statement about human perceptual limits, not a claim about a different kind of consciousness.
