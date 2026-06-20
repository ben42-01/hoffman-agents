# Setup Guide

This library works with both **uv** (recommended) and **pip**.

## Quick Start (uv — recommended)

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone or copy this directory, then:
cd hoffman-agents-python

# 3. Create venv and install everything
uv sync --group dev

# 4. Verify it works
uv run pytest tests/ -v

# 5. Run an example
uv run python examples/01_fitness_beats_truth/fitness_beats_truth.py
```

## Quick Start (pip)

```bash
cd hoffman-agents-python

# 3 ways to set up (pick one):

# Option A — standard venv
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest   # for tests

# Option B — with pip
pip install -e .[dev]

# Option C — bare minimum (runtime only)
pip install -e .

# Verify
python -m pytest tests/ -v
```

## Testing

```bash
# uv
uv run pytest tests/ -v

# pip
python -m pytest tests/ -v
```

## Running Examples

```bash
# Fitness Beats Truth (Hoffman's Interface Theory)
uv run python examples/01_fitness_beats_truth/fitness_beats_truth.py

# Quantum Signature (spectral analysis of combined agents)
uv run python examples/02_quantum_signature/quantum_signature.py

# Weather Benchmark (Markov world vs classical baselines)
uv run python examples/03_weather_benchmark/weather_benchmark.py data/weather_berlin.csv

# Stop Lights Dashboard (web UI at http://localhost:8765)
uv run python examples/04_stop_lights/stop_lights.py
```

## Publishing to PyPI

```bash
# Build
uv build

# Publish (requires PyPI credentials)
uv publish
```

## Requirements

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10
- That's it — no pandas, no pytorch, no sklearn
