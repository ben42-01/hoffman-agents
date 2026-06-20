#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Setup ──────────────────────────────────────────────────────
uv sync --group dev 2>/dev/null
RUN="uv run python"

# ── Examples ────────────────────────────────────────────────────
EXAMPLES=(
  "01_fitness_beats_truth:fitness_beats_truth.py:Fitness Beats Truth — Interface Theory"
  "02_quantum_signature:quantum_signature.py:Quantum Signature — Spectral Analysis"
  "03_weather_benchmark:weather_benchmark.py:Weather Benchmark — Markov v Classical"
  "04_stop_lights:stop_lights.py:Stop Lights — Web Dashboard (localhost:8765)"
  "05_self_ref_ablation:self_ref_ablation.py:Self-Reference ON/OFF Contrast"
)

# ── Dispatch ────────────────────────────────────────────────────
if [ $# -ge 1 ]; then
  # Run specific example by number or partial name
  for entry in "${EXAMPLES[@]}"; do
    dir="${entry%%:*}"
    rest="${entry#*:}"
    file="${rest%%:*}"
    label="${rest#*:}"
    if [[ "$dir" == *"$1"* ]] || [[ "$(basename "$dir")" == "0$1"* ]]; then
      echo ""
      echo "═══════════════════════════════════════════════════════"
      echo "  $label"
      echo "═══════════════════════════════════════════════════════"
      $RUN "examples/$dir/$file"
      exit $?
    fi
  done
  echo "Unknown example: $1"
  echo "Usage: $0 [number|name]    — run specific example"
  echo "       $0 all              — run all examples"
  exit 1
fi

# ── Run all ─────────────────────────────────────────────────────
for entry in "${EXAMPLES[@]}"; do
  dir="${entry%%:*}"
  rest="${entry#*:}"
  file="${rest%%:*}"
  label="${rest#*:}"

  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  $label"
  echo "═══════════════════════════════════════════════════════"

  if [[ "$dir" == "04_stop_lights" ]]; then
    echo "  Starting server on http://localhost:8765"
    echo "  Press Ctrl+C to stop"
    $RUN "examples/$dir/$file" &
    PID=$!
    sleep 2
    echo "  Server running (PID $PID). Opening..."
    if command -v xdg-open &>/dev/null; then
      xdg-open "http://localhost:8765" 2>/dev/null || true
    elif command -v open &>/dev/null; then
      open "http://localhost:8765" 2>/dev/null || true
    fi
    wait $PID 2>/dev/null || true
  else
    $RUN "examples/$dir/$file"
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All examples complete."
echo "═══════════════════════════════════════════════════════"

exit 0