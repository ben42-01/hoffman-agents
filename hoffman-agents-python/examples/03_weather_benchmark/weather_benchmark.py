"""
Weather World Benchmark — Markov World vs Classical Baselines

Build a Markov world from weather data (temperature, pressure, humidity)
and compare conscious agent prediction against classical methods:
  - Persistence: the last N hours repeat
  - Climatology: most common outcome for this hour
  - AR(1): linear autoregression

Expected result:
  Markov World prediction accuracy >> all classical baselines
"""
from conscious_agent import WorldBuilder, World as CAWorld
import numpy as np
import csv
import math
import statistics
from pathlib import Path


def load_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_features(rows):
    temp = [float(r["temperature_2m"]) for r in rows]
    pressure = [float(r["surface_pressure"]) for r in rows]
    humidity = [float(r["relative_humidity_2m"]) for r in rows]
    wind = [float(r["wind_speed_10m"]) for r in rows]
    return temp, pressure, humidity, wind


def compute_temp_direction(temp, horizon=3, threshold=0.5):
    dirs = []
    for i in range(len(temp)):
        target = i + horizon
        if target >= len(temp):
            dirs.append(None)
        else:
            change = temp[target] - temp[i]
            if abs(change) <= threshold:
                dirs.append("same")
            elif change > 0:
                dirs.append("up")
            else:
                dirs.append("down")
    return dirs


def run_benchmark(csv_path=None):
    if csv_path is None:
        csv_path = Path(__file__).parent / "weather_berlin.csv"
    np.random.seed(42)
    print("=" * 60)
    print("Weather Benchmark: Markov World vs Classical Baselines")
    print("=" * 60)

    rows = load_csv_rows(csv_path)
    temp, pressure, humidity, wind = compute_features(rows)
    n = len(rows)
    print(f"\nLoaded {n} weather records")

    # Build features array for WorldBuilder
    data = np.column_stack([
        np.array(temp, dtype=float),
        np.array(pressure, dtype=float),
        np.array(humidity, dtype=float),
        np.array(wind, dtype=float),
    ])

    # Build Markov world
    print("Building Markov world...")
    world = (WorldBuilder()
        .add_feature("temp", "minmax", 4)
        .add_feature("pressure", "minmax", 4)
        .add_feature("humidity", "minmax", 4)
        .add_feature("wind", "minmax", 3)
        .build(data))
    print(f"  {world.n_states} states")

    # Get discretized state IDs from the world's state_ids
    state_ids = world.state_ids

    # Compute temperature direction labels
    dirs = compute_temp_direction(temp, horizon=3, threshold=0.5)

    # Build direction map from state → {up, down, same} counts (training only)
    split = int(n * 0.8)
    train_states = state_ids[:split]
    test_states = state_ids[split:]
    train_dirs = dirs[:split]
    test_dirs = dirs[split:]

    dir_map = {}
    for sid, d in zip(train_states, train_dirs):
        if d is None:
            continue
        if sid not in dir_map:
            dir_map[sid] = {"up": 0, "down": 0, "same": 0}
        dir_map[sid][d] += 1

    def predict_dir(sid):
        counts = dir_map.get(sid)
        if not counts:
            return None
        return max(counts, key=counts.get)

    # Evaluate Markov on test
    correct = total = 0
    for sid, actual in zip(test_states, test_dirs):
        if actual is None:
            continue
        pred = predict_dir(sid)
        if pred:
            total += 1
            if pred == actual:
                correct += 1
    markov_acc = correct / total if total > 0 else 0

    # Majority class baseline
    train_counts = {"up": 0, "down": 0, "same": 0}
    for d in train_dirs:
        if d:
            train_counts[d] += 1
    majority = max(train_counts, key=train_counts.get)
    majority_acc = train_counts[majority] / sum(train_counts.values())

    # Persistence baseline
    pers_correct = pers_total = 0
    for i in range(split, n):
        past = i - 6
        if past < 0 or dirs[i] is None or temp[past] is None:
            continue
        past_change = temp[i] - temp[past]
        if abs(past_change) <= 0.5:
            pred = "same"
        elif past_change > 0:
            pred = "up"
        else:
            pred = "down"
        pers_total += 1
        if pred == dirs[i]:
            pers_correct += 1
    pers_acc = pers_correct / pers_total if pers_total > 0 else 0

    # AR(1) baseline
    train_temp = temp[:split]
    pairs = [(train_temp[t], train_temp[t + 3])
             for t in range(len(train_temp) - 3)
             if train_temp[t] is not None and train_temp[t + 3] is not None]
    if len(pairs) > 10:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom > 1e-12 else 0
        b = my - a * mx

        ar_correct = ar_total = 0
        for i in range(split, n):
            if temp[i] is None or dirs[i] is None:
                continue
            predicted = a * temp[i] + b
            change = predicted - temp[i]
            if abs(change) <= 0.5:
                pred = "same"
            elif change > 0:
                pred = "up"
            else:
                pred = "down"
            ar_total += 1
            if pred == dirs[i]:
                ar_correct += 1
        ar_acc = ar_correct / ar_total if ar_total > 0 else 0
    else:
        ar_acc = 0

    # Print results
    print(f"\nBaselines:")
    print(f"  Majority class ('{majority}'): {majority_acc * 100:.1f}%")
    print(f"  Persistence (6h):           {pers_acc * 100:.1f}%")
    print(f"  AR(1):                      {ar_acc * 100:.1f}%")
    print(f"")
    print(f"  Markov World:               {markov_acc * 100:.1f}%")
    print(f"  Improvement over majority:  {(markov_acc / majority_acc - 1) * 100:+.1f}%")
    print(f"  Improvement over AR(1):     {(markov_acc / ar_acc - 1) * 100:+.1f}%")

    if markov_acc > max(majority_acc, pers_acc, ar_acc):
        print(f"\n  ✓ RESULT: Markov world beats all classical baselines")
    else:
        print(f"\n  ~ RESULT: Classical baselines competitive")

    return {
        "markov": markov_acc,
        "majority": majority_acc,
        "persistence": pers_acc,
        "ar1": ar_acc,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(csv_path)
