"""
generate_dataset.py

Generates a synthetic dataset that mirrors the bazelbuild/bazel repo's
commit distribution as described in the Stanford paper:
  - 500 total samples (300 train / 100 val / 100 test)
  - Features: most_common_prefix, most_common_type
  - Target: cpu_time_ms (BEP timingMetric CPUTime)
  - CPU times roughly in the 1400–24000ms range (paper's data column shows
    values like 2440, 7844, 10158, 12167, 1459, 1629, 7754, 23810, 5638, 4021)

The generative model encodes real domain knowledge:
  - C/C++ and heavily-depended-on prefixes (src/main, third_party) → longer builds
  - True prefix × type interaction (feature cross target the paper proposes)
  - Realistic heteroscedastic noise
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Vocabularies (from paper) ─────────────────────────────────────────────────
PREFIXES = [
    "bazelci/", "examples/", "scripts/", "site/",
    "src/conditions/", "src/java_tools/", "src/main/",
    "src/test/", "src/tools/", "third_party/", "tools/"
]

TYPES = ["JAVA", "C/C++", "Starlark", "python", "HTML/CSS/JS"]

# ── Domain-knowledge weights ──────────────────────────────────────────────────
# Base contribution of each file type to CPU time (ms)
TYPE_BASE = {
    "JAVA":        4500,
    "C/C++":       7000,
    "Starlark":    2000,
    "python":      1800,
    "HTML/CSS/JS": 1500,
}

# Multiplicative factor per prefix (reflects dependency centrality)
PREFIX_MULT = {
    "src/main/":       2.0,
    "third_party/":    1.8,
    "src/java_tools/": 1.6,
    "src/conditions/": 1.4,
    "src/tools/":      1.3,
    "tools/":          1.2,
    "src/test/":       1.0,
    "bazelci/":        0.85,
    "scripts/":        0.75,
    "examples/":       0.65,
    "site/":           0.55,
}

# Interaction bonus: certain (prefix, type) combos take extra long
# This is the non-linear signal that feature crossing should capture
INTERACTION_BONUS = {
    ("src/main/",       "C/C++"):   3000,
    ("third_party/",    "C/C++"):   2500,
    ("src/java_tools/", "JAVA"):    2000,
    ("src/main/",       "JAVA"):    1500,
    ("src/conditions/", "C/C++"):   1200,
    ("src/tools/",      "C/C++"):   1000,
}

# Realistic prior over prefix (src/main and src/test dominate bazel repo)
PREFIX_PRIOR = np.array([
    0.04,  # bazelci/
    0.03,  # examples/
    0.04,  # scripts/
    0.02,  # site/
    0.05,  # src/conditions/
    0.08,  # src/java_tools/
    0.25,  # src/main/       ← most common in bazel repo
    0.20,  # src/test/
    0.06,  # src/tools/
    0.12,  # third_party/
    0.11,  # tools/
])
PREFIX_PRIOR /= PREFIX_PRIOR.sum()

# Type prior (JAVA-heavy project, some C++ for core)
TYPE_PRIOR = np.array([0.45, 0.20, 0.18, 0.10, 0.07])
TYPE_PRIOR /= TYPE_PRIOR.sum()


def generate_cpu_time(prefix: str, ftype: str) -> float:
    base = 1200.0  # ms global base
    type_contrib = TYPE_BASE[ftype]
    prefix_factor = PREFIX_MULT[prefix]
    interaction = INTERACTION_BONUS.get((prefix, ftype), 0)

    mean = base + type_contrib * prefix_factor + interaction

    # Heteroscedastic noise: larger builds have more variance
    noise_std = 0.15 * mean + 300
    cpu_time = np.random.normal(mean, noise_std)

    return max(500.0, cpu_time)


def generate_dataset(n: int = 500) -> pd.DataFrame:
    prefixes = np.random.choice(PREFIXES, size=n, p=PREFIX_PRIOR)
    types    = np.random.choice(TYPES,    size=n, p=TYPE_PRIOR)

    cpu_times = np.array([
        generate_cpu_time(p, t) for p, t in zip(prefixes, types)
    ])

    df = pd.DataFrame({
        "most_common_prefix": prefixes,
        "most_common_type":   types,
        "cpu_time_ms":        np.round(cpu_times, 2),
    })
    return df


if __name__ == "__main__":
    os.makedirs("/home/claude/bazel-build-predictor/data", exist_ok=True)

    df = generate_dataset(500)

    # 300 / 100 / 100 split matching the paper
    train = df.iloc[:300]
    val   = df.iloc[300:400]
    test  = df.iloc[400:]

    train.to_csv("/home/claude/bazel-build-predictor/data/train.csv", index=False)
    val.to_csv(  "/home/claude/bazel-build-predictor/data/val.csv",   index=False)
    test.to_csv( "/home/claude/bazel-build-predictor/data/test.csv",  index=False)
    df.to_csv(   "/home/claude/bazel-build-predictor/data/full.csv",  index=False)

    print(f"Generated {len(df)} samples")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"\nCPU time stats (ms):")
    print(df["cpu_time_ms"].describe().round(1))
    print(f"\nPrefix distribution:")
    print(df["most_common_prefix"].value_counts())
    print(f"\nType distribution:")
    print(df["most_common_type"].value_counts())
    print(f"\nSample rows:")
    print(df.head(10).to_string())
