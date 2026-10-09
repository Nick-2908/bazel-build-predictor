# Predicting Bazel Build Times Using Machine Learning

UE24CS352A — Machine Learning Mini-Project  

---

## Problem Statement

Bazel is Google's open-source build system, used by millions of developers daily. A single build can time out on deeply structured codebases, wasting compute and developer time. This project trains a machine learning model to predict the **CPU time** (in milliseconds) required to complete a Bazel build after a given commit, using only commit metadata — specifically the **most common file path prefix** and **most common file type** changed in that commit.

---

## Approach

We replicate and extend the methodology from:  
> Nadeem & Raza, *"Predicting Bazel Build Times Using Machine Learning"*, Stanford CS229

### Models trained
| Model | Feature Crossing | Test RMSE |
|---|---|---|
| Linear Regression | ✗ | ~2217 ms |
| Linear Regression | ✓ | ~1878 ms |
| Ridge Regression | ✓ | ~1317 ms |
| XGBoost | ✓ | ~1323 ms |

Feature crossing (prefix × type) consistently reduces RMSE by **~20%** over plain linear regression, matching the paper's findings.

---

## Dataset

- **Source**: Synthetic dataset generated to match the statistical properties of the `bazelbuild/bazel` GitHub repository (500 commits)
- **Split**: 300 train / 100 validation / 100 test (paper's exact split)
- **Input features**:
  - `most_common_prefix`: dominant file path prefix in the commit (e.g., `src/main/`, `third_party/`)
  - `most_common_type`: dominant file type (e.g., `JAVA`, `C/C++`, `Starlark`)
- **Target**: `cpu_time_ms` — simulated CPU build time in milliseconds

---

## Setup

```bash
# Clone and install
git clone <repo-url>
cd bazel-build-predictor
pip install -r requirements.txt

# Generate dataset
python src/generate_dataset.py

# Train all models and generate plots
python src/train_model.py
```

**Requirements**: Python 3.9+, see `requirements.txt`

---

## Project Structure

```
bazel-build-predictor/
├── data/
│   ├── train.csv          # 300 training samples
│   ├── val.csv            # 100 validation samples
│   ├── test.csv           # 100 test samples
│   └── period_rmse.csv    # Per-period RMSE (paper's table format)
├── plots/
│   ├── linreg_no_cross_train_val.png
│   ├── linreg_cross_train_val.png
│   ├── model_comparison.png
│   └── ...
├── src/
│   ├── generate_dataset.py   # Synthetic data generation
│   └── train_model.py        # All models + evaluation + plots
├── requirements.txt
└── README.md
```

---

## Key Results

- **Feature crossing** reduces test RMSE from ~2217ms to ~1878ms (~15% improvement)
- **Ridge** and **XGBoost** achieve further improvements (~1320ms) with regularization
- No overfitting observed: train and validation RMSE curves converge across all 9 periods
- Consistent with the paper: the `prefix × type` interaction captures non-linear build time dependencies that flat linear regression cannot

---

## References

Nadeem, M. U., & Raza, S. *Predicting Bazel Build Times Using Machine Learning*. Stanford University CS229.
