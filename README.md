# FN-DBSCAN: Fuzzy Neighborhood DBSCAN

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scikit-learn compatible implementation of **Fuzzy Neighborhood DBSCAN (FN-DBSCAN)**, a density-based clustering algorithm that extends classic DBSCAN using fuzzy set theory.

## Why FN-DBSCAN?

Traditional DBSCAN treats all neighbors within epsilon radius equally. FN-DBSCAN uses fuzzy membership functions to weight neighbors by distance, providing:

- **Better boundary detection** - Gradual membership instead of hard cutoffs
- **Improved robustness** - More stable across parameter variations
- **Flexible clustering** - Multiple fuzzy functions for different data types
- **Scikit-learn compatible** - Drop-in replacement for DBSCAN

## Installation

```bash
git clone https://github.com/onurceldir123/fn-dbscan.git
cd fn-dbscan
pip install -e .
```

**Requirements:** Python ≥3.8, NumPy, scikit-learn, scipy

## Quick Start

```python
import numpy as np
from fn_dbscan import FN_DBSCAN

# Your data
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Cluster with FN-DBSCAN
model = FN_DBSCAN(
    eps=0.3,
    min_fuzzy_neighbors=2.0,
    fuzzy_function='exponential',
    normalize=True
)
labels = model.fit_predict(X)

print(f"Found {model.n_clusters_} clusters")
# Found 2 clusters
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | float | 0.5 | Maximum neighborhood radius (0-1 for normalized data) |
| `min_fuzzy_neighbors` | float | 5.0 | Minimum fuzzy cardinality to be a core point (like `min_samples` in DBSCAN) |
| `min_membership` | float | 0.0 | Minimum membership threshold (0-1, use 0 to include all neighbors) |
| `fuzzy_function` | str | 'linear' | Membership function: `'linear'`, `'exponential'`, or `'trapezoidal'` |
| `normalize` | bool | True | Normalize data (strongly recommended) |
| `k` | float | None | Membership function shape parameter (auto-calculated if None) |
| `metric` | str | 'euclidean' | Distance metric (any scikit-learn metric) |

### Fuzzy Functions

- **`'exponential'`** - Recommended for most cases, especially non-convex clusters
- **`'linear'`** - Simple linear decay, good for well-separated clusters
- **`'trapezoidal'`** - Maintains full membership for very close points

## Usage Examples

### With scikit-learn datasets

```python
from sklearn.datasets import make_moons
from fn_dbscan import FN_DBSCAN

# Non-convex clusters
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

model = FN_DBSCAN(
    eps=0.2,
    min_fuzzy_neighbors=5,
    fuzzy_function='exponential'
)
labels = model.fit_predict(X)
```

### In scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fn_dbscan import FN_DBSCAN

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clustering', FN_DBSCAN(eps=0.5, min_fuzzy_neighbors=5))
])

labels = pipeline.fit_predict(X)
```

### Comparing fuzzy functions

```python
for func in ['linear', 'exponential', 'trapezoidal']:
    model = FN_DBSCAN(eps=0.5, min_fuzzy_neighbors=5, fuzzy_function=func)
    labels = model.fit_predict(X)
    print(f"{func}: {model.n_clusters_} clusters")
```

## Model Attributes

After fitting, the model provides:

- **`labels_`** - Cluster labels for each sample (-1 for noise)
- **`core_sample_indices_`** - Indices of core points
- **`n_clusters_`** - Number of clusters found

## Algorithm Overview

FN-DBSCAN extends DBSCAN by computing fuzzy cardinality instead of crisp point counts:

```
Traditional DBSCAN:  cardinality = count(neighbors)
FN-DBSCAN:          cardinality = Σ membership(distance(p, q))
```

A point is a **core point** if its fuzzy cardinality ≥ `min_fuzzy_neighbors`.

**Complexity:** O(n log n) with KD-tree (low dimensions), O(n²) worst case.

## Citation

If you use FN-DBSCAN in your research, please cite the original paper:

```bibtex
@article{nasibov2009robustness,
  title={Robustness of density-based clustering methods with various neighborhood relations},
  author={Nasibov, Efendi N and Ulutagay, G{\"o}zde},
  journal={Fuzzy Sets and Systems},
  volume={160},
  number={24},
  pages={3601--3615},
  year={2009},
  publisher={Elsevier}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/onurceldir123/fn-dbscan).

---

**Reference:** Nasibov, E. N., & Ulutagay, G. (2009). Robustness of density-based clustering methods with various neighborhood relations. *Fuzzy Sets and Systems*, 160(24), 3601-3615.
