# FN-DBSCAN: Fuzzy Neighborhood DBSCAN

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **Fuzzy Neighborhood DBSCAN (FN-DBSCAN)**, a density-based clustering algorithm that extends classic DBSCAN using fuzzy theory.

## Installation

```bash
pip install fn-dbscan
```

For development:
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

## Why FN-DBSCAN?

While classic DBSCAN is powerful, it relies on a "crisp" boundary—a point is either a neighbor or it isn't. FN-DBSCAN improves upon this by introducing fuzzy set theory:

* [cite_start]**Robustness to Density Variations:** It is more robust than DBSCAN when handling datasets with varying densities and shapes[cite: 19, 61].
* **Soft Boundaries:** Instead of an all-or-nothing approach, it calculates a "fuzzy cardinality" (sum of membership degrees). [cite_start]This handles border points and noise more naturally [cite: 153-155].
* [cite_start]**Scale Invariance:** The implementation includes the normalization technique proposed in the paper, making the `eps` parameter adaptable to the data scale [cite: 127-141].
* [cite_start]**Best of Both Worlds:** Combines the speed of DBSCAN with the robustness of fuzzy clustering methods like NRFJP[cite: 18, 60].

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | float | 0.5 | Maximum neighborhood radius (0-1 for normalized data). |
| `min_fuzzy_neighbors` | float | 5.0 | Minimum fuzzy cardinality to be a core point (analogous to `min_samples` in DBSCAN). |
| `min_membership` | float | 0.0 | Minimum membership threshold ($\epsilon_1$). Points with membership below this are ignored. |
| `fuzzy_function` | str | 'linear' | Membership function: `'linear'`, `'exponential'`, or `'trapezoidal'`. |
| `normalize` | bool | True | Normalize data to make `eps` scale-independent (Strongly Recommended). |
| `k` | float | None | **Steepness parameter.** Controls how fast membership drops. Higher $k$ = stricter neighborhood. Auto-calculated as $d_{max}/\epsilon$ if None. |
| `metric` | str | 'euclidean' | Distance metric (any scikit-learn compatible metric). |

### Fuzzy Functions

- **`'exponential'`** - Recommended for most cases, especially non-convex clusters
- **`'linear'`** - Simple linear decay, good for well-separated clusters
- **`'trapezoidal'`** - Maintains full membership for very close points


## Model Attributes

After fitting, the model provides:

- **`labels_`** - Cluster labels for each sample (-1 for noise)
- **`core_sample_indices_`** - Indices of core points
- **`n_clusters_`** - Number of clusters found

## Algorithm Overview

FN-DBSCAN extends DBSCAN by computing fuzzy cardinality instead of discrete point counts:

```
Traditional DBSCAN:  cardinality = count(neighbors)
FN-DBSCAN:          cardinality = Σ membership(distance(p, q))
```

A point is a **core point** if its fuzzy cardinality ≥ `min_fuzzy_neighbors`.

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
