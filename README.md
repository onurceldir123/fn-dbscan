# FN-DBSCAN: Fuzzy Neighborhood DBSCAN

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of the **Fuzzy Neighborhood DBSCAN (FN-DBSCAN)** clustering algorithm with a scikit-learn compatible API.

## Overview

FN-DBSCAN is a density-based clustering algorithm that extends the classic DBSCAN by using fuzzy set theory to define neighborhood cardinality. Instead of counting discrete points within an epsilon radius, FN-DBSCAN computes a fuzzy cardinality based on a membership function that decreases with distance.

This approach provides:
- More nuanced cluster assignment, especially for border points
- Better handling of varying densities
- Flexibility through different fuzzy membership functions
- Smooth transition between core and noise points

## Features

- **Scikit-learn compatible API**: Drop-in replacement for DBSCAN in existing pipelines
- **Multiple fuzzy membership functions**: Linear, exponential, and trapezoidal
- **Efficient implementation**: Uses KD-trees for fast neighbor search
- **Comprehensive testing**: >90% test coverage
- **Well-documented**: Extensive docstrings and examples

## Installation

### From source

```bash
git clone https://github.com/onurceldir123/fn-dbscan.git
cd fn-dbscan
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.19.0
- scikit-learn >= 1.0.0
- scipy >= 1.5.0

## Quick Start

### Basic Example

```python
import numpy as np
from fn_dbscan import FN_DBSCAN

# Create sample data
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Create and fit the model
model = FN_DBSCAN(
    eps=0.3,                      # ε: Neighborhood radius (for normalized data: 0-1)
    min_membership=0.0,           # Minimum fuzzy membership threshold (0 = no filtering)
    min_fuzzy_neighbors=2.0,      # Minimum fuzzy cardinality (like MinPts in DBSCAN)
    fuzzy_function='exponential', # Membership function type
    k=5,                          # k: Shape parameter
    normalize=True                # Normalize data (recommended)
)
labels = model.fit_predict(X)

print(f"Cluster labels: {labels}")
print(f"Number of clusters: {model.n_clusters_}")
```

### With scikit-learn datasets

```python
from sklearn.datasets import make_moons
from fn_dbscan import FN_DBSCAN

# Create non-convex dataset
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Cluster with exponential membership function (recommended in paper)
model = FN_DBSCAN(
    eps=0.2,                 # ε: Neighborhood radius
    min_membership=0.0,      # No membership filtering
    min_fuzzy_neighbors=5.0, # Minimum fuzzy cardinality
    fuzzy_function='exponential',
    k=2,                     # k: Lower values for gradual membership decay
    normalize=True
)
labels = model.fit_predict(X)

print(f"Found {model.n_clusters_} clusters")
```

## Parameters

### `FN_DBSCAN`

The main class for Fuzzy Neighborhood DBSCAN clustering.

- **`eps`** : *float, default=0.5* Maximum distance for neighborhood ($\epsilon$ in the paper). For normalized data, this should be in $[0, 1]$. Points within this distance are considered potential neighbors.

- **`min_membership`** : *float, default=0.0* Minimum fuzzy membership threshold ($\epsilon_1$ or $\alpha$-cut level in the paper). Points with membership degree $\mu(d) <$ `min_membership` are not considered neighbors.
  *Range:* $[0, 1]$. Use `0.0` to include all points within the `eps` radius.
  *Formerly called:* `epsilon1`

- **`min_fuzzy_neighbors`** : *float, default=5.0* Minimum fuzzy cardinality for a point to be classified as a core point ($\epsilon_2$ in the paper). This is the fuzzy equivalent of standard DBSCAN's `min_samples`.
  *Formerly called:* `epsilon2` or `min_cardinality`

- **`fuzzy_function`** : *{'linear', 'exponential', 'trapezoidal'}, default='linear'* The fuzzy membership function $\mu(d)$ to use for calculating neighborhood density:
  
  - `'linear'`: $\mu(d) = \max(0, 1 - \frac{k \cdot d}{d_{max}})$ where $k = d_{max}/\epsilon$
  - `'exponential'`: $\mu(d) = \exp\left(-\left(\frac{k \cdot d}{d_{max}}\right)^2\right)$ where $k$ is user-defined
  - `'trapezoidal'`: $\mu(d) = 1$ if $d \le \epsilon/2$, else $2(1 - d/\epsilon)$

- **`metric`** : *str or callable, default='euclidean'* The distance metric to use. Can be any metric supported by `sklearn.metrics.pairwise`.

- **`k`** : *float or None, default=None* Parameter that controls the steepness/shape of the fuzzy membership function.
  - **If `None` (Auto):** $k = d_{max} / \epsilon$ for all fuzzy functions
    - Automatically adapts to data scale and neighborhood radius
    - The paper suggests k=20 for exponential, but dynamic calculation provides better adaptability
  - **Manual:** Higher $k$ values create a steeper decay.
    - *Recommended:* `1-5` for gradual decay, `15-20` for steep decay.

- **`normalize`** : *bool, default=True* Whether to normalize the data so that the maximum distance is $\le 1$. This is strongly recommended to make the `eps` parameter scale-independent.

---

## Attributes

After fitting the model, the following attributes are available:

- **`labels_`** : *ndarray of shape (n_samples,)* Cluster labels for each point in the dataset. Noisy samples are given the label `-1`.

- **`core_sample_indices_`** : *ndarray* Indices of the core samples (points that satisfy the $\epsilon_2$ density condition).

- **`n_clusters_`** : *int* Estimated number of clusters found (excluding noise).

## Fuzzy Membership Functions

Based on the paper by **Nasibov & Ulutagay (2009)**, FN-DBSCAN supports three fuzzy membership functions:

### 1. Linear
A simple linear decay. Higher $k$ values create a steeper decay curve.

$$
\mu(d) = \max\left(0, 1 - \frac{k \cdot d}{d_{max}}\right)
$$

Where $k = d_{max} / \epsilon$ (auto-calculated) or user-defined.

* **Recommended $k$:** Auto ($d_{max}/\epsilon$) or `1-5` for gradual decay, `15-20` for steep decay.

---

### 2. Exponential
Smooth exponential decay. Emphasizes closer neighbors more strongly than the linear function.

$$
\mu(d) = \exp\left(-\left(\frac{k \cdot d}{d_{max}}\right)^2\right)
$$

* **Recommended $k$:** Auto ($d_{max}/\epsilon$) for dynamic adaptation, or `20` as suggested in the paper for fixed behavior. Manual values: `1-10` for gradual decay.
* **Best for:** Non-convex clusters and datasets with varying densities.

---

### 3. Trapezoidal
Features a plateau region with full membership for very close neighbors, followed by a linear decay.

$$
\mu(d) =
\begin{cases}
1.0 & \text{if } d \le \epsilon/2 \\
2\left(1 - \frac{d}{\epsilon}\right) & \text{if } \epsilon/2 < d \le \epsilon \\
0.0 & \text{if } d > \epsilon
\end{cases}
$$

* **Best for:** Data with clear dense cores and defined boundaries.
## Examples

### Basic Usage

```python
from sklearn.datasets import make_blobs
from fn_dbscan import FN_DBSCAN

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Fit FN-DBSCAN
model = FN_DBSCAN(eps=0.7, min_fuzzy_neighbors=5)
labels = model.fit_predict(X)

print(f"Found {model.n_clusters_} clusters")
print(f"Noise points: {sum(labels == -1)}")
```

### Comparison with DBSCAN

```python
from sklearn.cluster import DBSCAN
from fn_dbscan import FN_DBSCAN
from sklearn.datasets import make_moons

# Generate non-convex data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Standard DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# FN-DBSCAN with linear membership
fn_dbscan = FN_DBSCAN(eps=0.2, min_fuzzy_neighbors=5, fuzzy_function='linear')
labels_fn = fn_dbscan.fit_predict(X)

print(f"DBSCAN found {len(set(labels_dbscan)) - 1} clusters")
print(f"FN-DBSCAN found {fn_dbscan.n_clusters_} clusters")
```

### Using in Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fn_dbscan import FN_DBSCAN

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clustering', FN_DBSCAN(eps=0.5, min_fuzzy_neighbors=5))
])

# Fit pipeline
labels = pipeline.fit_predict(X)
```

### Different Fuzzy Functions

```python
from fn_dbscan import FN_DBSCAN

# Try different fuzzy membership functions
for fuzzy_func in ['linear', 'exponential', 'trapezoidal']:
    model = FN_DBSCAN(
        eps=0.5,
        min_fuzzy_neighbors=5,
        fuzzy_function=fuzzy_func
    )
    labels = model.fit_predict(X)
    print(f"{fuzzy_func}: {model.n_clusters_} clusters, "
          f"{sum(labels == -1)} noise points")
```

## Algorithm Details

### Key Differences from DBSCAN

1. **Fuzzy Cardinality**: Instead of counting neighbors, FN-DBSCAN sums fuzzy memberships:
   ```
   cardinality(p) = Σ μ(dist(p, q)) for all q in N_ε(p)
   ```

2. **Core Point Definition**: A point p is core if `cardinality(p) >= epsilon2` (ε₂)

3. **Soft Boundaries**: Border points receive graded membership based on distance

### Complexity

- **Time**: O(n log n) with KD-tree (low dimensions), O(n²) worst case
- **Space**: O(n)

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=fn_dbscan --cov-report=html
```

## Performance

FN-DBSCAN performance is comparable to standard DBSCAN:

| Dataset Size | FN-DBSCAN | DBSCAN |
|-------------|-----------|--------|
| 1,000 points | <1s | <1s |
| 10,000 points | ~5s | ~3s |
| 100,000 points | ~60s | ~40s |

*Benchmarked on Intel i5, 8GB RAM*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{nasibov2009robustness,
  title={Robustness of density-based clustering methods with various neighborhood relations},
  author={Nasibov, Efendi N and Ulutagay, Gokhan},
  journal={Fuzzy Sets and Systems},
  volume={160},
  number={24},
  pages={3601--3615},
  year={2009},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Nasibov, E. N., & Ulutagay, G. (2009). Robustness of density-based clustering methods with various neighborhood relations. *Fuzzy Sets and Systems*, 160(24), 3601-3615.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 96(34), 226-231.

## Roadmap

- [ ] Parallel processing support (n_jobs parameter)
- [ ] Incremental/online learning (partial_fit)
- [ ] Additional fuzzy membership functions
- [ ] Cython optimization for critical loops
- [ ] GPU acceleration support

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/onurceldir123/fn-dbscan/issues) on GitHub.

## Acknowledgments

- Original algorithm by Nasibov & Ulutagay (2009)
- Inspired by scikit-learn's DBSCAN implementation
- Built with NumPy and scikit-learn
