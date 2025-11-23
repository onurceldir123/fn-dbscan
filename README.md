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
    eps=0.3,                    # ε: Neighborhood radius (for normalized data: 0-1)
    epsilon1=0.0,               # ε₁: Membership threshold (0 = no filtering)
    epsilon2=2.0,               # ε₂: Minimum fuzzy cardinality (like MinPts)
    fuzzy_function='exponential', # Membership function type
    k=5,                        # k: Shape parameter
    normalize=True              # Normalize data (recommended)
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
    eps=0.2,            # ε: Neighborhood radius
    epsilon1=0.0,       # ε₁: No membership filtering
    epsilon2=5.0,       # ε₂: Minimum fuzzy cardinality
    fuzzy_function='exponential',
    k=2,                # k: Lower values for gradual membership decay
    normalize=True
)
labels = model.fit_predict(X)

print(f"Found {model.n_clusters_} clusters")
```

## Parameters

### FN_DBSCAN

- **eps** (float, default=0.5): Maximum distance for neighborhood (ε in the paper). For normalized data, this should be in [0, 1]. Points within this distance are considered potential neighbors.

- **epsilon1** (float, default=0.0): Minimum membership threshold (ε₁ or α-cut level in the paper). Points with membership degree < epsilon1 are not considered neighbors. Should be in [0, 1]. Use 0.0 to include all points within eps radius.

- **epsilon2** (float, default=5.0): Minimum fuzzy cardinality for a point to be classified as a core point (ε₂ in the paper). This is the fuzzy equivalent of DBSCAN's `min_samples`.

- **fuzzy_function** ({'linear', 'exponential', 'trapezoidal'}, default='linear'): The fuzzy membership function to use:
  - `'linear'`: μ(d) = max(0, 1 - k·d/d_max) where k = d_max/ε
  - `'exponential'`: μ(d) = exp(-(k·d/d_max)²) where k is user-defined
  - `'trapezoidal'`: μ(d) = 1 if d ≤ ε/2, else 2(1-d/ε)

- **metric** (str or callable, default='euclidean'): Distance metric to use. Can be any metric supported by sklearn.metrics.pairwise.

- **k** (float or None, default=None): Parameter that controls the shape of the fuzzy membership function. If None, it's automatically calculated:
  - For linear: k = d_max / eps
  - For exponential: k = 20 (recommended by the paper)
  Higher k values make the membership function steeper. Recommended values: 1-5 for gradual decay, 15-20 for steep decay.

- **normalize** (bool, default=True): Whether to normalize the data so that maximum distance ≤ 1. This is recommended in the paper to make the eps parameter scale-independent.

## Attributes

After fitting, the following attributes are available:

- **labels_** (ndarray of shape (n_samples,)): Cluster labels for each point. Noisy samples are given the label -1.

- **core_sample_indices_** (ndarray): Indices of core samples.

- **n_clusters_** (int): Number of clusters found (excluding noise).

## Fuzzy Membership Functions

Based on the paper by Nasibov & Ulutagay (2009), FN-DBSCAN supports three fuzzy membership functions:

### Linear
```python
μ(d) = max(0, 1 - k·d/d_max)
```
where k = d_max / ε (auto-calculated) or user-defined.

Simple linear decay. Higher k values create steeper decay.
- **Recommended k**: Auto (d_max/ε) or 1-5 for gradual, 15-20 for steep

### Exponential
```python
μ(d) = exp(-(k·d/d_max)²)
```
Smooth exponential decay. Emphasizes closer neighbors more strongly.
- **Recommended k**: 20 (best results from paper), or 1-10 for more gradual decay
- **Best for**: Non-convex clusters, varying densities

### Trapezoidal
```python
μ(d) = {
    1.0,           if d ≤ ε/2
    2(1 - d/ε),    if ε/2 < d ≤ ε
    0.0,           if d > ε
}
```
Plateau region with full membership for very close neighbors, then linear decay.
- **Best for**: Clear dense cores with defined boundaries

## Examples

### Basic Usage

```python
from sklearn.datasets import make_blobs
from fn_dbscan import FN_DBSCAN

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Fit FN-DBSCAN
model = FN_DBSCAN(eps=0.7, epsilon2=5)
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
fn_dbscan = FN_DBSCAN(eps=0.2, epsilon2=5, fuzzy_function='linear')
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
    ('clustering', FN_DBSCAN(eps=0.5, epsilon2=5))
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
        epsilon2=5,
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
