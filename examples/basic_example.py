# """Simple FN-DBSCAN clustering example with synthetic data.

# This example demonstrates the basic usage of FN-DBSCAN on a simple
# synthetic dataset with three well-separated clusters.
# """
from sklearn.datasets import make_moons
from fn_dbscan import FN_DBSCAN

X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

model = FN_DBSCAN(
    eps=0.1,                  
    min_fuzzy_neighbors=5.0,    
    min_membership=0.0,        
    fuzzy_function='exponential', 
    normalize=True              
)

labels = model.fit_predict(X)

print(f"Found {model.n_clusters_} clusters")
