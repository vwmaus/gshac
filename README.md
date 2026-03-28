# GSHAC

**Geographically Sparse Hierarchical Agglomerative Clustering**

Exact hierarchical clustering for spatial data, using a sparse geographic distance graph instead of the O(n^2) dense distance matrix. For any cut height h <= h_max, GSHAC produces results **identical** to standard HAC (fastcluster/scipy) while requiring only O(n*k) time and memory, where k is the mean neighbourhood size within h_max.

An default C extension provides optimised union-find linkage, haversine distances, and batch dendrogram cutting. The library has a fallback pure Python implementation, but the C is recommended for large datasets.

## Installation

Requires Python >= 3.10 and a C compiler (for the optional extension).

```bash
# Clone and install
git clone https://github.com/vwmaus/gshac.git
cd gshac
pip install -e .

# Build the C extension (optional but recommended for performance)
python3 setup.py build_ext --inplace
```

The library falls back to pure Python if the C extension is not built.

### Extras

```bash
pip install -e ".[plot]"   # + matplotlib for dendrogram visualisation
pip install -e ".[dev]"    # + pytest + all optional deps
```

## Quick start

### scikit-learn API

```python
from gshac import SpatialAgglomerativeClustering
import numpy as np

coords = np.random.default_rng(0).uniform(0, 500_000, size=(5_000, 2))

model = SpatialAgglomerativeClustering(
    h_max=10_000,              # sparse graph radius (metres)
    distance_threshold=5_000,  # cut height (metres)
    linkage="single",
)
model.fit(coords)
print(model.labels_)       # cluster labels
print(model.n_clusters_)   # number of clusters
```

### Low-level API

```python
from gshac import spatial_dist_graph, sparse_hclust

graph = spatial_dist_graph(coords, h_max=10_000)
result = sparse_hclust(graph, h_cuts=[5_000, 10_000], method="single")
# result["labels"][5000] -> cluster labels at 5 km cut
```

### Dendrograms

```python
from gshac import sparse_hclust, spatial_dist_graph, plot_dendrogram

graph = spatial_dist_graph(coords, h_max=10_000)
result = sparse_hclust(graph, h_cuts=[5_000], return_linkage=True)
plot_dendrogram(result, color_threshold=5_000)
```

Or from a fitted estimator:

```python
plot_dendrogram(model)
```

Per-component dendrograms (avoids the infinity stitching):

```python
from gshac import plot_component_dendrograms
plot_component_dendrograms(result, top_k=4)
```

### Linkage matrix access

The `return_linkage=True` flag returns per-component scipy linkage matrices.
Use `stitch_linkage()` to combine them into a single `(n-1, 4)` matrix
compatible with `scipy.cluster.hierarchy.dendrogram` and `fcluster`:

```python
from gshac import stitch_linkage
from scipy.cluster.hierarchy import dendrogram, fcluster

Z = stitch_linkage(result)
labels = fcluster(Z, t=5_000, criterion="distance")
dendrogram(Z)
```

The sklearn estimator also exposes `children_`, `distances_`, and
`linkage_matrix_` after fitting.

## Running tests

```bash
pip install -e ".[dev]"
python3 setup.py build_ext --inplace
pytest tests/ -v
```

## API reference

| Function / Class | Description |
|---|---|
| `spatial_dist_graph(coords, h_max, metric)` | Build sparse distance graph (CSR matrix) |
| `sparse_hclust(graph, h_cuts, method, return_linkage)` | Component-wise HAC on the sparse graph |
| `stitch_linkage(result)` | Combine per-component Z matrices into one |
| `SpatialAgglomerativeClustering` | sklearn-compatible estimator |
| `geographic_connectivity(coords, h_max, metric)` | Binary connectivity matrix for sklearn |
| `plot_dendrogram(model_or_result, ...)` | Plot dendrogram from estimator or result |
| `plot_component_dendrograms(result, top_k)` | Plot dendrograms for largest components |

## License

GPL-3.0. See [LICENSE](LICENSE).
