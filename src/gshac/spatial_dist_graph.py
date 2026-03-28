"""
spatial_dist_graph.py

Computes a sparse symmetric distance matrix for hierarchical clustering by
exploiting geographic locality: only pairs of features within a maximum
distance h_max are computed and stored. Pairs beyond h_max are structurally
absent and are treated as infinity by any hierarchical clustering algorithm,
which is correct when cutting the dendrogram at any height h <= h_max.

The algorithm:
  1. Build a spatial index (KD-tree for projected, Ball-tree for geodesic)
  2. For each feature i, query neighbours within h_max -> candidate pairs (i,j)
     with j > i  (upper triangle only)
  3. Compute exact pairwise distances for candidate pairs
  4. Store as scipy sparse symmetric CSR matrix
  5. Find connected components (scipy.sparse.csgraph)

Public API
----------
spatial_dist_graph(coords, h_max, metric)
    Full sparse distance graph (distances stored as edge weights).

geographic_connectivity(coords, h_max, metric)
    Binary connectivity matrix for use with
    sklearn.cluster.AgglomerativeClustering(connectivity=...).

Dependencies: numpy, scipy, scikit-learn, pyproj (optional, for geodesic)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Literal, Optional
import warnings

try:
    from ._gshac import haversine_edges as _c_haversine
    _GSHAC_C = True
except ImportError:
    _GSHAC_C = False


def spatial_dist_graph(
    coords: np.ndarray,
    h_max: float,
    metric: Literal["euclidean", "haversine"] = "euclidean",
) -> dict:
    """
    Build a sparse geographic distance graph.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Feature coordinates. For metric='euclidean': (x, y) in metres (projected
        CRS). For metric='haversine': (lon, lat) in degrees.
    h_max : float
        Maximum distance threshold. Metres for both metric options.
    metric : str
        'euclidean' or 'haversine'. For 'haversine', distances are geodesic
        (WGS-84 approximation via the haversine formula, accurate to ~0.5%).

    Returns
    -------
    dict with keys:
        matrix       : scipy.sparse.csr_matrix (n x n), symmetric distance matrix.
                       Only entries for pairs within h_max are stored.
                       Co-located pairs (dist = 0) are stored as 1.0 (1 metre).
        components   : ndarray of int, shape (n,), connected component labels.
        n_components : int, number of connected components.
        n_edges      : int, number of edges computed (upper triangle).
        density      : float, fill ratio n_edges / (n*(n-1)/2).
    """
    n = len(coords)
    if n < 2:
        raise ValueError("coords must have at least 2 features")

    # ------------------------------------------------------------------
    # Step 1-2: candidate pairs within h_max
    # ------------------------------------------------------------------
    if metric == "euclidean":
        tree = cKDTree(coords)
        # query_pairs returns a set of (i, j) with i < j and ||xi - xj|| <= h_max
        pairs_set = tree.query_pairs(r=h_max, output_type="ndarray")
        if len(pairs_set) == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        else:
            pairs = pairs_set  # already (m, 2) ndarray

    elif metric == "haversine":
        from sklearn.neighbors import BallTree

        # BallTree haversine expects (lat, lon) in radians
        coords_rad = np.radians(coords[:, [1, 0]])  # swap lon/lat -> lat/lon
        EARTH_RADIUS_M = 6_371_000.0
        h_max_rad = h_max / EARTH_RADIUS_M

        tree = BallTree(coords_rad, metric="haversine")
        indices = tree.query_radius(coords_rad, r=h_max_rad)

        # Build pairs as numpy arrays (avoids Python tuple overhead at large n).
        # counts[i] = number of neighbours of point i within h_max.
        counts = np.array([len(js) for js in indices], dtype=np.intp)
        if counts.sum() == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        else:
            i_rep  = np.repeat(np.arange(n, dtype=np.int64), counts)
            j_flat = np.concatenate(list(indices)).astype(np.int64)
            mask   = j_flat > i_rep          # upper triangle only
            pairs  = np.column_stack([i_rep[mask], j_flat[mask]])
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'euclidean' or 'haversine'.")

    if len(pairs) == 0:
        # No edges: every point is its own cluster
        mat = csr_matrix((n, n), dtype=np.float64)
        n_comp, labels = connected_components(mat, directed=False)
        return dict(
            matrix=mat,
            components=labels,
            n_components=n_comp,
            n_edges=0,
            density=0.0,
        )

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    # ------------------------------------------------------------------
    # Step 3: exact distances for candidate pairs
    # ------------------------------------------------------------------
    if metric == "euclidean":
        diffs = coords[i_idx] - coords[j_idx]
        dists = np.linalg.norm(diffs, axis=1)

    elif metric == "haversine":
        lon1 = np.radians(coords[i_idx, 0])
        lat1 = np.radians(coords[i_idx, 1])
        lon2 = np.radians(coords[j_idx, 0])
        lat2 = np.radians(coords[j_idx, 1])
        if _GSHAC_C:
            dists = _c_haversine(lon1, lat1, lon2, lat2, EARTH_RADIUS_M)
        else:
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            dists = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

    # Replace co-located points (dist = 0) with 1 m
    dists = np.where(dists == 0.0, 1.0, dists)

    # ------------------------------------------------------------------
    # Step 4: sparse symmetric matrix
    # ------------------------------------------------------------------
    data = np.concatenate([dists, dists])
    row  = np.concatenate([i_idx, j_idx])
    col  = np.concatenate([j_idx, i_idx])

    mat = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 5: connected components
    # ------------------------------------------------------------------
    n_comp, labels = connected_components(mat, directed=False)

    return dict(
        matrix=mat,
        components=labels,
        n_components=n_comp,
        n_edges=len(pairs),
        density=len(pairs) / (n * (n - 1) / 2),
    )


def geographic_connectivity(
    coords: np.ndarray,
    h_max: float,
    metric: Literal["euclidean", "haversine"] = "euclidean",
) -> "csr_matrix":
    """
    Binary connectivity matrix for use with sklearn AgglomerativeClustering.

    Returns a sparse symmetric CSR matrix where entry (i, j) = 1 iff
    dist(i, j) <= h_max.  Pass directly to::

        from sklearn.cluster import AgglomerativeClustering
        connectivity = geographic_connectivity(coords, h_max)
        model = AgglomerativeClustering(
            distance_threshold=h_cut,
            n_clusters=None,
            connectivity=connectivity,
        )
        model.fit(coords)

    Note: AgglomerativeClustering will still compute pairwise distances
    internally; the connectivity matrix constrains *which* pairs may merge.
    For large n, use SpatialAgglomerativeClustering instead, which stores
    pre-computed distances in the sparse graph and avoids redundant computation.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Feature coordinates. (x, y) in metres for 'euclidean';
        (lon, lat) in degrees for 'haversine'.
    h_max : float
        Maximum distance threshold in metres.
    metric : str
        'euclidean' or 'haversine'.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n, n), dtype float64
        Binary (0/1) symmetric connectivity matrix.
    """
    graph = spatial_dist_graph(coords, h_max, metric=metric)
    mat = graph["matrix"].copy()
    mat.data[:] = 1.0
    return mat
