"""Tests for sparse_hclust, dense_hclust, stitch_linkage, and the sklearn API."""

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import (
    sparse_hclust,
    dense_hclust,  # not part of public API; tested here as benchmark baseline
    stitch_linkage,
    SpatialAgglomerativeClustering,
)


# ---------------------------------------------------------------------------
# sparse_hclust — basic
# ---------------------------------------------------------------------------

def test_labels_shape(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000, 5_000])
    assert set(result["labels"].keys()) == {2_000.0, 5_000.0}
    for labels in result["labels"].values():
        assert labels.shape == (200,)


def test_labels_are_positive_integers(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000])
    labels = result["labels"][2_000.0]
    assert labels.dtype == np.int64
    assert np.all(labels >= 1)


def test_cluster_count_monotone(small_clustered_coords):
    """Fewer clusters at larger cut heights."""
    graph = spatial_dist_graph(small_clustered_coords, h_max=20_000)
    result = sparse_hclust(graph, h_cuts=[1_000, 5_000, 15_000])
    n1 = len(np.unique(result["labels"][1_000.0]))
    n5 = len(np.unique(result["labels"][5_000.0]))
    n15 = len(np.unique(result["labels"][15_000.0]))
    assert n1 >= n5 >= n15


def test_custom_ids(small_clustered_coords):
    n = len(small_clustered_coords)
    ids = [f"pt_{i}" for i in range(n)]
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], ids=ids)
    assert result["ids"] == ids


def test_all_singletons_no_edges():
    coords = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0]])
    graph = spatial_dist_graph(coords, h_max=1.0)
    result = sparse_hclust(graph, h_cuts=[0.5])
    labels = result["labels"][0.5]
    assert len(np.unique(labels)) == 3


def test_results_field_keys(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000])
    assert "ids" in result
    assert "components" in result
    assert "labels" in result
    assert "linkage_trees" not in result  # not requested


# ---------------------------------------------------------------------------
# Exactness: sparse vs dense produce the same cluster counts
# ---------------------------------------------------------------------------

def test_sparse_matches_dense():
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 50_000, size=(300, 2))
    h_max = 10_000
    h_cuts = [2_000, 5_000, 8_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    sp = sparse_hclust(graph, h_cuts=h_cuts)
    dn = dense_hclust(coords, h_cuts=h_cuts)

    for h in h_cuts:
        n_sp = len(np.unique(sp["labels"][float(h)]))
        n_dn = len(np.unique(dn["labels"][float(h)]))
        assert n_sp == n_dn, f"Mismatch at h={h}: sparse={n_sp}, dense={n_dn}"


def test_sparse_matches_dense_exact_labels():
    """Not just cluster counts — verify identical cluster memberships."""
    rng = np.random.default_rng(99)
    coords = rng.uniform(0, 30_000, size=(150, 2))
    h_max = 15_000
    h_cuts = [3_000, 7_000, 12_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    sp = sparse_hclust(graph, h_cuts=h_cuts)
    dn = dense_hclust(coords, h_cuts=h_cuts)

    for h in h_cuts:
        sp_labels = sp["labels"][float(h)]
        dn_labels = dn["labels"][float(h)]
        # Labels may differ in numbering, but the partition must be identical.
        # Two points share a label in sparse iff they share a label in dense.
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                same_sp = sp_labels[i] == sp_labels[j]
                same_dn = dn_labels[i] == dn_labels[j]
                assert same_sp == same_dn, (
                    f"h={h}, pts ({i},{j}): sparse_same={same_sp}, dense_same={same_dn}"
                )


# ---------------------------------------------------------------------------
# return_linkage and stitch_linkage
# ---------------------------------------------------------------------------

def test_return_linkage_flag(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], return_linkage=True)
    assert "linkage_trees" in result
    assert isinstance(result["linkage_trees"], list)
    for Z, idx in result["linkage_trees"]:
        assert Z.ndim == 2 and Z.shape[1] == 4
        assert Z.shape[0] == len(idx) - 1
        assert idx.ndim == 1


def test_stitch_linkage_shape(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], return_linkage=True)
    Z = stitch_linkage(result)
    n = len(small_clustered_coords)
    assert Z.shape == (n - 1, 4)


def test_stitch_linkage_valid_structure(small_clustered_coords):
    """Stitched Z should be a valid scipy linkage matrix."""
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], return_linkage=True)
    Z = stitch_linkage(result)
    n = len(small_clustered_coords)

    # All leaf references (< n) and internal references (>= n) must be valid.
    all_refs = Z[:, :2].ravel()
    assert np.all(all_refs >= 0)
    assert np.all(all_refs < 2 * n - 1)

    # Last row count should equal n.
    assert Z[-1, 3] == n

    # Distances should be non-decreasing within each component's block,
    # and the inf merges come last.
    finite_mask = np.isfinite(Z[:, 2])
    inf_mask = ~finite_mask
    if inf_mask.any():
        # All inf rows should come after all finite rows.
        first_inf = np.argmax(inf_mask)
        assert np.all(finite_mask[:first_inf])
        assert np.all(inf_mask[first_inf:])


def test_stitch_linkage_fcluster_matches_sparse():
    """Cutting the stitched Z at a given height should produce the same
    cluster count as sparse_hclust."""
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 50_000, size=(200, 2))
    h_max = 10_000
    h_cuts = [3_000, 7_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    result = sparse_hclust(graph, h_cuts=h_cuts, return_linkage=True)
    Z = stitch_linkage(result)

    for h in h_cuts:
        from_sparse = len(np.unique(result["labels"][float(h)]))
        from_Z = len(np.unique(fcluster(Z, t=h, criterion="distance")))
        assert from_sparse == from_Z, (
            f"h={h}: sparse_hclust gives {from_sparse} clusters, "
            f"fcluster(Z) gives {from_Z}"
        )


def test_stitch_linkage_all_singletons():
    """When all points are singletons, stitched Z has only inf merges."""
    coords = np.array([[0.0, 0.0], [1e6, 0.0], [0.0, 1e6]])
    graph = spatial_dist_graph(coords, h_max=1.0)
    result = sparse_hclust(graph, h_cuts=[0.5], return_linkage=True)

    assert len(result["linkage_trees"]) == 0
    Z = stitch_linkage(result)
    assert Z.shape == (2, 4)
    assert np.all(np.isinf(Z[:, 2]))


def test_stitch_linkage_single_component():
    """When all points are in one component, stitched Z has no inf merges."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(20, 2))
    graph = spatial_dist_graph(coords, h_max=200)
    assert graph["n_components"] == 1

    result = sparse_hclust(graph, h_cuts=[50], return_linkage=True)
    Z = stitch_linkage(result)
    assert Z.shape == (19, 4)
    assert np.all(np.isfinite(Z[:, 2]))


def test_dense_hclust_return_linkage():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10_000, size=(50, 2))
    result = dense_hclust(coords, h_cuts=[3_000], return_linkage=True)
    assert "linkage_trees" in result
    assert len(result["linkage_trees"]) == 1
    Z, idx = result["linkage_trees"][0]
    assert Z.shape == (49, 4)
    assert len(idx) == 50


# ---------------------------------------------------------------------------
# dense_hclust
# ---------------------------------------------------------------------------

def test_dense_hclust_labels_shape():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10_000, size=(50, 2))
    result = dense_hclust(coords, h_cuts=[3_000, 8_000])
    for h in [3_000.0, 8_000.0]:
        assert result["labels"][h].shape == (50,)


# ---------------------------------------------------------------------------
# SpatialAgglomerativeClustering (sklearn API)
# ---------------------------------------------------------------------------

def test_sklearn_api_fit(small_clustered_coords):
    model = SpatialAgglomerativeClustering(h_max=5_000, distance_threshold=3_000)
    model.fit(small_clustered_coords)
    assert hasattr(model, "labels_")
    assert model.labels_.shape == (200,)
    assert model.n_clusters_ >= 1
    assert model.n_leaves_ == 200
    assert model.n_features_in_ == 2


def test_sklearn_api_fit_predict(small_clustered_coords):
    model = SpatialAgglomerativeClustering(h_max=5_000, distance_threshold=3_000)
    labels = model.fit_predict(small_clustered_coords)
    assert labels.shape == (200,)


def test_sklearn_api_exposes_linkage(small_clustered_coords):
    model = SpatialAgglomerativeClustering(h_max=5_000, distance_threshold=3_000)
    model.fit(small_clustered_coords)
    n = len(small_clustered_coords)

    assert hasattr(model, "linkage_matrix_")
    assert model.linkage_matrix_.shape == (n - 1, 4)

    assert hasattr(model, "children_")
    assert model.children_.shape == (n - 1, 2)

    assert hasattr(model, "distances_")
    assert model.distances_.shape == (n - 1,)
