"""Tests for spatial_dist_graph and geographic_connectivity."""

import numpy as np
import pytest
from scipy.sparse import issparse

from gshac.spatial_dist_graph import spatial_dist_graph, geographic_connectivity


def test_basic_euclidean(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    assert issparse(graph["matrix"])
    assert graph["matrix"].shape == (200, 200)
    assert graph["n_edges"] > 0
    assert graph["n_components"] >= 1
    assert len(graph["components"]) == 200


def test_symmetric_matrix(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    mat = graph["matrix"]
    diff = (mat - mat.T).data
    assert np.allclose(diff, 0), "Distance matrix is not symmetric"


def test_diagonal_zero(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    diag = graph["matrix"].diagonal()
    assert np.all(diag == 0), "Diagonal should be zero"


def test_no_edges_all_singletons():
    # Points far apart: h_max=1 m, so no pairs within radius
    coords = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0]])
    graph = spatial_dist_graph(coords, h_max=1.0)
    assert graph["n_edges"] == 0
    assert graph["n_components"] == 3


def test_haversine_metric(small_lonlat_coords, backend):
    graph = spatial_dist_graph(small_lonlat_coords, h_max=10_000, metric="haversine")
    assert issparse(graph["matrix"])
    assert graph["n_edges"] > 0
    # All stored distances should be positive and <= h_max
    assert np.all(graph["matrix"].data > 0)
    assert np.all(graph["matrix"].data <= 10_000)


def test_haversine_c_and_python_agree(small_lonlat_coords, monkeypatch):
    """C haversine and numpy haversine fallback must produce identical distances."""
    import sys
    import gshac.spatial_dist_graph

    monkeypatch.setattr(sys.modules["gshac.spatial_dist_graph"], "_GSHAC_C", True)
    graph_c = spatial_dist_graph(small_lonlat_coords, h_max=10_000, metric="haversine")

    monkeypatch.setattr(sys.modules["gshac.spatial_dist_graph"], "_GSHAC_C", False)
    graph_py = spatial_dist_graph(small_lonlat_coords, h_max=10_000, metric="haversine")

    assert graph_c["n_edges"] == graph_py["n_edges"]
    assert np.allclose(
        np.sort(graph_c["matrix"].data),
        np.sort(graph_py["matrix"].data),
        rtol=1e-10,
    ), "C and Python haversine produce different distances"


def test_density_in_range(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    assert 0.0 <= graph["density"] <= 1.0


def test_invalid_metric(small_clustered_coords):
    with pytest.raises(ValueError):
        spatial_dist_graph(small_clustered_coords, h_max=5_000, metric="cosine")


def test_too_few_points():
    with pytest.raises(ValueError):
        spatial_dist_graph(np.array([[0.0, 0.0]]), h_max=1_000)


def test_geographic_connectivity(small_clustered_coords):
    conn = geographic_connectivity(small_clustered_coords, h_max=5_000)
    assert issparse(conn)
    # All stored values should be 1
    assert np.all(conn.data == 1.0)
