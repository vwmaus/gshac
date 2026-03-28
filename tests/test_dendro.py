"""Tests for dendrogram visualisation helpers."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust, SpatialAgglomerativeClustering
from gshac.dendro import plot_dendrogram, plot_component_dendrograms


@pytest.fixture
def result_with_linkage(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=10_000)
    return sparse_hclust(graph, h_cuts=[3_000], return_linkage=True)


@pytest.fixture
def fitted_model(small_clustered_coords):
    model = SpatialAgglomerativeClustering(h_max=10_000, distance_threshold=5_000)
    model.fit(small_clustered_coords)
    return model


def test_plot_dendrogram_from_result(result_with_linkage):
    fig, ax = plt.subplots()
    R = plot_dendrogram(result_with_linkage, ax=ax)
    assert "icoord" in R
    assert "dcoord" in R
    plt.close(fig)


def test_plot_dendrogram_from_model(fitted_model):
    fig, ax = plt.subplots()
    R = plot_dendrogram(fitted_model, ax=ax)
    assert "icoord" in R
    plt.close(fig)


def test_plot_dendrogram_creates_figure(result_with_linkage):
    R = plot_dendrogram(result_with_linkage)
    assert R is not None
    plt.close("all")


def test_plot_dendrogram_no_truncation(result_with_linkage):
    R = plot_dendrogram(result_with_linkage, truncate_mode=None)
    assert R is not None
    plt.close("all")


def test_plot_dendrogram_with_color_threshold(result_with_linkage):
    R = plot_dendrogram(result_with_linkage, color_threshold=5_000)
    assert R is not None
    plt.close("all")


def test_plot_dendrogram_show_inf(result_with_linkage):
    R = plot_dendrogram(result_with_linkage, show_inf=True)
    assert R is not None
    plt.close("all")


def test_plot_dendrogram_rejects_no_linkage(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000])  # no return_linkage
    with pytest.raises(ValueError, match="linkage_trees"):
        plot_dendrogram(result)


def test_plot_component_dendrograms(result_with_linkage):
    fig, axes = plot_component_dendrograms(result_with_linkage, top_k=2)
    assert len(axes) <= 2
    plt.close(fig)


def test_plot_component_dendrograms_single(result_with_linkage):
    fig, axes = plot_component_dendrograms(result_with_linkage, top_k=1)
    assert len(axes) == 1
    plt.close(fig)
