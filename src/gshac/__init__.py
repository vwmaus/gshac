"""GSHAC: Sparse Geographic Hierarchical Agglomerative Clustering."""

from gshac.spatial_dist_graph import spatial_dist_graph, geographic_connectivity
from gshac.sparse_hclust import (
    sparse_hclust,
    stitch_linkage,
    SparseAgglomerativeClustering,
)
from gshac.dendro import plot_dendrogram, plot_component_dendrograms

__all__ = [
    "spatial_dist_graph",
    "geographic_connectivity",
    "sparse_hclust",
    "stitch_linkage",
    "SparseAgglomerativeClustering",
    "plot_dendrogram",
    "plot_component_dendrograms",
]
