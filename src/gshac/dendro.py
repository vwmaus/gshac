"""
dendro.py

Dendrogram visualisation for GSHAC results.

Uses :func:`scipy.cluster.hierarchy.dendrogram` under the hood; this module
provides convenience wrappers that accept GSHAC output directly.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram
from typing import Optional, Sequence


def plot_dendrogram(
    model_or_result,
    *,
    ax=None,
    truncate_mode: Optional[str] = "lastp",
    p: int = 30,
    color_threshold: Optional[float] = None,
    no_labels: bool = True,
    show_inf: bool = False,
    **kwargs,
):
    """
    Plot a dendrogram from a fitted estimator or ``sparse_hclust`` result.

    Parameters
    ----------
    model_or_result : SpatialAgglomerativeClustering or dict
        Either a fitted :class:`SpatialAgglomerativeClustering` instance
        (uses ``model.linkage_matrix_``) or a dict returned by
        ``sparse_hclust(..., return_linkage=True)`` (will call
        :func:`stitch_linkage` automatically).
    ax : matplotlib Axes, optional
        Axes to draw on.  If *None*, creates a new figure.
    truncate_mode : str or None, default ``"lastp"``
        Passed to :func:`scipy.cluster.hierarchy.dendrogram`.  ``"lastp"``
        shows only the last *p* merges, which keeps large dendrograms
        readable.  Set to ``None`` for the full tree.
    p : int, default 30
        Number of merges to show when ``truncate_mode="lastp"``.
    color_threshold : float or None
        Merges above this distance are coloured the default colour.
        Useful for highlighting a specific cut height.
    no_labels : bool, default True
        Suppress leaf labels (often unreadable for large n).
    show_inf : bool, default False
        If *False* (the default), inter-component merges at infinity are
        excluded from the y-axis range — the dendrogram is clipped to the
        maximum finite merge distance.  Set to *True* to show the full
        range including the infinity merges.
    **kwargs
        Additional keyword arguments passed to
        :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    R : dict
        The dendrogram data dict returned by scipy (contains ``'icoord'``,
        ``'dcoord'``, ``'ivl'``, ``'leaves'``, ``'color_list'``).
    """
    import matplotlib.pyplot as plt
    from gshac.sparse_hclust import stitch_linkage

    # Get the linkage matrix.
    if isinstance(model_or_result, dict):
        if "linkage_trees" not in model_or_result:
            raise ValueError(
                "Result dict has no 'linkage_trees' key.  "
                "Re-run sparse_hclust(..., return_linkage=True)."
            )
        Z = stitch_linkage(model_or_result)
    else:
        # Assume sklearn-style estimator with linkage_matrix_.
        Z = model_or_result.linkage_matrix_

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # scipy's dendrogram chokes on inf distances (it tries to set ylim=inf).
    # Replace inf merges with a finite sentinel so plotting succeeds.
    finite_dists = Z[np.isfinite(Z[:, 2]), 2]
    max_finite = finite_dists.max() if len(finite_dists) > 0 else 1.0
    has_inf = np.any(np.isinf(Z[:, 2]))

    if has_inf:
        Z_plot = Z.copy()
        inf_sentinel = max_finite * 1.5
        Z_plot[np.isinf(Z_plot[:, 2]), 2] = inf_sentinel
    else:
        Z_plot = Z

    R = _scipy_dendrogram(
        Z_plot,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        no_labels=no_labels,
        **kwargs,
    )

    if not show_inf and has_inf:
        # Clip y-axis to max finite distance for readability.
        ax.set_ylim(0, max_finite * 1.05)

    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Sample index (or cluster size)")

    return R


def plot_component_dendrograms(
    result: dict,
    *,
    top_k: int = 4,
    figsize: Optional[tuple] = None,
    color_threshold: Optional[float] = None,
    **kwargs,
):
    """
    Plot individual dendrograms for the largest connected components.

    Parameters
    ----------
    result : dict
        Output of ``sparse_hclust(..., return_linkage=True)``.
    top_k : int, default 4
        Number of largest components to plot.
    figsize : tuple, optional
        Figure size.  Defaults to ``(12, 3 * top_k)``.
    color_threshold : float or None
        Highlight merges below this distance.
    **kwargs
        Passed to :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    fig : matplotlib Figure
    axes : list of Axes
    """
    import matplotlib.pyplot as plt

    linkage_trees = result["linkage_trees"]
    if not linkage_trees:
        raise ValueError("No non-singleton components to plot.")

    # Sort by component size (descending).
    ranked = sorted(linkage_trees, key=lambda t: len(t[1]), reverse=True)
    ranked = ranked[:top_k]

    if figsize is None:
        figsize = (12, 3 * len(ranked))

    fig, axes = plt.subplots(len(ranked), 1, figsize=figsize)
    if len(ranked) == 1:
        axes = [axes]

    for ax, (Z_local, idx) in zip(axes, ranked):
        _scipy_dendrogram(
            Z_local,
            ax=ax,
            color_threshold=color_threshold,
            no_labels=len(idx) > 50,
            **kwargs,
        )
        ax.set_title(f"Component (n={len(idx):,})")
        ax.set_ylabel("Distance (m)")

    fig.tight_layout()
    return fig, axes
