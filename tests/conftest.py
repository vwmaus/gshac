"""Shared pytest fixtures for GSHAC tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_clustered_coords(rng):
    """200 points in 4 tight clusters, domain 100 km, euclidean (metres)."""
    n_centres = 4
    n = 200
    centres = rng.uniform(0, 100_000, size=(n_centres, 2))
    assigns = rng.integers(0, n_centres, size=n)
    noise = rng.normal(0, 500, size=(n, 2))
    return centres[assigns] + noise


@pytest.fixture
def small_lonlat_coords(rng):
    """100 points near London in lon/lat degrees."""
    base_lon, base_lat = -0.1, 51.5
    lon = base_lon + rng.normal(0, 0.05, size=100)
    lat = base_lat + rng.normal(0, 0.05, size=100)
    return np.column_stack([lon, lat])
