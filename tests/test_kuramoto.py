import sys
sys.path.insert(0, "src/kuramoto")

import pytest
import numpy as np
from kuramoto import Kuramoto


@pytest.fixture
def small_model():
    return Kuramoto(n_nodes=10, coupling=1.0, dt=0.01, T=1.0)

@pytest.fixture
def small_adj():
    adj = np.ones((10, 10), dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def test_init_angles_shape(small_model):
    angles = small_model.init_angles()
    assert angles.shape == (10,)

def test_init_angles_range(small_model):
    angles = small_model.init_angles()
    assert np.all(angles >= 0) and np.all(angles <= 2 * np.pi)

def test_run_output_shape(small_model, small_adj):
    activity = small_model.run(adj_mat=small_adj)
    n_steps = int(small_model.T / small_model.dt)
    assert activity.shape == (10, n_steps)

def test_phase_coherence_synced():
    angles = np.zeros(100, dtype=np.float64)
    assert Kuramoto.phase_coherence(angles) == pytest.approx(1.0)

def test_phase_coherence_uniform():
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    assert Kuramoto.phase_coherence(angles) == pytest.approx(0.0, abs=1e-6)

def test_phase_coherence_range(small_model, small_adj):
    activity = small_model.run(adj_mat=small_adj)
    for t in range(activity.shape[1]):
        r = Kuramoto.phase_coherence(activity[:, t])
        assert 0.0 <= r <= 1.0

def test_mean_frequency_shape(small_model, small_adj):
    activity = small_model.run(adj_mat=small_adj)
    mf = small_model.mean_frequency(activity, small_adj)
    assert mf.shape == (10,)

def test_adj_mat_mismatch_raises(small_model):
    bad_adj = np.ones((5, 5), dtype=np.float64)
    activity = np.zeros((10, 100))
    with pytest.raises(AssertionError):
        small_model.mean_frequency(activity, bad_adj)