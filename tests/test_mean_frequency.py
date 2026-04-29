import pytest
import numpy as np

from kuramoto import Kuramoto


def test_mean_frequency_dimension_mismatch():
    """Test that mean_frequency raises error when adj_mat does not match act_mat."""
    kuramoto = Kuramoto(n_nodes=5)
    act_mat = np.random.rand(5, 10)
    adj_mat = np.random.rand(4, 4)  # wrong size
    with pytest.raises(AssertionError):
        kuramoto.mean_frequency(act_mat, adj_mat)


def test_mean_frequency_output_shape():
    """Test that mean_frequency returns an array of the correct shape."""
    n_nodes = 7
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.random.rand(n_nodes, n_nodes)
    act_mat = np.random.rand(n_nodes, 10)
    meanfreq = kuramoto.mean_frequency(act_mat, adj_mat)
    assert len(meanfreq) == n_nodes
    assert isinstance(meanfreq, np.ndarray)


def test_mean_frequency_all_zeros():
    """Test mean_frequency with all-zero activity matrix."""
    n_nodes = 3
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1, natfreqs=np.zeros(n_nodes))
    adj_mat = np.random.rand(n_nodes, n_nodes)
    act_mat = np.zeros((n_nodes, 10))
    meanfreq = kuramoto.mean_frequency(act_mat, adj_mat)
    assert np.allclose(meanfreq, 0)


def test_mean_frequency_constant_activity():
    """Test mean_frequency with constant activity."""
    n_nodes = 3
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.random.rand(n_nodes, n_nodes)
    act_mat = np.ones((n_nodes, 10))
    meanfreq = kuramoto.mean_frequency(act_mat, adj_mat)
    assert not np.allclose(meanfreq, 0)
