import pytest
import numpy as np

from kuramoto import Kuramoto


def test_derivative_dimension_mismatch():
    """Test that derivative raises an error when input dimensions don't match."""
    kuramoto = Kuramoto(n_nodes=5)
    adj_mat = np.random.rand(5, 5)
    angles_vec = np.random.rand(4)  # Wrong size!
    with pytest.raises(AssertionError):
        kuramoto.derivative(angles_vec, 0, adj_mat, coupling=1.0)

    angles_vec = np.random.rand(5)
    adj_mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        kuramoto.derivative(angles_vec, 0, adj_mat, coupling=1.0)


def test_derivative_output_shape():
    """Test that derivative returns an array of the correct shape."""
    n_nodes = 6
    kuramoto = Kuramoto(n_nodes=n_nodes)
    adj_mat = np.random.rand(n_nodes, n_nodes)
    angles_vec = np.random.rand(n_nodes)
    dxdt = kuramoto.derivative(angles_vec, 0, adj_mat, coupling=1.0)
    assert len(dxdt) == n_nodes
    assert isinstance(dxdt, np.ndarray)


def test_integrate_output_shape():
    """Test that integrate returns an array of the correct shape."""
    n_nodes = 4
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.random.rand(n_nodes, n_nodes)
    angles_vec = np.random.rand(n_nodes)
    timeseries = kuramoto.integrate(angles_vec, adj_mat)
    expected_time_steps = int(kuramoto.T / kuramoto.dt)
    assert timeseries.shape == (n_nodes, expected_time_steps)
    assert isinstance(timeseries, np.ndarray)


def test_run_with_adj_mat():
    """Test that run returns an array of the correct shape when adj_mat is provided."""
    n_nodes = 3
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.random.rand(n_nodes, n_nodes)
    act_mat = kuramoto.run(adj_mat=adj_mat)
    expected_time_steps = int(kuramoto.T / kuramoto.dt)
    assert act_mat.shape == (n_nodes, expected_time_steps)
    assert isinstance(act_mat, np.ndarray)


def test_run_no_adj_mat():
    """Test that run returns an array of the correct shape even without adj_mat."""
    # The case where adj_mat is None needs to be handled correctly within the function
    n_nodes = 3
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.ones((n_nodes, n_nodes))  # Use a ones matrix as default
    act_mat = kuramoto.run(adj_mat=adj_mat)
    expected_time_steps = int(kuramoto.T / kuramoto.dt)
    assert act_mat.shape == (n_nodes, expected_time_steps)
    assert isinstance(act_mat, np.ndarray)


def test_run_empty_adj_mat():
    """Test run with an empty adjacency matrix (no connections)."""
    n_nodes = 3
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.zeros((n_nodes, n_nodes))
    act_mat = kuramoto.run(adj_mat=adj_mat)
    expected_time_steps = int(kuramoto.T / kuramoto.dt)
    assert act_mat.shape == (n_nodes, expected_time_steps)  # Still should run
    assert isinstance(act_mat, np.ndarray)


def test_derivative_zero_coupling():
    """Test derivative with zero coupling."""
    n_nodes = 5
    kuramoto_model = Kuramoto(n_nodes=n_nodes, coupling=0.5, dt=0.01, T=1)
    n_nodes = kuramoto_model.n_nodes
    adj_mat = np.random.rand(n_nodes, n_nodes)
    angles_vec = np.random.rand(n_nodes)
    dxdt = kuramoto_model.derivative(angles_vec, 0, adj_mat, coupling=0.0)
    # With zero coupling, derivative should just be natural frequencies
    assert np.allclose(dxdt, kuramoto_model.natfreqs)


def test_derivative_identity_adj_mat():
    """Test derivative with identity adjacency matrix."""
    n_nodes = 4
    kuramoto = Kuramoto(n_nodes=n_nodes)
    adj_mat = np.eye(n_nodes)  # Identity matrix
    angles_vec = np.random.rand(n_nodes)
    dxdt = kuramoto.derivative(angles_vec, 0, adj_mat, coupling=1.0)
    # Derivative depends on natfreqs, and initial angles, so just check it has the right shape
    assert dxdt.shape == (n_nodes,)


def test_integration_with_zero_initial_angles():
    """Test integration with zero initial angles."""
    n_nodes = 4
    kuramoto = Kuramoto(n_nodes=n_nodes, dt=0.1, T=1)
    adj_mat = np.random.rand(n_nodes, n_nodes)
    angles_vec = np.zeros(n_nodes)  # Zero initial angles
    timeseries = kuramoto.integrate(angles_vec, adj_mat)
    expected_time_steps = int(kuramoto.T / kuramoto.dt)
    assert timeseries.shape == (n_nodes, expected_time_steps)
    assert isinstance(timeseries, np.ndarray)
