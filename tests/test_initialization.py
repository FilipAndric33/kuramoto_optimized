import pytest
import numpy as np

from kuramoto import Kuramoto


def test_n_nodes_natfreqs_none():
    with pytest.raises(ValueError):
        Kuramoto(n_nodes=None, natfreqs=None)

    with pytest.raises(ValueError):
        Kuramoto()


def test_natfreqs():
    model = Kuramoto(n_nodes=None, natfreqs=np.array([1, 2, 3]))
    assert model.n_nodes == 3
    assert np.all(model.natfreqs == np.array([1, 2, 3]))


def test_n_nodes():
    model = Kuramoto(n_nodes=3, natfreqs=np.array([1, 2, 3]))
    assert model.n_nodes == 3
    assert np.all(model.natfreqs == np.array([1, 2, 3]))


def test_initialization_with_n_nodes():
    """Test initialization when only n_nodes is provided."""
    kuramoto = Kuramoto(n_nodes=10)
    assert kuramoto.n_nodes == 10
    assert len(kuramoto.natfreqs) == 10
    assert isinstance(kuramoto.natfreqs, np.ndarray)


def test_initialization_with_natfreqs():
    """Test initialization when only natfreqs is provided."""
    natfreqs = np.random.rand(5)
    kuramoto = Kuramoto(natfreqs=natfreqs)
    assert kuramoto.n_nodes == 5
    assert np.allclose(kuramoto.natfreqs, natfreqs)


def test_initialization_with_both():
    """Test initialization with both n_nodes and natfreqs."""
    natfreqs = np.random.rand(7)
    kuramoto = Kuramoto(n_nodes=10, natfreqs=natfreqs)
    assert kuramoto.n_nodes == 7  # natfreqs should override n_nodes
    assert np.allclose(kuramoto.natfreqs, natfreqs)


def test_init_angles():
    """Test that init_angles returns an array of the correct shape."""
    kuramoto = Kuramoto(n_nodes=8)
    angles = kuramoto.init_angles()
    assert len(angles) == 8
    assert isinstance(angles, np.ndarray)
    assert np.all(angles >= 0)
    assert np.all(angles <= 2 * np.pi)


def test_example_integration():
    """Test for valid output shape on example network"""
    N = 5
    T = 10
    dt = 0.01
    adj_mat = np.ones((N, N)) - np.eye(N)  # fully connected
    model = Kuramoto(n_nodes=N, T=T, dt=dt)
    activity = model.run(adj_mat)
    assert activity.shape == (N, int(T / dt))


@pytest.fixture
def kuramoto_model():
    return Kuramoto(n_nodes=5, coupling=0.5, dt=0.01, T=1)


def test_coupling_strength_parameter(kuramoto_model):
    """Test if the coupling strength is correctly set during initialization."""
    assert kuramoto_model.coupling == 0.5


def test_dt_parameter(kuramoto_model):
    """Test if the dt is correctly set during initialization."""
    assert kuramoto_model.dt == 0.01


def test_t_parameter(kuramoto_model):
    """Test if the T is correctly set during initialization."""
    assert kuramoto_model.T == 1


def test_natfreqs_are_different(kuramoto_model):
    """Test that the natural frequencies are not all the same after initialization."""
    assert not np.all(kuramoto_model.natfreqs == kuramoto_model.natfreqs[0])
