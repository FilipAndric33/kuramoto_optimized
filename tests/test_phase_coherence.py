import pytest
import numpy as np

from kuramoto import Kuramoto


def test_phase_coherence_perfect_sync():
    """Test phase coherence with perfectly synchronized oscillators."""
    angles_vec = np.zeros(5)
    coherence = Kuramoto.phase_coherence(angles_vec)
    assert np.isclose(coherence, 1.0)  # Should be close to 1.0


def test_phase_coherence_random_angles():
    """Test phase coherence with random angles. Should be between 0 and 1"""
    angles_vec = np.random.rand(10) * 2 * np.pi  # Random angles
    coherence = Kuramoto.phase_coherence(angles_vec)
    assert 0 <= coherence <= 1


def test_phase_coherence_output():
    """Test that phase_coherence returns a single float value."""
    angles_vec = np.random.rand(4)
    coherence = Kuramoto.phase_coherence(angles_vec)
    assert isinstance(coherence, float)
    assert 0 <= coherence <= 1  # Coherence must be in range [0,1]
