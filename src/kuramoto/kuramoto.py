from typing import Union

import numpy as np
import torch
from scipy.integrate import odeint
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import cythonfn


class Kuramoto:

    def __init__(
        self,
        coupling: float = 1,
        dt: float = 0.01,
        T: float = 30,
        n_nodes: Union[int, None] = None,
        natfreqs: Union[np.ndarray, None] = None,
        cur: Union[np.ndarray, None] = None
    ):
        """
        coupling: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        T: float
            Total time of simulated activity.
            From that the number of integration steps is T/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        """

        self.dt = dt
        self.T = T

        c = np.full(n_nodes, coupling, dtype=np.float32)
        self.cur = torch.from_numpy(c).to(device='cuda', dtype=torch.float32)

        self.coupling = coupling

        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            temp = np.random.normal(size=self.n_nodes)
            self.natfreqs = torch.from_numpy(temp).to(device='cuda')

    def init_angles(self):
        """
        Random initial random angles (position, "theta").
        """
        return 2 * np.pi * np.random.random(size=self.n_nodes)

    def _derivative_gpu(self, angles_t):
        angles_i, angles_j = torch.meshgrid(
            angles_t,
            angles_t,
            indexing='xy'
        )

        interactions = self.adj_mat_t * torch.sin(angles_j - angles_i)

        return self.natfreqs + self.cur * interactions.sum(axis=0)

    def integrate(self, angles_vec, adj_mat):

        n_interactions = (adj_mat != 0).sum(axis=0)

        coupling_np = self.coupling / n_interactions

        self.cur = torch.from_numpy(
            coupling_np
        ).to(device='cuda', dtype=torch.float32)

        self.adj_mat_t = torch.from_numpy(
            adj_mat
        ).to(device='cuda', dtype=torch.float32)

        angles_t = torch.from_numpy(
            angles_vec
        ).to(device='cuda', dtype=torch.float32)

        steps = int(self.T / self.dt)

        results = torch.zeros(
            (steps, self.n_nodes),
            device='cuda'
        )

        for i in range(steps):

            k1 = self._derivative_gpu(angles_t)

            k2 = self._derivative_gpu(
                angles_t + 0.5 * self.dt * k1
            )

            k3 = self._derivative_gpu(
                angles_t + 0.5 * self.dt * k2
            )

            k4 = self._derivative_gpu(
                angles_t + self.dt * k3
            )

            angles_t = angles_t + (
                self.dt / 6
            ) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

            results[i] = angles_t

        return results.cpu().numpy().T

    @staticmethod
    def phase_coherence(angles_vec):

        return cythonfn.phase_coherence(
            np.asarray(angles_vec, dtype=np.float64)
        )

    def mean_frequency(self, act_mat, adj_mat):

        return cythonfn.mean_frequency(
            np.ascontiguousarray(
                act_mat,
                dtype=np.float64
            ),
            np.ascontiguousarray(
                adj_mat,
                dtype=np.float64
            ),
            np.ascontiguousarray(
                self.natfreqs.cpu().numpy(),
                dtype=np.float64
            ),
            float(self.coupling),
            float(self.dt),
            float(self.T),
        )

    def run(self, adj_mat=None, angles_vec=None):
        """
        adj_mat: 2D nd array
            Adjacency matrix representing connectivity.

        angles_vec: 1D ndarray, optional
            States vector of nodes representing the position in radians.
            If not specified, random initialization [0, 2pi].

        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        """

        if angles_vec is None:
            angles_vec = self.init_angles()

        return self.integrate(angles_vec, adj_mat)