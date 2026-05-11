from typing import Union

import numpy as np
import torch
from scipy.integrate import odeint


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
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")

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
            temp =  np.random.normal(size=self.n_nodes)
            self.natfreqs = torch.from_numpy(temp).to(device='cuda')

     #6 -5
    def init_angles(self):
        """
        Random initial random angles (position, "theta").
        """
        return 2 * np.pi * np.random.random(size=self.n_nodes)

     #8 -2
    def derivative(self, angles_vec, t, adj_mat, coupling):
        """
        Compute derivative of all nodes for current state, defined as

        dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) )
        ---- =             ---
         dt                M_i

        t: Not used, kept for compatibility with scipy.odeint
        """
        assert (
            len(angles_vec) == len(self.natfreqs) == len(adj_mat)
        ), "Input dimensions do not match, check lengths"

        angles_t = torch.from_numpy(angles_vec).to(device='cuda')
        angles_i, angles_j = torch.meshgrid(angles_t, angles_t)
        interactions = self.adj_mat_t * torch.sin(angles_j - angles_i)  # Aij * sin(j-i)
        
        # sum over incoming interactions
        dxdt = self.natfreqs + self.cur * interactions.sum(axis=0)
        new = torch.Tensor.cpu(dxdt)
        return new

     #8 -2

    def integrate(self, angles_vec, adj_mat):
        """Updates all states by integrating state of all nodes"""
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        coupling = (
            self.coupling / n_interactions
        )  # normalize coupling by number of interactions

        self.adj_mat_t = torch.from_numpy(adj_mat).to(device='cuda')

        t = np.linspace(0, self.T, int(self.T / self.dt))
        timeseries = odeint(self.derivative, angles_vec, t, args=(adj_mat, coupling))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

     #8 -2
    
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

    @staticmethod
    def phase_coherence(angles_vec):
        """
        Compute global order parameter R_t - mean length of resultant vector
        """
        suma = sum([(np.e ** (1j * i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

    def mean_frequency(self, act_mat, adj_mat):
        """
        Compute average frequency within the time window (self.T) for all nodes
        """
        assert len(adj_mat) == act_mat.shape[0], "adj_mat does not match act_mat"
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(
                angles_vec=act_mat[:, time],
                t=None,
                adj_mat=adj_mat,
                coupling=self.coupling,
            )

        # Integrate all nodes over the time window T
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.T
        return meanfreq

if __name__ == "__main__":
    n_nodes = 200
    arr = np.ones((n_nodes, n_nodes), dtype=np.float64)
    for n in range(n_nodes):
        arr[n][n] = 0
    
    model = Kuramoto(
        n_nodes = n_nodes,
        coupling = 1.95
    )

    activity = model.run(adj_mat=arr)