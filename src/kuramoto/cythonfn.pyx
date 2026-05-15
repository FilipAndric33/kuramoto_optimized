# cython: boundscheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos

ctypedef cnp.float64_t DTYPE_t


def phase_coherence(cnp.ndarray[DTYPE_t, ndim=1] angles_vec):
    """
    Compute global order parameter R_t - mean length of resultant vector.
    """
    cdef:
        double[:] angles = angles_vec
        Py_ssize_t n = angles_vec.shape[0]
        Py_ssize_t i
        double re = 0.0
        double im = 0.0

    for i in range(n):
        re += cos(angles[i])
        im += sin(angles[i])

    re /= n
    im /= n

    return (re * re + im * im) ** 0.5


def mean_frequency(
    cnp.ndarray[DTYPE_t, ndim=2] act_mat,
    cnp.ndarray[DTYPE_t, ndim=2] adj_mat,
    cnp.ndarray[DTYPE_t, ndim=1] natfreqs,
    double coupling,
    double dt,
    double T,
):
    cdef:
        Py_ssize_t n_nodes = act_mat.shape[0]
        Py_ssize_t n_steps = act_mat.shape[1]
        Py_ssize_t node, j, t

        double[:, :] act  = act_mat
        double[:, :] adj  = adj_mat
        double[:] natf    = natfreqs

        double[:] n_interactions = np.zeros(n_nodes, dtype=np.float64)
        double[:] integral       = np.zeros(n_nodes, dtype=np.float64)

        double dxdt_node, norm_coupling

    for node in range(n_nodes):
        for j in range(n_nodes):
            if adj[node, j] != 0.0:
                n_interactions[node] += 1.0

    for t in range(n_steps):
        for node in range(n_nodes):
            dxdt_node = natf[node]
            norm_coupling = coupling / n_interactions[node] if n_interactions[node] > 0.0 else 0.0
            for j in range(n_nodes):
                if adj[node, j] != 0.0:
                    dxdt_node += norm_coupling * adj[node, j] * sin(act[j, t] - act[node, t])
            integral[node] += dxdt_node * dt

    result = np.empty(n_nodes, dtype=np.float64)
    cdef double[:] res = result
    for node in range(n_nodes):
        res[node] = integral[node] / T

    return result