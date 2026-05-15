import sys
sys.path.insert(0, 'src')

from kuramoto import Kuramoto, plot_activity, plot_phase_coherence 
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np

def run_sim(): 
    n_nodes = 2000
    t1 = time.time()
    arr = np.ones((n_nodes, n_nodes), dtype=np.float64)
    for n in range(n_nodes):
        arr[n][n] = 0
    
    model = Kuramoto(
        n_nodes = n_nodes,
        coupling = 1.95
    )

    activity = model.run(adj_mat=arr)
    t2 = time.time()
    print(f"Time needed to calculate the model: {t2 - t1}")
    plot_activity(activity)

    plot_phase_coherence(activity)

run_sim()