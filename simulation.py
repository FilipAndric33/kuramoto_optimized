import sys
sys.path.insert(0, 'src')

from kuramoto import Kuramoto, plot_activity, plot_phase_coherence 
import networkx as nx
import matplotlib.pyplot as plt


def run_sim(): 
    n_nodes = 100
    graph = nx.erdos_renyi_graph(n=n_nodes, p=1)
    A = nx.to_numpy_array(graph)
    
    model = Kuramoto(
        n_nodes = 100,
        coupling = 1.95
    )

    activity = model.run(adj_mat=A)

    #plot_activity(activity)
    #plt.show()
#
    #plot_phase_coherence(activity)
    #plt.show()

run_sim()