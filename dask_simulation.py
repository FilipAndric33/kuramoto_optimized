import sys
sys.path.insert(0, "src/kuramoto")

import time
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
from src.kuramoto import Kuramoto, plot_activity, plot_phase_coherence
import h5py
import os

def run_single(coupling, adj_mat, n_nodes):
    import sys
    sys.path.insert(0, "src/kuramoto")
    from kuramoto import Kuramoto

    model    = Kuramoto(n_nodes=n_nodes, coupling=coupling, dt=0.01, T=30.0)
    activity = model.run(adj_mat=adj_mat)

    n_steps      = activity.shape[1]
    coherence_ts = np.array([Kuramoto.phase_coherence(activity[:, t]) for t in range(n_steps)])
    mean_freq    = model.mean_frequency(activity, adj_mat)

    os.makedirs("results", exist_ok=True)
    fname = f"results/coupling_{coupling:.2f}.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("activity",     data=activity)
        f.create_dataset("coherence_ts", data=coherence_ts)
        f.create_dataset("mean_frequency", data=mean_freq)
        f.attrs["coupling"] = coupling
        f.attrs["n_nodes"]  = n_nodes

    return {
        "coupling":       coupling,
        "final_R":        float(coherence_ts[-1]),
        "mean_R":         float(coherence_ts.mean()),
        "mean_frequency": mean_freq.tolist(),
        "coherence_ts":   coherence_ts.tolist(),
        "activity":       activity,
    }

def plot_coupling_sweep(results):
    results   = sorted(results, key=lambda r: r["coupling"])
    couplings = [r["coupling"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(couplings, [r["mean_R"]  for r in results], "o-",  label="Mean R")
    axes[0].plot(couplings, [r["final_R"] for r in results], "s--", label="Final R")
    axes[0].set_xlabel("Coupling K")
    axes[0].set_ylabel("R")
    axes[0].set_title("Phase coherence vs coupling")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        axes[1].plot(r["coherence_ts"], color=cmap(i / max(len(results)-1, 1)),
                     alpha=0.8, label=f"K={r['coupling']:.2f}")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("R(t)")
    axes[1].set_title("R(t) per coupling")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_sim():
    n_nodes = 2000
    adj_mat = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(adj_mat, 0.0)

    coupling_values = np.round(np.arange(0.4, 2.21, 0.2), 2).tolist()
    print(f"Sweeping couplings: {coupling_values}")

    t0 = time.time()

    cluster = LocalCluster(
        n_workers=min(len(coupling_values), 6),
        threads_per_worker=1,
        memory_limit="4GB",
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    adj_future = client.scatter(adj_mat, broadcast=True)
    futures    = [client.submit(run_single, k, adj_future, n_nodes) for k in coupling_values]
    results    = client.gather(futures)

    client.close()
    cluster.close()

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"\n{'Coupling':>10}  {'Mean R':>8}  {'Final R':>8}")
    print("-" * 32)
    for r in sorted(results, key=lambda x: x["coupling"]):
        print(f"{r['coupling']:>10.2f}  {r['mean_R']:>8.4f}  {r['final_R']:>8.4f}")

    plot_coupling_sweep(results)

    best = max(results, key=lambda r: r["final_R"])
    print(f"\nDetailed plots for K={best['coupling']:.2f}")
    plot_activity(best["activity"])
    plot_phase_coherence(best["activity"])


if __name__ == "__main__":
    run_sim()