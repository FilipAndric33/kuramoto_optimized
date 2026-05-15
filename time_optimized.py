import os
import sys
import time
import numpy as np

KURAMOTO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "final_project", "src", "kuramoto",
)

N_NODES_LIST    = [200, 500, 2000]
N_REPEATS       = 3
COUPLING_VALUES = np.round(np.arange(0.4, 2.21, 0.2), 2).tolist()
N_WORKERS       = min(len(COUPLING_VALUES), 4)
DT              = 0.01
T               = 30.0
OUTPUT_DIR      = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_dask_optimized",
)
HDF5_DIR = os.path.join(OUTPUT_DIR, "hdf5")


def run_single_optimized(coupling, adj_mat, n_nodes, kuramoto_dir, dt, T, hdf5_dir):
    import sys, os, time
    import numpy as np
    sys.path.insert(0, kuramoto_dir)
    from kuramoto import Kuramoto

    t0 = time.time()

    model    = Kuramoto(n_nodes=n_nodes, coupling=coupling, dt=dt, T=T)
    activity = model.run(adj_mat=adj_mat)

    n_steps      = activity.shape[1]
    coherence_ts = np.array([
        Kuramoto.phase_coherence(activity[:, t]) for t in range(n_steps)
    ])
    mean_freq = model.mean_frequency(activity, adj_mat)

    os.makedirs(hdf5_dir, exist_ok=True)
    import h5py
    fname = os.path.join(hdf5_dir, f"coupling_{coupling:.2f}.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("activity",       data=activity)
        f.create_dataset("coherence_ts",   data=coherence_ts)
        f.create_dataset("mean_frequency", data=mean_freq)
        f.attrs["coupling"] = coupling
        f.attrs["n_nodes"]  = n_nodes

    return time.time() - t0


def main():
    from dask.distributed import Client, LocalCluster
    from collections import defaultdict

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("DASK (optimized Kuramoto) benchmark")
    lines.append(f"Workers : {N_WORKERS}  |  Tasks per sweep : {len(COUPLING_VALUES)}")
    lines.append(f"Couplings: {COUPLING_VALUES}")
    lines.append(f"T={T}, dt={DT}")
    lines.append("=" * 60)
    lines.append(f"{'n_nodes':<10}  {'component':<20}  {'run':<5}  {'time (s)'}")
    lines.append("-" * 60)

    total_results   = defaultdict(list)
    per_sim_results = defaultdict(list)

    for n in N_NODES_LIST:
        print(f"\n── n_nodes = {n} ──")

        adj_mat = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(adj_mat, 0.0)

        for rep in range(N_REPEATS):

            cluster = LocalCluster(
                n_workers          = N_WORKERS,
                processes          = True,
                threads_per_worker = 1,
                memory_limit       = "4GB",
                silence_logs       = "error",
                timeout            = 120,
            )
            client = Client(cluster, timeout=120)
            print(f"  rep {rep+1}/{N_REPEATS}  dashboard → {client.dashboard_link}")

            adj_future = client.scatter(adj_mat, broadcast=True)

            t0      = time.time()
            futures = [
                client.submit(
                    run_single_optimized,
                    k, adj_future, n, KURAMOTO_DIR, DT, T, HDF5_DIR,
                )
                for k in COUPLING_VALUES
            ]
            client.gather(futures)
            wall = time.time() - t0

            client.close()
            cluster.close()

            per_sim = wall / N_WORKERS

            total_results[n].append(wall)
            per_sim_results[n].append(per_sim)

            print(f"  wall={wall:.3f}s  per_sim={per_sim:.3f}s")
            lines.append(f"{n:<10}  {'dask_total':<20}  {rep+1:<5}  {wall:.6f}")
            lines.append(f"{n:<10}  {'dask_per_sim':<20}  {rep+1:<5}  {per_sim:.6f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("AVERAGES")
    lines.append("=" * 60)
    lines.append(f"{'n_nodes':<10}  {'component':<20}  {'avg time (s)'}")
    lines.append("-" * 60)

    for n in N_NODES_LIST:
        avg_t  = np.mean(total_results[n])
        avg_ps = np.mean(per_sim_results[n])
        lines.append(f"{n:<10}  {'dask_total':<20}  {avg_t:.6f}")
        lines.append(f"{n:<10}  {'dask_per_sim':<20}  {avg_ps:.6f}")
        print(f"  n={n}  avg wall={avg_t:.3f}s  avg per_sim={avg_ps:.3f}s")

    out_path = os.path.join(OUTPUT_DIR, "timing_results.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()