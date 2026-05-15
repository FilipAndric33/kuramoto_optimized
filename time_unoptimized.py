import sys
import os
import time
import numpy as np

UNOPT_DIR = os.path.join(os.path.dirname(__file__), "kuramoto_init", "kuramoto", "src", "kuramoto")
sys.path.insert(0, UNOPT_DIR)

from kuramoto import Kuramoto

N_NODES_LIST = [200, 500, 2000]
N_REPEATS    = 3
COUPLING     = 1.0
DT           = 0.01
T            = 10.0
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "results_unoptimized")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_adj(n):
    A = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def time_fn(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.time()
        fn()
        times.append(time.time() - t0)
    return times


lines = []
lines.append("=" * 60)
lines.append("UNOPTIMIZED Kuramoto – phase_coherence & mean_frequency")
lines.append("=" * 60)
lines.append(f"{'n_nodes':<10}  {'component':<20}  {'run':<5}  {'time (s)'}")
lines.append("-" * 60)

pc_results = {}
mf_results = {}

for n in N_NODES_LIST:
    print(f"\n── n_nodes = {n} ──")
    adj_mat = make_adj(n)

    model      = Kuramoto(coupling=COUPLING, dt=DT, T=T, n_nodes=n)
    angles_vec = model.init_angles()
    activity   = model.integrate(angles_vec, adj_mat)

    def bench_phase_coherence():
        n_steps = activity.shape[1]
        for t_idx in range(n_steps):
            Kuramoto.phase_coherence(activity[:, t_idx])

    pc_times = time_fn(bench_phase_coherence, N_REPEATS)
    pc_results[n] = pc_times
    for i, t in enumerate(pc_times):
        print(f"  phase_coherence  run {i+1}/{N_REPEATS}: {t:.4f}s")
        lines.append(f"{n:<10}  {'phase_coherence':<20}  {i+1:<5}  {t:.6f}")

    def bench_mean_frequency():
        model.mean_frequency(activity, adj_mat)

    mf_times = time_fn(bench_mean_frequency, N_REPEATS)
    mf_results[n] = mf_times
    for i, t in enumerate(mf_times):
        print(f"  mean_frequency   run {i+1}/{N_REPEATS}: {t:.4f}s")
        lines.append(f"{n:<10}  {'mean_frequency':<20}  {i+1:<5}  {t:.6f}")

lines.append("")
lines.append("=" * 60)
lines.append("AVERAGES")
lines.append("=" * 60)
lines.append(f"{'n_nodes':<10}  {'component':<20}  {'avg time (s)'}")
lines.append("-" * 60)

for n in N_NODES_LIST:
    avg_pc = np.mean(pc_results[n])
    avg_mf = np.mean(mf_results[n])
    lines.append(f"{n:<10}  {'phase_coherence':<20}  {avg_pc:.6f}")
    lines.append(f"{n:<10}  {'mean_frequency':<20}  {avg_mf:.6f}")

out_path = os.path.join(OUTPUT_DIR, "timing_results.txt")
with open(out_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\nResults written to: {out_path}")