import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

COMPONENT_UNOPT_FILE    = "results_unoptimized/timing_results.txt"
COMPONENT_OPT_FILE      = "results_optimized/timing_results.txt"
DASK_UNOPT_FILE         = "results_dask_unoptimized/timing_results.txt"
DASK_OPT_FILE           = "results_dask_optimized/timing_results.txt"

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HARDCODED_FULL_UNOPT = {
    200:  [0.7992701530456543, 0.764631986618042, 0.8252792358398438],
    500:  [7.5130627155303955, 5.852824687957764, 6.657972812652588],
    2000: [91.90332579612732,  81.82598328590393, 92.22211837768555],
}

HARDCODED_FULL_GPU = {
    200:  [1.808081, 1.902022, 1.865599],
    500:  [2.054506, 1.873428, 2.007263],
    2000: [16.316003, 16.431690, 16.567248],
}

DASK_UNOPT_N_WORKERS = 4
DASK_OPT_N_WORKERS   = 4

KNOWN_COMPONENTS = {
    "phase_coherence", "mean_frequency",
    "dask_total", "dask_per_sim",
    "n_nodes",
}


def parse_component_file(path):
    result = defaultdict(lambda: defaultdict(list))
    in_avg = False

    if not os.path.exists(path):
        print(f"  [WARN] file not found: {path}")
        return {}

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("AVERAGES"):
                in_avg = True
                continue
            if in_avg:
                continue
            if line.startswith(("=", "-", "#")):
                continue

            parts = line.split()

            if len(parts) >= 2 and parts[0] == "n_nodes" and parts[1] == "component":
                continue
            if len(parts) != 4:
                continue

            try:
                n    = int(parts[0])
                comp = parts[1]
                t    = float(parts[3])
            except ValueError:
                continue

            if comp not in KNOWN_COMPONENTS:
                print(f"  [WARN] unknown component '{comp}' in {path} – skipping")
                continue

            result[comp][n].append(t)

    return {k: dict(v) for k, v in result.items()}


def avg_std(times_dict, node_list):
    avgs, stds = [], []
    for n in node_list:
        ts = times_dict.get(n, [])
        avgs.append(np.mean(ts) if ts else np.nan)
        stds.append(np.std(ts)  if len(ts) > 1 else 0.0)
    return np.array(avgs), np.array(stds)


def speedup(base_avgs, opt_avgs):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            (opt_avgs > 0) & ~np.isnan(opt_avgs),
            base_avgs / opt_avgs,
            np.nan,
        )


print("Loading data …")

comp_unopt    = parse_component_file(COMPONENT_UNOPT_FILE)
comp_opt      = parse_component_file(COMPONENT_OPT_FILE)
comp_dask_un  = parse_component_file(DASK_UNOPT_FILE)
comp_dask_opt = parse_component_file(DASK_OPT_FILE)

full_unopt = HARDCODED_FULL_UNOPT
full_gpu   = dict(HARDCODED_FULL_GPU) or comp_opt.get("n_nodes", {})

_all: set = set(full_unopt.keys()) | set(full_gpu.keys())
for d in [comp_unopt, comp_opt, comp_dask_un, comp_dask_opt]:
    for v in d.values():
        _all |= set(v.keys())

N_NODES = sorted(_all) if _all else [200, 500, 2000]
x       = np.arange(len(N_NODES))
labels  = [str(n) for n in N_NODES]

print(f"  Node sizes         : {N_NODES}")
print(f"  Unopt components   : {list(comp_unopt.keys())}")
print(f"  Opt   components   : {list(comp_opt.keys())}")
print(f"  Dask-unopt comps   : {list(comp_dask_un.keys())}")
print(f"  Dask-opt   comps   : {list(comp_dask_opt.keys())}")

avgs_u,  stds_u  = avg_std(full_unopt, N_NODES)
avgs_g,  stds_g  = avg_std(full_gpu,   N_NODES)
sp_full           = speedup(avgs_u, avgs_g)

avgs_pu, stds_pu = avg_std(comp_unopt.get("phase_coherence", {}), N_NODES)
avgs_po, stds_po = avg_std(comp_opt.get  ("phase_coherence", {}), N_NODES)
sp_pc             = speedup(avgs_pu, avgs_po)

avgs_mu, stds_mu = avg_std(comp_unopt.get("mean_frequency", {}), N_NODES)
avgs_mo, stds_mo = avg_std(comp_opt.get  ("mean_frequency", {}), N_NODES)
sp_mf             = speedup(avgs_mu, avgs_mo)

avgs_dun, stds_dun = avg_std(comp_dask_un.get ("dask_per_sim", {}), N_NODES)
avgs_dop, stds_dop = avg_std(comp_dask_opt.get("dask_per_sim", {}), N_NODES)
sp_dask_un          = speedup(avgs_u,   avgs_dun)
sp_dask_opt         = speedup(avgs_u,   avgs_dop)
sp_dask_head        = speedup(avgs_dun, avgs_dop)

C_UNOPT    = "#e07b54"
C_OPT      = "#4c9be8"
C_DASK_UN  = "#9b59b6"
C_DASK_OPT = "#1abc9c"
C_SPEED    = "#2ecc71"

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

BAR_W = 0.35


def annotate_bars(ax, rects, vals, fmt=".2f"):
    for rect, val in zip(rects, vals):
        if not np.isnan(val):
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + max(h * 0.02, 1e-9),
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=7.5,
            )


def make_speedup_ax(ax, node_list, sp_arr, marker="o", color=C_SPEED, label="Speedup"):
    ax.plot(node_list, sp_arr, f"{marker}-", color=color,
            linewidth=2.2, markersize=7, label=label)
    ax.axhline(1, color="grey", linestyle=":", linewidth=1.2)
    for xi, yi in zip(node_list, sp_arr):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.1f}×", (xi, yi),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Speedup (×)")
    ax.xaxis.set_major_locator(mticker.FixedLocator(node_list))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(n) for n in node_list]))
    ax.legend()


def save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")


if not np.all(np.isnan(avgs_g)):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Full Kuramoto model – Unoptimized vs GPU",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    b1 = ax.bar(x - BAR_W/2, avgs_u, BAR_W, yerr=stds_u, capsize=4,
                color=C_UNOPT, label="Unoptimized", alpha=0.9)
    b2 = ax.bar(x + BAR_W/2, avgs_g, BAR_W, yerr=stds_g, capsize=4,
                color=C_OPT,   label="GPU", alpha=0.9)
    annotate_bars(ax, list(b1)+list(b2), list(avgs_u)+list(avgs_g), fmt=".1f")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Number of nodes"); ax.set_ylabel("Time (s)")
    ax.set_title("Absolute run time"); ax.legend()

    make_speedup_ax(axes[1], N_NODES, sp_full, label="GPU speedup")
    axes[1].set_title("GPU speedup factor")
    plt.tight_layout(); save("01_full_model_run.png")
else:
    print("  [SKIP] Plot 1 – populate HARDCODED_FULL_GPU when ready")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("phase_coherence – Python vs Cython", fontsize=14, fontweight="bold")

ax = axes[0]
b1 = ax.bar(x - BAR_W/2, avgs_pu, BAR_W, yerr=stds_pu, capsize=4,
            color=C_UNOPT, label="Python (unopt)", alpha=0.9)
b2 = ax.bar(x + BAR_W/2, avgs_po, BAR_W, yerr=stds_po, capsize=4,
            color=C_OPT,   label="Cython (opt)",  alpha=0.9)
annotate_bars(ax, list(b1)+list(b2), list(avgs_pu)+list(avgs_po), fmt=".4f")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Number of nodes"); ax.set_ylabel("Time (s)")
ax.set_title("Absolute time"); ax.legend()

make_speedup_ax(axes[1], N_NODES, sp_pc, marker="s", label="Cython speedup")
axes[1].set_title("Cython speedup – phase_coherence")
plt.tight_layout(); save("02_phase_coherence.png")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("mean_frequency – Python vs Cython", fontsize=14, fontweight="bold")

ax = axes[0]
b1 = ax.bar(x - BAR_W/2, avgs_mu, BAR_W, yerr=stds_mu, capsize=4,
            color=C_UNOPT, label="Python (unopt)", alpha=0.9)
b2 = ax.bar(x + BAR_W/2, avgs_mo, BAR_W, yerr=stds_mo, capsize=4,
            color=C_OPT,   label="Cython (opt)",  alpha=0.9)
annotate_bars(ax, list(b1)+list(b2), list(avgs_mu)+list(avgs_mo), fmt=".3f")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Number of nodes"); ax.set_ylabel("Time (s)")
ax.set_title("Absolute time"); ax.legend()

make_speedup_ax(axes[1], N_NODES, sp_mf, marker="^", label="Cython speedup")
axes[1].set_title("Cython speedup – mean_frequency")
plt.tight_layout(); save("03_mean_frequency.png")


if not np.all(np.isnan(avgs_dun)):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Dask (unoptimized workers, {DASK_UNOPT_N_WORKERS} workers)\n"
        "Effective per-sim time vs sequential",
        fontsize=13, fontweight="bold",
    )
    ax = axes[0]
    b1 = ax.bar(x - BAR_W/2, avgs_u,   BAR_W, yerr=stds_u,   capsize=4,
                color=C_UNOPT,   label="Sequential (unopt)", alpha=0.9)
    b2 = ax.bar(x + BAR_W/2, avgs_dun, BAR_W, yerr=stds_dun, capsize=4,
                color=C_DASK_UN, label="Dask per-sim (unopt workers)", alpha=0.9)
    annotate_bars(ax, list(b1)+list(b2), list(avgs_u)+list(avgs_dun))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Number of nodes"); ax.set_ylabel("Time (s)")
    ax.set_title("Sequential vs Dask per-sim"); ax.legend()

    make_speedup_ax(axes[1], N_NODES, sp_dask_un, marker="D",
                    color=C_DASK_UN, label="Dask (unopt) speedup")
    axes[1].set_title("Dask (unopt) effective speedup")
    plt.tight_layout(); save("04_dask_unoptimized.png")
else:
    print("  [SKIP] Plot 4 – run time_dask_unoptimized.py first")


if not np.all(np.isnan(avgs_dop)):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Dask (optimized workers, {DASK_OPT_N_WORKERS} workers)\n"
        "Effective per-sim time vs sequential",
        fontsize=13, fontweight="bold",
    )
    ax = axes[0]
    b1 = ax.bar(x - BAR_W/2, avgs_u,   BAR_W, yerr=stds_u,   capsize=4,
                color=C_UNOPT,    label="Sequential (unopt)", alpha=0.9)
    b2 = ax.bar(x + BAR_W/2, avgs_dop, BAR_W, yerr=stds_dop, capsize=4,
                color=C_DASK_OPT, label="Dask per-sim (opt workers)", alpha=0.9)
    annotate_bars(ax, list(b1)+list(b2), list(avgs_u)+list(avgs_dop))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Number of nodes"); ax.set_ylabel("Time (s)")
    ax.set_title("Sequential vs Dask-opt per-sim"); ax.legend()

    make_speedup_ax(axes[1], N_NODES, sp_dask_opt, marker="D",
                    color=C_DASK_OPT, label="Dask (opt) speedup")
    axes[1].set_title("Dask (opt) effective speedup")
    plt.tight_layout(); save("05_dask_optimized.png")
else:
    print("  [SKIP] Plot 5 – run time_dask_optimized.py first")


if not np.all(np.isnan(avgs_dun)) and not np.all(np.isnan(avgs_dop)):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Dask head-to-head – unoptimized vs optimized workers",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    b1 = ax.bar(x - BAR_W/2, avgs_dun, BAR_W, yerr=stds_dun, capsize=4,
                color=C_DASK_UN,  label="Dask per-sim (unopt)", alpha=0.9)
    b2 = ax.bar(x + BAR_W/2, avgs_dop, BAR_W, yerr=stds_dop, capsize=4,
                color=C_DASK_OPT, label="Dask per-sim (opt)",   alpha=0.9)
    annotate_bars(ax, list(b1)+list(b2), list(avgs_dun)+list(avgs_dop))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Number of nodes"); ax.set_ylabel("Time (s)")
    ax.set_title("Dask per-sim: unopt vs opt"); ax.legend()

    make_speedup_ax(axes[1], N_NODES, sp_dask_head, marker="P",
                    color=C_DASK_OPT, label="Opt speedup over unopt")
    axes[1].set_title("Dask opt speedup over Dask unopt")
    plt.tight_layout(); save("06_dask_head_to_head.png")
else:
    print("  [SKIP] Plot 6 – need both Dask result files")


fig, ax = plt.subplots(figsize=(11, 5))
ax.set_title("Speedup overview – all optimizations", fontsize=14, fontweight="bold")

series = [
    (sp_full,      "o-",  "#e74c3c", "Full model (GPU vs sequential)"),
    (sp_pc,        "s--", "#3498db", "phase_coherence (Cython vs Python)"),
    (sp_mf,        "^:",  "#2ecc71", "mean_frequency (Cython vs Python)"),
    (sp_dask_un,   "D-.", C_DASK_UN,  f"Dask unopt per-sim ({DASK_UNOPT_N_WORKERS}w vs sequential)"),
    (sp_dask_opt,  "P-.", C_DASK_OPT, f"Dask opt per-sim ({DASK_OPT_N_WORKERS}w vs sequential)"),
]

for sp_arr, style, color, label in series:
    if not np.all(np.isnan(sp_arr)):
        ax.plot(N_NODES, sp_arr, style, color=color,
                linewidth=2.2, markersize=8, label=label)

ax.axhline(1, color="grey", linestyle=":", linewidth=1.2, label="No speedup (1×)")
ax.set_xlabel("Number of nodes", fontsize=12)
ax.set_ylabel("Speedup factor (×)", fontsize=12)
ax.xaxis.set_major_locator(mticker.FixedLocator(N_NODES))
ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
ax.legend(fontsize=9)
plt.tight_layout(); save("07_speedup_overview.png")

plt.show()
print("\nDone. All plots saved to:", OUTPUT_DIR)