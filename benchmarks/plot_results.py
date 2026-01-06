import matplotlib.pyplot as plt
import numpy as np

# Data (Ethanol 10k)
# Libraries
libs = ["ASE", "TorchNL (CPU)", "Matscipy", "Freud", "Vesin", "neighborlist-rs"]
# Time in ms (Lower is better)
# 6.0 A
t_6 = [1222.8, 487.8, 50.7, 34.9, 37.9, 19.8]
# 14.0 A
t_14 = [11948.5, 4605.0, 608.5, 485.1, 342.6, 250.5]


def plot_single_scaling():
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(libs))
    width = 0.35

    rects1 = ax.barh(
        y_pos + width / 2, t_6, width, label="Cutoff 6.0 Å", color="#4e79a7"
    )
    rects2 = ax.barh(
        y_pos - width / 2, t_14, width, label="Cutoff 14.0 Å", color="#f28e2b"
    )

    ax.set_xlabel("Time (ms) - Log Scale")
    ax.set_title("Neighbor List Construction Time (Ethanol 10k Atoms)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(libs)
    ax.legend()
    ax.set_xscale("log")

    # Add labels
    ax.bar_label(rects1, padding=3, fmt="%.1f")
    ax.bar_label(rects2, padding=3, fmt="%.1f")

    plt.tight_layout()
    plt.savefig("docs/assets/benchmark_single_scaling.png", dpi=300)
    print("Generated docs/assets/benchmark_single_scaling.png")


def plot_batch_throughput():
    # Batch Size 128
    # Throughput (sys/s) - Higher is better
    libs_batch = ["TorchNL (CPU)", "TorchCluster (CPU)", "neighborlist-rs"]
    thru = [445, 12173, 35293]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(libs_batch))

    rects = ax.barh(y_pos, thru, color="#59a14f")

    ax.set_xlabel("Throughput (systems/sec)")
    ax.set_title("Batch Processing Throughput (128 x Isolated 100 Atoms)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(libs_batch)

    ax.bar_label(rects, padding=3, fmt="%d")

    plt.tight_layout()
    plt.savefig("docs/assets/benchmark_batch_throughput.png", dpi=300)
    print("Generated docs/assets/benchmark_batch_throughput.png")


if __name__ == "__main__":
    plot_single_scaling()
    plot_batch_throughput()
