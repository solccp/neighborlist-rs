#!/usr/bin/env python3
"""
Compare benchmark results between Grace and EPYC7R13 machines
"""

import re
import matplotlib.pyplot as plt
import numpy as np


def parse_benchmark_file(filepath):
    """Parse benchmark file and extract timing data"""
    with open(filepath, "r") as f:
        content = f.read()

    # Extract all benchmark entries
    results = []
    current_cutoff = None

    for line in content.split("\n"):
        # Extract cutoff
        cutoff_match = re.search(r"BENCHMARK CUTOFF: ([\d.]+)", line)
        if cutoff_match:
            current_cutoff = float(cutoff_match.group(1))
            continue

        # Extract timing data
        # Format: System | Lib | Threads | Time (ms)
        if (
            "|" in line
            and "Time (ms)" not in line
            and "---" not in line
            and "Correctness" not in line
        ):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4 and parts[0] and parts[1] and parts[2] and parts[3]:
                system = parts[0]
                lib = parts[1]
                threads = parts[2]
                time_str = parts[3]

                # Parse time: "123.45 ± 1.23"
                time_match = re.search(r"([\d.]+)\s*±\s*([\d.]+)", time_str)
                if time_match:
                    mean = float(time_match.group(1))
                    std = float(time_match.group(2))

                    results.append(
                        {
                            "cutoff": current_cutoff,
                            "system": system,
                            "lib": lib,
                            "threads": threads,
                            "mean": mean,
                            "std": std,
                        }
                    )

    return results


def create_comparison_plots(grace_data, epyc_data):
    """Create comparison plots for different scenarios"""

    # Filter for neighborlist-rs multi-threaded results only
    grace_multi = [
        r
        for r in grace_data
        if r["lib"] == "neighborlist-rs" and r["threads"] not in ["1", "Auto"]
    ]
    epyc_multi = [
        r
        for r in epyc_data
        if r["lib"] == "neighborlist-rs" and r["threads"] not in ["1", "Auto"]
    ]

    # Group by cutoff
    cutoffs = sorted(set(r["cutoff"] for r in grace_multi))

    for cutoff in cutoffs:
        grace_cutoff = [r for r in grace_multi if r["cutoff"] == cutoff]
        epyc_cutoff = [r for r in epyc_multi if r["cutoff"] == cutoff]

        # Create a mapping by system name
        systems = sorted(set(r["system"] for r in grace_cutoff))

        # Prepare data
        grace_means = []
        grace_stds = []
        epyc_means = []
        epyc_stds = []
        labels = []

        for system in systems:
            grace_entry = next((r for r in grace_cutoff if r["system"] == system), None)
            epyc_entry = next((r for r in epyc_cutoff if r["system"] == system), None)

            if grace_entry and epyc_entry:
                grace_means.append(grace_entry["mean"])
                grace_stds.append(grace_entry["std"])
                epyc_means.append(epyc_entry["mean"])
                epyc_stds.append(epyc_entry["std"])
                # Shorten labels
                label = system.replace("(non-PBC)", "").replace("(PBC)", "PBC").strip()
                labels.append(label)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(
            x - width / 2,
            grace_means,
            width,
            yerr=grace_stds,
            label="Grace (20 threads)",
            capsize=5,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            epyc_means,
            width,
            yerr=epyc_stds,
            label="EPYC7R13 (64 threads)",
            capsize=5,
            alpha=0.8,
        )

        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_xlabel("System", fontsize=12)
        ax.set_title(
            f"Performance Comparison: neighborlist-rs Multi-threaded\nCutoff: {cutoff} Å",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Use log scale if there's a large range
        if (
            max(grace_means + epyc_means)
            / min(m for m in grace_means + epyc_means if m > 0)
            > 100
        ):
            ax.set_yscale("log")
            ax.set_ylabel("Time (ms, log scale)", fontsize=12)

        plt.tight_layout()
        plt.savefig(
            f"benchmark_comparison_cutoff_{cutoff}A.png", dpi=300, bbox_inches="tight"
        )
        print(f"Saved: benchmark_comparison_cutoff_{cutoff}A.png")
        plt.close()

    # Create a combined plot for large systems across all cutoffs
    create_large_system_comparison(grace_data, epyc_data)

    # Create speedup plot
    create_speedup_plot(grace_data, epyc_data)


def create_large_system_comparison(grace_data, epyc_data):
    """Create comparison for large systems (10,000 and 20,000) across cutoffs"""

    # Filter for large systems, multi-threaded neighborlist-rs
    large_systems = [
        "10,000 (non-PBC)",
        "20,000 (non-PBC)",
        "10,000 (Ethanol PBC)",
        "20,000 (Ethanol PBC)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, system in enumerate(large_systems):
        ax = axes[idx]

        grace_system = [
            r
            for r in grace_data
            if r["system"] == system
            and r["lib"] == "neighborlist-rs"
            and r["threads"] not in ["1", "Auto"]
        ]

        epyc_system = [
            r
            for r in epyc_data
            if r["system"] == system
            and r["lib"] == "neighborlist-rs"
            and r["threads"] not in ["1", "Auto"]
        ]

        cutoffs = sorted(set(r["cutoff"] for r in grace_system))

        grace_means = [
            next(r["mean"] for r in grace_system if r["cutoff"] == c) for c in cutoffs
        ]
        grace_stds = [
            next(r["std"] for r in grace_system if r["cutoff"] == c) for c in cutoffs
        ]
        epyc_means = [
            next(r["mean"] for r in epyc_system if r["cutoff"] == c) for c in cutoffs
        ]
        epyc_stds = [
            next(r["std"] for r in epyc_system if r["cutoff"] == c) for c in cutoffs
        ]

        x = np.arange(len(cutoffs))
        width = 0.35

        ax.bar(
            x - width / 2,
            grace_means,
            width,
            yerr=grace_stds,
            label="Grace (20 threads)",
            capsize=5,
            alpha=0.8,
            color="#2E86AB",
        )
        ax.bar(
            x + width / 2,
            epyc_means,
            width,
            yerr=epyc_stds,
            label="EPYC7R13 (64 threads)",
            capsize=5,
            alpha=0.8,
            color="#A23B72",
        )

        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_xlabel("Cutoff (Å)", fontsize=11)
        ax.set_title(system, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.1f}" for c in cutoffs])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Use log scale for better visualization
        ax.set_yscale("log")

    plt.suptitle(
        "Large System Performance: neighborlist-rs Multi-threaded",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig("benchmark_comparison_large_systems.png", dpi=300, bbox_inches="tight")
    print("Saved: benchmark_comparison_large_systems.png")
    plt.close()


def create_speedup_plot(grace_data, epyc_data):
    """Create speedup plot (Grace vs EPYC)"""

    # Filter for multi-threaded neighborlist-rs
    grace_multi = [
        r
        for r in grace_data
        if r["lib"] == "neighborlist-rs" and r["threads"] not in ["1", "Auto"]
    ]
    epyc_multi = [
        r
        for r in epyc_data
        if r["lib"] == "neighborlist-rs" and r["threads"] not in ["1", "Auto"]
    ]

    # Group by cutoff
    cutoffs = sorted(set(r["cutoff"] for r in grace_multi))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, cutoff in enumerate(cutoffs):
        ax = axes[idx]

        grace_cutoff = [r for r in grace_multi if r["cutoff"] == cutoff]
        epyc_cutoff = [r for r in epyc_multi if r["cutoff"] == cutoff]

        systems = sorted(set(r["system"] for r in grace_cutoff))

        speedups = []
        labels = []

        for system in systems:
            grace_entry = next((r for r in grace_cutoff if r["system"] == system), None)
            epyc_entry = next((r for r in epyc_cutoff if r["system"] == system), None)

            if grace_entry and epyc_entry and grace_entry["mean"] > 0:
                speedup = epyc_entry["mean"] / grace_entry["mean"]
                speedups.append(speedup)
                label = system.replace("(non-PBC)", "").replace("(PBC)", "PBC").strip()
                labels.append(label)

        x = np.arange(len(labels))
        bars = ax.barh(x, speedups, alpha=0.8, color="#06A77D")

        # Add speedup values on bars
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax.text(
                speedup + 0.05,
                i,
                f"{speedup:.2f}x",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.axvline(
            x=1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Equal performance",
        )
        ax.set_xlabel("Speedup (EPYC time / Grace time)", fontsize=11)
        ax.set_title(f"Cutoff: {cutoff} Å", fontsize=12, fontweight="bold")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        ax.set_xlim(0, max(speedups) * 1.15)

    plt.suptitle(
        "Grace Performance Advantage (Speedup > 1.0 means Grace is faster)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("benchmark_speedup_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: benchmark_speedup_comparison.png")
    plt.close()


def create_single_thread_comparison(grace_data, epyc_data):
    """Create comparison for single-threaded performance"""

    # Filter for single-threaded neighborlist-rs
    grace_single = [
        r for r in grace_data if r["lib"] == "neighborlist-rs" and r["threads"] == "1"
    ]
    epyc_single = [
        r for r in epyc_data if r["lib"] == "neighborlist-rs" and r["threads"] == "1"
    ]

    # Group by cutoff
    cutoffs = sorted(set(r["cutoff"] for r in grace_single))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    large_systems = [
        "10,000 (non-PBC)",
        "20,000 (non-PBC)",
        "10,000 (Ethanol PBC)",
        "20,000 (Ethanol PBC)",
    ]

    for idx, system in enumerate(large_systems):
        ax = axes[idx]

        grace_system = [r for r in grace_single if r["system"] == system]
        epyc_system = [r for r in epyc_single if r["system"] == system]

        cutoffs = sorted(set(r["cutoff"] for r in grace_system))

        grace_means = [
            next(r["mean"] for r in grace_system if r["cutoff"] == c) for c in cutoffs
        ]
        grace_stds = [
            next(r["std"] for r in grace_system if r["cutoff"] == c) for c in cutoffs
        ]
        epyc_means = [
            next(r["mean"] for r in epyc_system if r["cutoff"] == c) for c in cutoffs
        ]
        epyc_stds = [
            next(r["std"] for r in epyc_system if r["cutoff"] == c) for c in cutoffs
        ]

        x = np.arange(len(cutoffs))
        width = 0.35

        ax.bar(
            x - width / 2,
            grace_means,
            width,
            yerr=grace_stds,
            label="Grace (1 thread)",
            capsize=5,
            alpha=0.8,
            color="#2E86AB",
        )
        ax.bar(
            x + width / 2,
            epyc_means,
            width,
            yerr=epyc_stds,
            label="EPYC7R13 (1 thread)",
            capsize=5,
            alpha=0.8,
            color="#A23B72",
        )

        # Add speedup annotations
        for i, (grace_mean, epyc_mean) in enumerate(zip(grace_means, epyc_means)):
            if grace_mean > 0:
                speedup = epyc_mean / grace_mean
                max_height = max(grace_mean, epyc_mean)
                ax.text(
                    i,
                    max_height * 1.3,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#F77F00",
                )

        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_xlabel("Cutoff (Å)", fontsize=11)
        ax.set_title(system, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.1f}" for c in cutoffs])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Use log scale for better visualization
        ax.set_yscale("log")

    plt.suptitle(
        "Single-Threaded Performance Comparison: neighborlist-rs (1 CPU)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig("benchmark_comparison_single_thread.png", dpi=300, bbox_inches="tight")
    print("Saved: benchmark_comparison_single_thread.png")
    plt.close()

    # Create single-thread speedup plot
    create_single_thread_speedup(grace_data, epyc_data)


def create_single_thread_speedup(grace_data, epyc_data):
    """Create speedup plot for single-threaded performance"""

    grace_single = [
        r for r in grace_data if r["lib"] == "neighborlist-rs" and r["threads"] == "1"
    ]
    epyc_single = [
        r for r in epyc_data if r["lib"] == "neighborlist-rs" and r["threads"] == "1"
    ]

    cutoffs = sorted(set(r["cutoff"] for r in grace_single))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, cutoff in enumerate(cutoffs):
        ax = axes[idx]

        grace_cutoff = [r for r in grace_single if r["cutoff"] == cutoff]
        epyc_cutoff = [r for r in epyc_single if r["cutoff"] == cutoff]

        systems = sorted(set(r["system"] for r in grace_cutoff))

        speedups = []
        labels = []

        for system in systems:
            grace_entry = next((r for r in grace_cutoff if r["system"] == system), None)
            epyc_entry = next((r for r in epyc_cutoff if r["system"] == system), None)

            if grace_entry and epyc_entry and grace_entry["mean"] > 0:
                speedup = epyc_entry["mean"] / grace_entry["mean"]
                speedups.append(speedup)
                label = system.replace("(non-PBC)", "").replace("(PBC)", "PBC").strip()
                labels.append(label)

        x = np.arange(len(labels))
        bars = ax.barh(x, speedups, alpha=0.8, color="#F77F00")

        # Add speedup values on bars
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax.text(
                speedup + 0.05,
                i,
                f"{speedup:.2f}x",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.axvline(
            x=1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Equal performance",
        )
        ax.set_xlabel("Speedup (EPYC time / Grace time)", fontsize=11)
        ax.set_title(f"Cutoff: {cutoff} Å", fontsize=12, fontweight="bold")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        ax.set_xlim(0, max(speedups) * 1.15)

    plt.suptitle(
        "Single-Threaded Performance Advantage (Speedup > 1.0 means Grace is faster)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("benchmark_speedup_single_thread.png", dpi=300, bbox_inches="tight")
    print("Saved: benchmark_speedup_single_thread.png")
    plt.close()


def main():
    # Parse both files
    grace_data = parse_benchmark_file("benchmark_grace.txt")
    epyc_data = parse_benchmark_file("benchmark_EPYC7R13.txt")

    print(f"Parsed {len(grace_data)} entries from Grace benchmark")
    print(f"Parsed {len(epyc_data)} entries from EPYC7R13 benchmark")

    # Create plots
    create_comparison_plots(grace_data, epyc_data)
    create_single_thread_comparison(grace_data, epyc_data)

    print("\nPlots generated successfully!")


if __name__ == "__main__":
    main()
