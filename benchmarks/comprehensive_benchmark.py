import json
import time
import numpy as np
import os
import psutil
import subprocess
import sys
from ase.io import read
import vesin
import neighborlist_rs

# Benchmark settings
CUTOFFS = [6.0, 14.0, 30.0]
N_REPEAT = 15  # Increased for better statistics


def benchmark_neighborlist_rs_worker(filepath, cutoff, n_threads):
    """Executes the benchmark in a subprocess to ensure Rayon pool is sized correctly."""
    use_parallel = "True" if n_threads > 1 else "False"
    cmd = [
        sys.executable,
        "-c",
        f"""
import os
os.environ["RAYON_NUM_THREADS"] = "{n_threads}"
import neighborlist_rs
import numpy as np
import time
from ase.io import read

atoms = read("{filepath}")
pos = atoms.get_positions()

# Determine if we have a valid cell
cell_obj = None
if not np.all(atoms.get_cell() == 0):
    h_T = atoms.get_cell()[:].T.tolist()
    cell_obj = neighborlist_rs.PyCell(h_T)

# Warmup
for _ in range(2):
    neighborlist_rs.build_neighborlists(cell_obj, pos, {cutoff}, parallel={use_parallel})

times = []
for _ in range({N_REPEAT}):
    start = time.perf_counter()
    neighborlist_rs.build_neighborlists(cell_obj, pos, {cutoff}, parallel={use_parallel})
    end = time.perf_counter()
    times.append(end - start)
print(str(np.mean(times)) + "," + str(np.std(times)))
""",
    ]
    res = subprocess.check_output(cmd).decode().strip()
    # The last line should be the mean,std
    last_line = res.splitlines()[-1]
    parts = last_line.split(",")
    return float(parts[0]), float(parts[1])


def run_benchmarks():
    results = {}
    # Define system configurations
    # (Display Name, Filepath)
    data_dir = os.path.join(os.path.dirname(__file__), "../tests/data")
    configs = [
        ("100 (non-PBC)", os.path.join(data_dir, "isolated_100.xyz")),
        ("Two Clusters (non-PBC)", os.path.join(data_dir, "isolated_two_clusters.xyz")),
        ("1,000 (non-PBC)", os.path.join(data_dir, "isolated_1000.xyz")),
        ("10,000 (non-PBC)", os.path.join(data_dir, "isolated_10000.xyz")),
        ("20,000 (non-PBC)", os.path.join(data_dir, "isolated_20000.xyz")),
        ("1,000 (Ethanol PBC)", os.path.join(data_dir, "ethanol_1000.xyz")),
        ("10,000 (Ethanol PBC)", os.path.join(data_dir, "ethanol_10000.xyz")),
        ("20,000 (Ethanol PBC)", os.path.join(data_dir, "ethanol_20000.xyz")),
        ("Si Bulk (PBC)", os.path.join(data_dir, "si_bulk.xyz")),
    ]

    # Pre-load atoms for non-RS libraries
    systems_data = []
    for name, filepath in configs:
        atoms = read(filepath)
        systems_data.append((name, filepath, atoms))

    for cutoff in CUTOFFS:
        results[cutoff] = {}
        print(f"\n{'=' * 80}")
        print(f"BENCHMARK CUTOFF: {cutoff} Angstroms")
        print(f"{'=' * 80}")
        print(f"{'System':<30} | {'Lib':<15} | {'Threads':<5} | {'Time (ms)':<20}")
        print("-" * 80)

        for name, filepath, atoms in systems_data:
            results[cutoff][name] = {}
            # --- Correctness Check ---
            pos = atoms.get_positions()
            box = atoms.get_cell()[:]

            if np.all(box == 0):
                min_bound = np.min(pos, axis=0)
                max_bound = np.max(pos, axis=0)
                span = max_bound - min_bound
                margin = cutoff + 1.0
                L = span + 2.0 * margin
                box = np.diag(L)
                periodic = False
            else:
                periodic = True

            calc_vesin = vesin.NeighborList(cutoff=cutoff, full_list=False)
            i_v, j_v, _ = calc_vesin.compute(
                pos, box, periodic=periodic, quantities="ijS"
            )

            if np.all(atoms.get_cell() == 0):
                cell_rs = None
            else:
                h_T = atoms.get_cell()[:].T.tolist()
                cell_rs = neighborlist_rs.PyCell(h_T)

            res_rs = neighborlist_rs.build_neighborlists(
                cell_rs, pos, cutoff, parallel=True
            )
            edge_index = res_rs["edge_index"]
            i_rs, j_rs = edge_index[0], edge_index[1]

            p_v = set((min(u, v), max(u, v)) for u, v in zip(i_v, j_v) if u != v)
            p_rs = set((min(u, v), max(u, v)) for u, v in zip(i_rs, j_rs) if u != v)

            edges_v = sum(1 for u, v in zip(i_v, j_v) if u != v)
            edges_rs = sum(1 for u, v in zip(i_rs, j_rs) if u != v)

            if p_v != p_rs:
                print(
                    f"ERROR: Mismatch for {name} (Cutoff {cutoff})! Unique Pairs V:{len(p_v)} RS:{len(p_rs)}"
                )
            else:
                print(
                    f"Correctness: PASSED for {name} (Cutoff {cutoff}) - Unique Pairs V:{len(p_v)} RS:{len(p_rs)} | Total Edges V:{edges_v} RS:{edges_rs}"
                )

            # Vesin Benchmark
            t_v = []
            # Warmup
            for _ in range(2):
                vesin.NeighborList(cutoff=cutoff, full_list=False).compute(
                    pos, box, periodic=periodic
                )

            for _ in range(N_REPEAT):
                calculator = vesin.NeighborList(cutoff=cutoff, full_list=False)
                start = time.perf_counter()
                calculator.compute(pos, box, periodic=periodic)
                end = time.perf_counter()
                t_v.append(end - start)

            mean_v = np.mean(t_v) * 1000
            std_v = np.std(t_v) * 1000
            results[cutoff][name]["vesin"] = {"mean": mean_v, "std": std_v}
            print(
                f"{name:<30} | {'Vesin':<15} | {'Auto':<5} | {mean_v:>8.2f} ± {std_v:<6.2f}"
            )

            # neighborlist-rs (1 CPU)
            mean_1, std_1 = benchmark_neighborlist_rs_worker(filepath, cutoff, 1)
            mean_1_ms, std_1_ms = mean_1 * 1000, std_1 * 1000
            results[cutoff][name]["rs_1"] = {"mean": mean_1_ms, "std": std_1_ms}
            print(
                f"{name:<30} | {'neighborlist-rs':<15} | {'1':<5} | {mean_1_ms:>8.2f} ± {std_1_ms:<6.2f}"
            )

            # neighborlist-rs (Max CPUs)
            n_cpus = psutil.cpu_count(logical=True)
            mean_max, std_max = benchmark_neighborlist_rs_worker(
                filepath, cutoff, n_cpus
            )
            mean_max_ms, std_max_ms = mean_max * 1000, std_max * 1000
            results[cutoff][name][f"rs_{n_cpus}"] = {
                "mean": mean_max_ms,
                "std": std_max_ms,
            }
            print(
                f"{name:<30} | {'neighborlist-rs':<15} | {n_cpus:<5} | {mean_max_ms:>8.2f} ± {std_max_ms:<6.2f}"
            )
            print("-" * 80)
    return results


if __name__ == "__main__":
    res = run_benchmarks()
    with open("benchmarks/baseline_results.json", "w") as f:
        json.dump(res, f, indent=4)
