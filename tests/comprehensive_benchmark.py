import time
import numpy as np
import os
import psutil
import subprocess
import sys
from ase.io import read
import vesin
import neighborlist_rs
import torch
import torch_nl
from torch_cluster import radius_graph

# Benchmark settings
CUTOFFS = [6.0, 14.0, 20.0]
N_REPEAT = 3  # Reduced repeat for faster execution of more cases

def benchmark_neighborlist_rs_worker(filepath, cutoff, n_threads):
    """Executes the benchmark in a subprocess to ensure Rayon pool is sized correctly."""
    use_parallel = "True" if n_threads > 1 else "False"
    cmd = [
        sys.executable, "-c",
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

times = []
for _ in range({N_REPEAT}):
    start = time.perf_counter()
    neighborlist_rs.build_neighborlists(cell_obj, pos, {cutoff}, parallel={use_parallel})
    end = time.perf_counter()
    times.append(end - start)
print(np.mean(times))
"""
    ]
    res = subprocess.check_output(cmd).decode().strip()
    # The last line should be the mean time
    last_line = res.splitlines()[-1]
    return float(last_line)

def run_benchmarks():
    # Define system configurations
    # (Display Name, Filepath)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    configs = [
        ("100 (non-PBC)", os.path.join(data_dir, "isolated_100.xyz")),
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
        # Note: We rely on the worker to handle the 'None' cell logic for timing.
        # For correctness check below, we need to handle it here too.
        systems_data.append((name, filepath, atoms))

    for cutoff in CUTOFFS:
        print(f"\n{'='*80}")
        print(f"BENCHMARK CUTOFF: {cutoff} Angstroms")
        print(f"{'='*80}")
        print(f"{'System':<30} | {'Lib':<15} | {'Threads':<5} | {'Time (ms)':<10}")
        print("-" * 80)
        
        for name, filepath, atoms in systems_data:
            # --- Correctness Check ---
            # Using neighborlist-rs (20 CPU) vs Vesin
            pos = atoms.get_positions()
            box = atoms.get_cell()[:]
            
            # Vesin needs a box. If isolated, we must give it one or set periodic=False.
            if np.all(box == 0):
                # Dynamically compute bounding box for Vesin (mirroring Rust logic)
                # Rust uses: span + 2 * (cutoff + 1.0)
                min_bound = np.min(pos, axis=0)
                max_bound = np.max(pos, axis=0)
                span = max_bound - min_bound
                margin = cutoff + 1.0
                L = span + 2.0 * margin
                box = np.diag(L)
                periodic = False # Explicitly tell Vesin it's non-PBC
            else:
                periodic = True

            calc_vesin = vesin.NeighborList(cutoff=cutoff, full_list=False)
            i_v, j_v, _ = calc_vesin.compute(pos, box, periodic=periodic, quantities="ijS")
            
            # neighborlist-rs setup
            if np.all(atoms.get_cell() == 0):
                cell_rs = None
            else:
                h_T = atoms.get_cell()[:].T.tolist()
                cell_rs = neighborlist_rs.PyCell(h_T)

            # Use parallel=True for correctness check (default behavior)
            res_rs = neighborlist_rs.build_neighborlists(cell_rs, pos, cutoff, parallel=True)
            i_rs, j_rs = res_rs["local"]["edge_i"], res_rs["local"]["edge_j"]
            
            # Filter out self-interactions from Vesin to match neighborlist-rs behavior
            p_v = set((min(u, v), max(u, v)) for u, v in zip(i_v, j_v) if u != v)
            p_rs = set((min(u, v), max(u, v)) for u, v in zip(i_rs, j_rs))
            
            # Count total edges (excluding self-loops for consistency with p_v/p_rs logic if needed, 
            # but usually total edges is what matters for performance/correctness of MD)
            # Vesin i_v includes self-loops if periodic=True and we didn't filter them from the list itself, just the set.
            # Let's count filtered edges to be strictly comparable
            edges_v = sum(1 for u, v in zip(i_v, j_v) if u != v)
            edges_rs = len(i_rs) # RS already excludes self-loops

            if p_v != p_rs:
                print(f"ERROR: Mismatch for {name} (Cutoff {cutoff})! Unique Pairs V:{len(p_v)} RS:{len(p_rs)}")
            else:
                print(f"Correctness: PASSED for {name} (Cutoff {cutoff}) - Unique Pairs V:{len(p_v)} RS:{len(p_rs)} | Total Edges V:{edges_v} RS:{edges_rs}")
            
            # Vesin Benchmark
            t_v = []
            for _ in range(N_REPEAT):
                calculator = vesin.NeighborList(cutoff=cutoff, full_list=False)
                start = time.perf_counter()
                calculator.compute(pos, box, periodic=periodic)
                end = time.perf_counter()
                t_v.append(end - start)
            print(f"{name:<30} | {'Vesin':<15} | {'Auto':<5} | {np.mean(t_v)*1000:<10.2f}")
            
            # neighborlist-rs (1 CPU)
            t1 = benchmark_neighborlist_rs_worker(filepath, cutoff, 1)
            print(f"{name:<30} | {'neighborlist-rs':<15} | {'1':<5} | {t1*1000:<10.2f}")
            
            # neighborlist-rs (Max CPUs)
            n_cpus = min(8, psutil.cpu_count(logical=True))
            t20 = benchmark_neighborlist_rs_worker(filepath, cutoff, n_cpus)
            print(f"{name:<30} | {'neighborlist-rs':<15} | {n_cpus:<5} | {t20*1000:<10.2f}")
            print("-" * 80)

if __name__ == "__main__":
    run_benchmarks()
