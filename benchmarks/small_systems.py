import time
import numpy as np
import os
import subprocess
import sys
import json
import vesin

# Benchmark settings
SIZES = [50, 100, 200, 500, 1000]
CUTOFF = 6.0
N_REPEAT = 20


def generate_random_system(n_atoms, box_size=20.0):
    positions = np.random.rand(n_atoms, 3) * box_size
    return positions


def benchmark_neighborlist_rs_worker(pos, cutoff, n_threads):
    """Executes the benchmark in a subprocess to ensure Rayon pool is sized correctly."""
    use_parallel = "True" if n_threads > 1 else "False"

    # We pass positions as a temporary file to the subprocess for simplicity
    np.save("tmp_pos.npy", pos)

    cmd = [
        sys.executable,
        "-c",
        f"""
import os
os.environ["RAYON_NUM_THREADS"] = "{n_threads}"
import neighborlist_rs
import numpy as np
import time

pos = np.load("tmp_pos.npy")

times = []
for _ in range({N_REPEAT}):
    start = time.perf_counter()
    neighborlist_rs.build_neighborlists(None, pos, {cutoff}, parallel={use_parallel})
    end = time.perf_counter()
    times.append(end - start)
print(np.mean(times))
""",
    ]
    res = subprocess.check_output(cmd).decode().strip()
    os.remove("tmp_pos.npy")
    return float(res)


def run_benchmarks():
    results = {}

    print(
        f"{'N Atoms':<10} | {'Vesin (ms)':<12} | {'RS-1 (ms)':<12} | {'RS-8 (ms)':<12}"
    )
    print("-" * 55)

    for n in SIZES:
        results[n] = {}
        pos = generate_random_system(n)

        # Determine a safe box for Vesin (non-periodic)
        min_bound = np.min(pos, axis=0)
        max_bound = np.max(pos, axis=0)
        span = max_bound - min_bound
        margin = CUTOFF + 1.0
        L = span + 2.0 * margin
        box = np.diag(L)

        # Vesin Benchmark
        t_v = []
        calculator = vesin.NeighborList(cutoff=CUTOFF, full_list=False)
        for _ in range(N_REPEAT):
            start = time.perf_counter()
            calculator.compute(pos, box, periodic=False)
            end = time.perf_counter()
            t_v.append(end - start)
        results[n]["vesin"] = np.mean(t_v) * 1000

        # neighborlist-rs (1 CPU)
        t1 = benchmark_neighborlist_rs_worker(pos, CUTOFF, 1)
        results[n]["rs_1"] = t1 * 1000

        # neighborlist-rs (8 CPU)
        t8 = benchmark_neighborlist_rs_worker(pos, CUTOFF, 8)
        results[n]["rs_8"] = t8 * 1000

        print(
            f"{n:<10} | {results[n]['vesin']:<12.4f} | {results[n]['rs_1']:<12.4f} | {results[n]['rs_8']:<12.4f}"
        )

    return results


if __name__ == "__main__":
    res = run_benchmarks()
    with open("benchmarks/small_systems_baseline.json", "w") as f:
        # Convert keys to string for JSON
        json.dump({str(k): v for k, v in res.items()}, f, indent=4)
