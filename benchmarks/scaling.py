import numpy as np
import json
import psutil
import subprocess
import sys
from ase.build import molecule
from ase import Atoms


def generate_ethanol_system(target_n_atoms, density=1.0):
    eth = molecule("CH3CH2OH")
    n_atoms_per_mol = len(eth)
    n_mols = int(np.ceil(target_n_atoms / n_atoms_per_mol))
    vol = (n_mols * 46.07) / (0.6022 * density)
    box_len = vol ** (1 / 3)
    atoms = Atoms(cell=[box_len, box_len, box_len], pbc=True)
    rng = np.random.default_rng(42)
    for _ in range(n_mols):
        mol = eth.copy()
        mol.rotate(rng.random() * 360, "x")
        mol.rotate(rng.random() * 360, "y")
        mol.rotate(rng.random() * 360, "z")
        pos = rng.random(3) * box_len
        mol.translate(pos)
        atoms += mol
    return atoms


def benchmark_worker(n_atoms, cutoff, n_threads, n_repeat):
    cmd = [
        sys.executable,
        "-c",
        f"""
import os
os.environ["RAYON_NUM_THREADS"] = "{n_threads}"
import neighborlist_rs
import numpy as np
import time
from ase.build import molecule
from ase import Atoms

def generate_ethanol_system(target_n_atoms, density=1.0):
    eth = molecule('CH3CH2OH')
    n_atoms_per_mol = len(eth)
    n_mols = int(np.ceil(target_n_atoms / n_atoms_per_mol))
    vol = (n_mols * 46.07) / (0.6022 * density)
    box_len = vol ** (1/3)
    atoms = Atoms(cell=[box_len, box_len, box_len], pbc=True)
    rng = np.random.default_rng(42)
    for _ in range(n_mols):
        mol = eth.copy()
        mol.rotate(rng.random() * 360, 'x')
        mol.rotate(rng.random() * 360, 'y')
        mol.rotate(rng.random() * 360, 'z')
        pos = rng.random(3) * box_len
        mol.translate(pos)
        atoms += mol
    return atoms

atoms = generate_ethanol_system({n_atoms})
pos = atoms.get_positions()
h_T = atoms.get_cell()[:].T.tolist()

times = []
for _ in range({n_repeat}):
    cell = neighborlist_rs.PyCell(h_T)
    start = time.perf_counter()
    neighborlist_rs.build_neighborlists(cell, pos, {cutoff}, parallel=True)
    end = time.perf_counter()
    times.append(end - start)
print(np.mean(times))
""",
    ]
    res = subprocess.check_output(cmd).decode().strip()
    return float(res) * 1000


def run_scaling():
    n_atoms = 50000
    cutoff = 10.0
    max_threads = psutil.cpu_count(logical=True)
    thread_counts = [1, 2, 4, 8, 12, 16, max_threads]
    n_repeat = 3

    print(f"Scaling benchmark for {n_atoms} atoms at cutoff {cutoff}")

    scaling_results = {}

    for n in thread_counts:
        print(f"  Threads: {n}...")
        mean_ms = benchmark_worker(n_atoms, cutoff, n, n_repeat)
        scaling_results[n] = mean_ms
        print(f"    {mean_ms:.2f} ms")

    with open("benchmarks/scaling_results.json", "w") as f:
        json.dump(scaling_results, f, indent=2)


if __name__ == "__main__":
    run_scaling()
