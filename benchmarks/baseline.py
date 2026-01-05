import neighborlist_rs
import numpy as np
import time
import json
import psutil
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


def run_baseline():
    systems = [10000, 20000, 100000]
    cutoffs = [6.0, 14.0]
    n_threads = psutil.cpu_count(logical=True)
    n_repeat = 5

    results = {}

    # Ensure parallel is set correctly
    neighborlist_rs.set_num_threads(n_threads)

    for n_atoms in systems:
        print(f"Benchmarking {n_atoms} atoms...")
        atoms = generate_ethanol_system(n_atoms)
        pos = atoms.get_positions()
        h_T = atoms.get_cell()[:].T.tolist()

        results[str(n_atoms)] = {}

        for cutoff in cutoffs:
            print(f"  Cutoff {cutoff}...")
            times = []
            for _ in range(n_repeat):
                cell = neighborlist_rs.PyCell(h_T)
                start = time.perf_counter()
                neighborlist_rs.build_neighborlists(cell, pos, cutoff, parallel=True)
                end = time.perf_counter()
                times.append(end - start)

            mean_time = np.mean(times)
            results[str(n_atoms)][str(cutoff)] = {
                "mean_ms": mean_time * 1000,
                "std_ms": np.std(times) * 1000,
                "repeats": n_repeat,
            }

    output_path = "benchmarks/baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline results saved to {output_path}")


if __name__ == "__main__":
    run_baseline()
