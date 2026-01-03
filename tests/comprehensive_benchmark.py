import time
import numpy as np
import os
import psutil
import subprocess
import sys
from ase import Atoms
from ase.build import molecule, bulk
import vesin
import neighborlist_rs
import torch
import torch_nl
from torch_cluster import radius_graph

# Benchmark settings
CUTOFFS = [6.0, 14.0, 20.0]
N_REPEAT = 3  # Reduced repeat for faster execution of more cases

def generate_random_molecule(n_atoms, box_size=500.0):
    """Generates a random cluster of atoms in a massive box (effectively non-PBC)."""
    rng = np.random.default_rng(42)
    spread = 2.0 * (n_atoms ** (1/3))
    # Place atoms in the center of the massive box
    positions = (rng.random((n_atoms, 3)) - 0.5) * spread + box_size / 2
    symbols = rng.choice(['C', 'H', 'O', 'N'], size=n_atoms)
    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_size]*3, pbc=True)
    return atoms

def generate_ethanol_system(target_n_atoms, density=1.0):
    """Generates a box of ethanol molecules matching target atom count and density."""
    eth = molecule('CH3CH2OH')
    n_atoms_per_mol = len(eth)
    n_mols = int(np.ceil(target_n_atoms / n_atoms_per_mol))
    vol = (n_mols * 46.07) / (0.6022 * density)
    l = vol ** (1/3)
    atoms = Atoms(cell=[l, l, l], pbc=True)
    rng = np.random.default_rng(42)
    for _ in range(n_mols):
        mol = eth.copy()
        mol.rotate(rng.random() * 360, 'x')
        mol.rotate(rng.random() * 360, 'y')
        mol.rotate(rng.random() * 360, 'z')
        pos = rng.random(3) * l
        mol.translate(pos)
        atoms += mol
    return atoms

def benchmark_neighborlist_rs_worker(n_atoms_type, n_atoms_val, cutoff, n_threads):
    """Executes the benchmark in a subprocess to ensure Rayon pool is sized correctly."""
    cmd = [
        sys.executable, "-c",
        f"""
import os
os.environ["RAYON_NUM_THREADS"] = "{n_threads}"
import neighborlist_rs
import numpy as np
import time
from ase.build import molecule
from ase import Atoms

def generate_random_molecule(n_atoms, box_size=500.0):
    rng = np.random.default_rng(42)
    spread = 2.0 * (n_atoms ** (1/3))
    positions = (rng.random((n_atoms, 3)) - 0.5) * spread + box_size / 2
    symbols = rng.choice(['C', 'H', 'O', 'N'], size=n_atoms)
    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_size]*3, pbc=True)
    return atoms

def generate_ethanol_system(target_n_atoms, density=1.0):
    eth = molecule('CH3CH2OH')
    n_atoms_per_mol = len(eth)
    n_mols = int(np.ceil(target_n_atoms / n_atoms_per_mol))
    vol = (n_mols * 46.07) / (0.6022 * density)
    l = vol ** (1/3)
    atoms = Atoms(cell=[l, l, l], pbc=True)
    rng = np.random.default_rng(42)
    for _ in range(n_mols):
        mol = eth.copy()
        mol.rotate(rng.random() * 360, 'x')
        mol.rotate(rng.random() * 360, 'y')
        mol.rotate(rng.random() * 360, 'z')
        pos = rng.random(3) * l
        mol.translate(pos)
        atoms += mol
    return atoms

if "{n_atoms_type}" == "isolated":
    atoms = generate_random_molecule({n_atoms_val})
elif "{n_atoms_type}" == "ethanol":
    atoms = generate_ethanol_system({n_atoms_val})

pos = atoms.get_positions()
h_T = atoms.get_cell()[:].T.tolist()

times = []
for _ in range({N_REPEAT}):
    cell = neighborlist_rs.PyCell(h_T)
    start = time.perf_counter()
    neighborlist_rs.build_neighborlists(cell, pos, {cutoff}, parallel=True)
    end = time.perf_counter()
    times.append(end - start)
print(np.mean(times))
"""
    ]
    res = subprocess.check_output(cmd).decode().strip()
    return float(res)

def run_benchmarks():
    # Define system configurations
    # (Display Name, Type, N_Atoms)
    configs = [
        ("100 (non-PBC)", "isolated", 100),
        ("1,000 (non-PBC)", "isolated", 1000),
        ("10,000 (non-PBC)", "isolated", 10000),
        ("20,000 (non-PBC)", "isolated", 20000),
        ("1,000 (Ethanol PBC)", "ethanol", 1000),
        ("10,000 (Ethanol PBC)", "ethanol", 10000),
        ("20,000 (Ethanol PBC)", "ethanol", 20000),
    ]
    
    # Pre-generate atoms for non-RS libraries
    systems_data = []
    for name, dtype, val in configs:
        if dtype == "isolated": atoms = generate_random_molecule(val)
        elif dtype == "ethanol": atoms = generate_ethanol_system(val)
        systems_data.append((name, dtype, val, atoms))

    for cutoff in CUTOFFS:
        print(f"\n{'='*80}")
        print(f"BENCHMARK CUTOFF: {cutoff} Angstroms")
        print(f"{'='*80}")
        print(f"{'System':<30} | {'Lib':<15} | {'Threads':<5} | {'Time (ms)':<10}")
        print("-" * 80)
        
        for name, dtype, val, atoms in systems_data:
            # --- Correctness Check ---
            # Using neighborlist-rs (20 CPU) vs Vesin
            pos = atoms.get_positions()
            box = atoms.get_cell()[:]
            calc_vesin = vesin.NeighborList(cutoff=cutoff, full_list=False)
            i_v, j_v, _ = calc_vesin.compute(pos, box, periodic=True, quantities="ijS")
            
            h_T = box.T.tolist()
            cell_rs = neighborlist_rs.PyCell(h_T)
            res_rs = neighborlist_rs.build_neighborlists(cell_rs, pos, cutoff, parallel=True)
            i_rs, j_rs = res_rs["local"]["edge_i"], res_rs["local"]["edge_j"]
            
            p_v = set((min(u, v), max(u, v)) for u, v in zip(i_v, j_v))
            p_rs = set((min(u, v), max(u, v)) for u, v in zip(i_rs, j_rs))
            
            if p_v != p_rs:
                print(f"ERROR: Mismatch for {name} (Cutoff {cutoff})! V:{len(p_v)} RS:{len(p_rs)}")
            else:
                print(f"Correctness: PASSED for {name} (Cutoff {cutoff})")
            
            # Vesin Benchmark
            t_v = []
            for _ in range(N_REPEAT):
                calculator = vesin.NeighborList(cutoff=cutoff, full_list=False)
                start = time.perf_counter()
                calculator.compute(atoms.get_positions(), atoms.get_cell()[:], periodic=True)
                end = time.perf_counter()
                t_v.append(end - start)
            print(f"{name:<30} | {'Vesin':<15} | {'Auto':<5} | {np.mean(t_v)*1000:<10.2f}")
            
            # neighborlist-rs (1 CPU)
            t1 = benchmark_neighborlist_rs_worker(dtype, val, cutoff, 1)
            print(f"{name:<30} | {'neighborlist-rs':<15} | {'1':<5} | {t1*1000:<10.2f}")
            
            # neighborlist-rs (20 CPUs)
            n_cpus = min(20, psutil.cpu_count(logical=True))
            t20 = benchmark_neighborlist_rs_worker(dtype, val, cutoff, n_cpus)
            print(f"{name:<30} | {'neighborlist-rs':<15} | {n_cpus:<5} | {t20*1000:<10.2f}")
            print("-" * 80)

if __name__ == "__main__":
    run_benchmarks()
