import neighborlist_rs
import numpy as np
import time

def benchmark():
    # 10,000 atoms
    n_atoms = 10000
    box_size = 50.0
    h = [[box_size, 0.0, 0.0], [0.0, box_size, 0.0], [0.0, 0.0, box_size]]
    cell = neighborlist_rs.PyCell(h)
    
    rng = np.random.default_rng(42)
    positions = rng.random((n_atoms, 3)) * box_size
    
    # Cutoff to get roughly ~100 neighbors per atom -> ~1M pairs
    # Vol = 125000. Density = 10000/125000 = 0.08 atoms/A^3
    # Vol_sphere = 4/3 * pi * r^3. 
    # 0.08 * Vol_sphere = 100 => Vol_sphere = 1250. r^3 = 300 => r = 10.7
    cutoff = 6.0 
    
    print(f"Benchmarking {n_atoms} atoms with cutoff {cutoff}...")

    # Warmup
    _ = neighborlist_rs.build_neighborlists(cell, positions, cutoff, parallel=False)

    # Sequential
    start = time.perf_counter()
    res_seq = neighborlist_rs.build_neighborlists(cell, positions, cutoff, parallel=False)
    end = time.perf_counter()
    n_pairs = len(res_seq["local"]["edge_i"])
    print(f"Sequential: {end - start:.4f} seconds ({n_pairs} pairs)")

    # Parallel
    start = time.perf_counter()
    res_par = neighborlist_rs.build_neighborlists(cell, positions, cutoff, parallel=True)
    end = time.perf_counter()
    print(f"Parallel:   {end - start:.4f} seconds ({len(res_par['local']['edge_i'])} pairs)")

if __name__ == "__main__":
    benchmark()
