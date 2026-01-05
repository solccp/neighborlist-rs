import time
import numpy as np
import torch
from torch_cluster import radius_graph
from torch_nl import compute_neighborlist
import neighborlist_rs
import vesin
from ase.build import molecule

# Benchmark settings
N_SYSTEMS = 128
CUTOFF = 6.0
N_REPEAT = 10


def get_real_molecules():
    # Mix of medium-sized molecules
    mol_names = ["C6H6", "CH3CH2OH", "C6H12", "C8H10", "C10H8"]
    mols = []
    while len(mols) < N_SYSTEMS:
        for name in mol_names:
            if len(mols) >= N_SYSTEMS:
                break
            try:
                m = molecule(name)
                # Ensure 30-50 atoms
                if len(m) < 30:
                    m.set_cell([10, 10, 10])
                    m = m * (2, 1, 1)
                if len(m) < 30:
                    m = m * (2, 1, 1)
                mols.append(m)
            except Exception:
                continue
    return mols


def run_batch_benchmark():
    mols = get_real_molecules()

    # Flatten everything for batching
    all_pos = []
    batch_idx = []
    for i, m in enumerate(mols):
        pos = m.get_positions()
        all_pos.append(pos)
        batch_idx.append(np.full(len(m), i, dtype=np.int32))

    pos_np = np.concatenate(all_pos).astype(np.float64)
    batch_np = np.concatenate(batch_idx)
    pos_torch = torch.from_numpy(pos_np).float()
    batch_torch = torch.from_numpy(batch_np).long()

    print(f"Batch Benchmark (NON-PBC): {N_SYSTEMS} systems, {len(pos_np)} total atoms.")
    print(f"Average atoms per system: {len(pos_np) / N_SYSTEMS:.1f}")
    print("-" * 60)

    # 1. torch_cluster.radius_graph
    # Warmup
    _ = radius_graph(
        pos_torch, r=CUTOFF, batch=batch_torch, loop=False, max_num_neighbors=1000
    )
    start = time.perf_counter()
    for _ in range(N_REPEAT):
        _ = radius_graph(
            pos_torch, r=CUTOFF, batch=batch_torch, loop=False, max_num_neighbors=1000
        )
    t_torch_cluster = (time.perf_counter() - start) / N_REPEAT * 1000
    print(f"torch_cluster.radius_graph: {t_torch_cluster:.2f} ms")

    # 2. torch-nl
    # Note: torch-nl expects (cutoff, pos, cell, pbc, batch)
    try:
        # Warmup
        _ = compute_neighborlist(
            CUTOFF,
            pos_torch,
            torch.zeros((N_SYSTEMS, 3, 3)),
            torch.zeros((N_SYSTEMS, 3), dtype=torch.bool),
            batch_torch,
        )
        start = time.perf_counter()
        for _ in range(N_REPEAT):
            _ = compute_neighborlist(
                CUTOFF,
                pos_torch,
                torch.zeros((N_SYSTEMS, 3, 3)),
                torch.zeros((N_SYSTEMS, 3), dtype=torch.bool),
                batch_torch,
            )
        t_torch_nl = (time.perf_counter() - start) / N_REPEAT * 1000
        print(f"torch-nl:                   {t_torch_nl:.2f} ms")
    except Exception as e:
        print(f"torch-nl failed: {e}")

    # 3. vesin (Looping over systems)
    # Vesin doesn't have a batch API, so we loop (representative of standard use)
    start = time.perf_counter()
    for _ in range(N_REPEAT):
        for i, m in enumerate(mols):
            pos = m.get_positions()
            # Vesin needs a box
            box = np.diag([50.0, 50.0, 50.0])
            calc = vesin.NeighborList(cutoff=CUTOFF, full_list=False)
            _ = calc.compute(pos, box, periodic=False)
    t_vesin = (time.perf_counter() - start) / N_REPEAT * 1000
    print(f"vesin (serial loop):        {t_vesin:.2f} ms")

    # 4. neighborlist-rs (1 CPU)
    # Warmup
    _ = neighborlist_rs.build_neighborlists_batch(
        pos_np, batch_np, None, CUTOFF, parallel=False
    )
    start = time.perf_counter()
    for _ in range(N_REPEAT):
        _ = neighborlist_rs.build_neighborlists_batch(
            pos_np, batch_np, None, CUTOFF, parallel=False
        )
    t_rs_1 = (time.perf_counter() - start) / N_REPEAT * 1000
    print(f"neighborlist-rs (1 CPU):    {t_rs_1:.2f} ms")

    # 5. neighborlist-rs (8 CPU)
    # Warmup
    _ = neighborlist_rs.build_neighborlists_batch(
        pos_np, batch_np, None, CUTOFF, parallel=True
    )
    start = time.perf_counter()
    for _ in range(N_REPEAT):
        _ = neighborlist_rs.build_neighborlists_batch(
            pos_np, batch_np, None, CUTOFF, parallel=True
        )
    t_rs_8 = (time.perf_counter() - start) / N_REPEAT * 1000
    print(f"neighborlist-rs (8 CPU):    {t_rs_8:.2f} ms")


if __name__ == "__main__":
    run_batch_benchmark()
