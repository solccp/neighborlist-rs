import neighborlist_rs
import numpy as np
import time
from ase.build import molecule


def generate_water_batch(batch_size=128):
    water = molecule("H2O")
    pos_single = water.get_positions()
    n_atoms_single = len(pos_single)

    positions = []
    batch = []
    for i in range(batch_size):
        # Offset each molecule slightly so they don't overlap (though it doesn't matter for NL correctness)
        positions.append(pos_single + np.array([i * 10.0, 0.0, 0.0]))
        batch.extend([i] * n_atoms_single)

    return np.concatenate(positions, axis=0), np.array(batch, dtype=np.int32)


def benchmark_batch():
    batch_size = 128
    positions, batch = generate_water_batch(batch_size)
    cutoff = 5.0

    print(
        f"Benchmarking batch of {batch_size} water molecules ({len(positions)} atoms total)..."
    )

    # 1. Sequential Python Loop
    start = time.perf_counter()
    n_repeat = 10
    for _ in range(n_repeat):
        for i in range(batch_size):
            mask = batch == i
            pos_i = positions[mask]
            # No cell passed -> auto-box
            neighborlist_rs.build_neighborlists(None, pos_i, cutoff)
    t_seq = (time.perf_counter() - start) / n_repeat * 1000
    print(f"Sequential Python Loop: {t_seq:.2f} ms")

    # 2. Batched Rust Implementation
    # Warmup
    neighborlist_rs.build_neighborlists_batch(positions, batch, cutoff=cutoff)

    start = time.perf_counter()
    for _ in range(n_repeat):
        neighborlist_rs.build_neighborlists_batch(positions, batch, cutoff=cutoff)
    t_batch = (time.perf_counter() - start) / n_repeat * 1000
    print(f"Batched Rust (Parallel): {t_batch:.2f} ms")

    speedup = t_seq / t_batch
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_batch()
