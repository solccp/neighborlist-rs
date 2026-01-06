import time
import numpy as np
import os
import torch
import warnings
from ase.io import read
from ase.neighborlist import neighbor_list
import matscipy.neighbours
import freud
import vesin
import neighborlist_rs

try:
    import torch_cluster
except ImportError:
    torch_cluster = None
try:
    import torch_nl
except ImportError:
    torch_nl = None

# --- Configuration ---
warnings.filterwarnings("ignore")  # Suppress warnings
N_REPEAT = 5
N_WARMUP = 2
CUTOFFS = [6.0, 14.0]
MULTI_CUTOFFS = [6.0, 14.0]  # Short and Long for multi-cutoff benchmark
BATCH_SIZES = [1, 8, 32, 128]


def get_device_name(device):
    if device == "cpu":
        return "CPU"
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"


def benchmark_func(func, name, n_repeat=N_REPEAT, n_warmup=N_WARMUP):
    times = []
    try:
        # Warmup
        for _ in range(n_warmup):
            func()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Bench
        for _ in range(n_repeat):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            func()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        return np.mean(times) * 1000.0, np.std(times) * 1000.0
    except Exception:
        # print(f"FAILED {name}: {e}")
        return None, None


# --- Wrappers (Single Cutoff) ---


def run_ase(atoms, cutoff):
    neighbor_list("ij", atoms, cutoff)


def run_matscipy(atoms, cutoff):
    matscipy.neighbours.neighbour_list("ij", atoms, cutoff)


def run_freud(atoms, cutoff):
    box = freud.box.Box.from_matrix(atoms.get_cell().array)
    points = atoms.get_positions()
    aq = freud.locality.AABBQuery(box, points)
    aq.query(points, dict(r_max=cutoff, exclude_ii=True)).toNeighborList()


def run_vesin(atoms, cutoff):
    nl = vesin.NeighborList(cutoff=cutoff, full_list=True)
    box = atoms.get_cell().array
    points = atoms.get_positions()
    nl.compute(points, box=box, periodic=True, quantities="ij")


def run_rs_single(atoms, cutoff, parallel=True):
    neighborlist_rs.build_from_ase(atoms, cutoff)


def run_rs_batch(pos_np, batch_np, cells_np, cutoff, parallel=True):
    neighborlist_rs.build_neighborlists_batch(
        pos_np, batch_np, cells=cells_np, cutoff=cutoff, parallel=parallel
    )


def run_torch_cluster(pos_t, batch_t, cutoff, device):
    torch_cluster.radius_graph(pos_t, r=cutoff, batch=batch_t, max_num_neighbors=100000)


def run_torch_nl(pos_t, cell_t, batch_t, cutoff, device):
    # Expand PBC to (B, 3)
    # Assuming batch_ids are monotonic/sorted
    n_batch = cell_t.shape[0]
    pbc = torch.tensor([True, True, True], device=device)
    pbc = pbc.unsqueeze(0).expand(n_batch, -1)

    torch_nl.compute_neighborlist(cutoff, pos_t, cell_t, pbc, batch=batch_t)


# --- Wrappers (Multi Cutoff) ---


def run_rs_multi(atoms, cutoffs):
    neighborlist_rs.build_multi_from_ase(atoms, cutoffs)


def run_ase_multi_filtered(atoms, cutoffs):
    max_cut = max(cutoffs)
    min_cut = min(cutoffs)
    i, j, d = neighbor_list("ijd", atoms, max_cut)
    # Filter
    mask = d <= min_cut
    _ = i[mask]


def run_matscipy_multi_filtered(atoms, cutoffs):
    max_cut = max(cutoffs)
    min_cut = min(cutoffs)
    i, j, d = matscipy.neighbours.neighbour_list("ijd", atoms, max_cut)
    mask = d <= min_cut
    _ = i[mask]


def run_freud_multi_filtered(atoms, cutoffs):
    max_cut = max(cutoffs)
    min_cut = min(cutoffs)
    box = freud.box.Box.from_matrix(atoms.get_cell().array)
    points = atoms.get_positions()
    aq = freud.locality.AABBQuery(box, points)
    nl = aq.query(points, dict(r_max=max_cut, exclude_ii=True)).toNeighborList()
    # Filter
    d = nl.distances
    _ = d <= min_cut
    # In Freud, nl.distances is property, returns copy or view?
    # Accessing it computes it if not stored? Freud stores it.
    # Filtering:
    # idx = np.where(d <= min_cut)[0]
    # small_nl = nl[idx] # Freud NL might not support slicing like this efficiently or returns new object?
    # Just accessing distances and creating mask is enough work to simulate.


def run_vesin_multi_filtered(atoms, cutoffs):
    max_cut = max(cutoffs)
    min_cut = min(cutoffs)
    nl = vesin.NeighborList(cutoff=max_cut, full_list=True)
    box = atoms.get_cell().array
    points = atoms.get_positions()
    i, j, d = nl.compute(points, box=box, periodic=True, quantities="ijd")
    mask = d <= min_cut
    _ = i[mask]


def run_torch_cluster_multi_filtered(pos_t, batch_t, cutoffs, device):
    max_cut = max(cutoffs)
    min_cut = min(cutoffs)
    edge_index = torch_cluster.radius_graph(
        pos_t, r=max_cut, batch=batch_t, max_num_neighbors=100000
    )
    # Compute distances to filter
    row, col = edge_index
    dist = (pos_t[row] - pos_t[col]).norm(dim=-1)
    mask = dist <= min_cut
    _ = edge_index[:, mask]


def run_torch_nl_multi_filtered(pos_t, cell_t, batch_t, cutoffs, device):
    # TorchNL returns mapping, filtering is complex on mapping object or raw tensors.
    # We will compute for max_cut. TorchNL doesn't return distances by default?
    # compute_neighborlist returns mapping.
    # We'd need to compute distances from mapping.
    # This is likely too slow/complex to simulate easily without being unfair.
    # We will skip TorchNL multi-cutoff simulation for now or just run max_cut.
    pass


# --- Runners ---


def benchmark_single_systems(data_dir):
    print(f"\n{'=' * 40} Single System Benchmarks {'=' * 40}")

    files = [
        ("Isolated 1k", "isolated_1000.xyz"),
        ("Ethanol 1k (PBC)", "ethanol_1000.xyz"),
        ("Ethanol 10k (PBC)", "ethanol_10000.xyz"),
    ]

    for cutoff in CUTOFFS:
        print(f"\n\n>>> CUTOFF: {cutoff} Angstroms <<<")
        for name, f in files:
            path = os.path.join(data_dir, f)
            atoms = read(path)
            n_atoms = len(atoms)

            # Check for isolated
            is_isolated = np.all(atoms.get_cell() == 0)
            atoms_boxed = atoms.copy()
            if is_isolated:
                # Create a large bounding box
                pos = atoms.get_positions()
                min_p = np.min(pos, axis=0)
                max_p = np.max(pos, axis=0)
                L = (max_p - min_p) + 2 * (cutoff + 1.0)
                # Center
                center = (max_p + min_p) / 2
                atoms_boxed.set_cell(np.diag(L))
                atoms_boxed.set_pbc(True)  # Use PBC for libs that need it
                atoms_boxed.positions = pos - center + L / 2  # Shift to center of box

            print(f"\n--- {name} (N={n_atoms}, rc={cutoff}) ---")
            print(f"{'Library':<20} | {'Device':<5} | {'Time (ms)':<10} | {'Std':<10}")
            print("-" * 55)

            # Single Cutoff Runs
            t, std = benchmark_func(lambda: run_ase(atoms_boxed, cutoff), "ASE")
            if t:
                print(f"{' ASE':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

            t, std = benchmark_func(
                lambda: run_matscipy(atoms_boxed, cutoff), "Matscipy"
            )
            if t:
                print(f"{' Matscipy':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

            t, std = benchmark_func(lambda: run_freud(atoms_boxed, cutoff), "Freud")
            if t:
                print(f"{' Freud':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

            t, std = benchmark_func(lambda: run_vesin(atoms_boxed, cutoff), "Vesin")
            if t:
                print(f"{' Vesin':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

            neighborlist_rs.set_num_threads(1)
            t, std = benchmark_func(lambda: run_rs_single(atoms, cutoff), "RS (1)")
            if t:
                print(f"{' RS (1 thread)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

            neighborlist_rs.set_num_threads(8)
            t, std = benchmark_func(lambda: run_rs_single(atoms, cutoff), "RS (Par)")
            if t:
                print(f"{' RS (Parallel)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

            # Torch Cluster
            if torch_cluster:
                if is_isolated:
                    pos_t = torch.tensor(atoms.get_positions(), dtype=torch.float32)
                    batch_t = torch.zeros(len(atoms), dtype=torch.long)
                    t, std = benchmark_func(
                        lambda: run_torch_cluster(pos_t, batch_t, cutoff, "cpu"),
                        "TC CPU",
                    )
                    if t:
                        print(
                            f"{' TorchCluster':<20} | CPU   | {t:<10.2f} | {std:<10.2f}"
                        )

                    if torch.cuda.is_available():
                        device = "cuda"
                        pos_gpu = pos_t.to(device)
                        batch_gpu = batch_t.to(device)
                        t, std = benchmark_func(
                            lambda: run_torch_cluster(
                                pos_gpu, batch_gpu, cutoff, device
                            ),
                            "TC GPU",
                        )
                        if t:
                            print(
                                f"{' TorchCluster':<20} | GPU   | {t:<10.2f} | {std:<10.2f}"
                            )
                else:
                    print(
                        f"{' TorchCluster':<20} | ---   | {'N/A (No PBC support)':<10} | {'':<10}"
                    )

            if torch_nl:
                # Must use boxed version for cell
                pos_t = torch.tensor(atoms_boxed.get_positions(), dtype=torch.float32)
                batch_t = torch.zeros(len(atoms), dtype=torch.long)
                cell_t = torch.tensor(
                    atoms_boxed.get_cell().array, dtype=torch.float32
                ).unsqueeze(0)
                t, std = benchmark_func(
                    lambda: run_torch_nl(pos_t, cell_t, batch_t, cutoff, "cpu"),
                    "TNL CPU",
                )
                if t:
                    print(f"{' TorchNL':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

                if torch.cuda.is_available():
                    device = "cuda"
                    pos_gpu = pos_t.to(device)
                    batch_gpu = batch_t.to(device)
                    cell_gpu = cell_t.to(device)
                    t, std = benchmark_func(
                        lambda: run_torch_nl(
                            pos_gpu, cell_gpu, batch_gpu, cutoff, device
                        ),
                        "TNL GPU",
                    )
                    if t:
                        print(f"{' TorchNL':<20} | GPU   | {t:<10.2f} | {std:<10.2f}")


def benchmark_multi_cutoff(data_dir):
    print(f"\n{'=' * 40} Multi-Cutoff Benchmarks {MULTI_CUTOFFS} {'=' * 40}")
    # Using Ethanol 10k as representative dense system
    f = "ethanol_10000.xyz"
    path = os.path.join(data_dir, f)
    atoms = read(path)
    atoms_boxed = atoms.copy()  # PBC system, box exists
    n_atoms = len(atoms)

    print(f"\n--- {f} (N={n_atoms}) ---")
    print(f"{'Library':<20} | {'Device':<5} | {'Time (ms)':<10} | {'Std':<10}")
    print("-" * 55)

    # RS Multi
    neighborlist_rs.set_num_threads(8)
    t, std = benchmark_func(lambda: run_rs_multi(atoms, MULTI_CUTOFFS), "RS Multi")
    if t:
        print(f"{' RS (Multi)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

    # ASE Filtered
    t, std = benchmark_func(
        lambda: run_ase_multi_filtered(atoms_boxed, MULTI_CUTOFFS), "ASE Filtered"
    )
    if t:
        print(f"{' ASE (Filtered)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

    # Matscipy Filtered
    t, std = benchmark_func(
        lambda: run_matscipy_multi_filtered(atoms_boxed, MULTI_CUTOFFS),
        "Matscipy Filtered",
    )
    if t:
        print(f"{' Matscipy (Filt)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

    # Freud Filtered
    t, std = benchmark_func(
        lambda: run_freud_multi_filtered(atoms_boxed, MULTI_CUTOFFS), "Freud Filtered"
    )
    if t:
        print(f"{' Freud (Filt)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

    # Vesin Filtered
    t, std = benchmark_func(
        lambda: run_vesin_multi_filtered(atoms_boxed, MULTI_CUTOFFS), "Vesin Filtered"
    )
    if t:
        print(f"{' Vesin (Filt)':<20} | CPU   | {t:<10.2f} | {std:<10.2f}")

    # Torch Cluster Filtered
    if torch_cluster:
        # ethanol_10000 has PBC, TC doesn't support it.
        print(f"{' TC (Filt)':<20} | ---   | {'N/A (No PBC support)':<10} | {'':<10}")


def benchmark_batching(data_dir):
    print(f"\n{'=' * 40} Batch Scaling Benchmarks {'=' * 40}")

    base_atoms = read(os.path.join(data_dir, "isolated_100.xyz"))
    base_pos = base_atoms.get_positions()
    cutoff = 6.0

    # Create a bounding box for isolated system
    min_p = np.min(base_pos, axis=0)
    max_p = np.max(base_pos, axis=0)
    L = (max_p - min_p) + 2 * (cutoff + 1.0)
    base_cell = np.diag(L)
    # Center positions in box
    center = (max_p + min_p) / 2
    base_pos = base_pos - center + L / 2

    print(f"Base System: {len(base_atoms)} atoms (Isolated 100). Cutoff: {cutoff}")
    print(
        f"{'Batch Size':<10} | {'Library':<20} | {'Device':<5} | {'Time (ms)':<10} | {'Throughput':<15}"
    )
    print("-" * 80)

    for b_size in BATCH_SIZES:
        pos_list = [base_pos for _ in range(b_size)]
        pos_np = np.vstack(pos_list)
        batch_ids = np.repeat(np.arange(b_size), len(base_atoms)).astype(np.int32)
        cells_np = np.array([base_cell for _ in range(b_size)])

        pos_t = torch.tensor(pos_np, dtype=torch.float32)
        batch_t = torch.tensor(batch_ids, dtype=torch.long)
        cells_t = torch.tensor(cells_np, dtype=torch.float32)

        neighborlist_rs.set_num_threads(8)
        t, std = benchmark_func(
            lambda: run_rs_batch(pos_np, batch_ids, cells_np, cutoff), "RS Batch"
        )
        if t:
            thru = b_size / (t / 1000.0)
            print(f"{b_size:<10} | {'RS':<20} | CPU   | {t:<10.2f} | {thru:<15.2f}")

        if torch_cluster:
            t, std = benchmark_func(
                lambda: run_torch_cluster(pos_t, batch_t, cutoff, "cpu"), "TC CPU"
            )
            if t:
                thru = b_size / (t / 1000.0)
                print(
                    f"{b_size:<10} | {'TorchCluster':<20} | CPU   | {t:<10.2f} | {thru:<15.2f}"
                )

            if torch.cuda.is_available():
                device = "cuda"
                pos_gpu = pos_t.to(device)
                batch_gpu = batch_t.to(device)
                t, std = benchmark_func(
                    lambda: run_torch_cluster(pos_gpu, batch_gpu, cutoff, device),
                    "TC GPU",
                )
                if t:
                    thru = b_size / (t / 1000.0)
                    print(
                        f"{b_size:<10} | {'TorchCluster':<20} | GPU   | {t:<10.2f} | {thru:<15.2f}"
                    )

        if torch_nl:
            t, std = benchmark_func(
                lambda: run_torch_nl(pos_t, cells_t, batch_t, cutoff, "cpu"), "TNL CPU"
            )
            if t:
                thru = b_size / (t / 1000.0)
                print(
                    f"{b_size:<10} | {'TorchNL':<20} | CPU   | {t:<10.2f} | {thru:<15.2f}"
                )

            if torch.cuda.is_available():
                device = "cuda"
                pos_gpu = pos_t.to(device)
                batch_gpu = batch_t.to(device)
                cells_gpu = cells_t.to(device)
                t, std = benchmark_func(
                    lambda: run_torch_nl(pos_gpu, cells_gpu, batch_gpu, cutoff, device),
                    "TNL GPU",
                )
                if t:
                    thru = b_size / (t / 1000.0)
                    print(
                        f"{b_size:<10} | {'TorchNL':<20} | GPU   | {t:<10.2f} | {thru:<15.2f}"
                    )


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "../tests/data")
    benchmark_single_systems(data_dir)
    benchmark_multi_cutoff(data_dir)
    benchmark_batching(data_dir)
