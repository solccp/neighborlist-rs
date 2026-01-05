import neighborlist_rs
import numpy as np
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


def profile():
    neighborlist_rs.init_logging("info")
    n_atoms = 10000
    atoms = generate_ethanol_system(n_atoms)
    pos = atoms.get_positions()
    h_T = atoms.get_cell()[:].T.tolist()
    cutoff = 6.0

    print(f"Profiling {n_atoms} atoms at cutoff {cutoff}...")
    cell = neighborlist_rs.PyCell(h_T)
    # This should trigger tracing output to stdout
    neighborlist_rs.build_neighborlists(cell, pos, cutoff, parallel=True)


if __name__ == "__main__":
    profile()
