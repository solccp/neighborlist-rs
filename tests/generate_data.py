import numpy as np
from ase import Atoms
from ase.build import molecule
import os


def generate_random_molecule(n_atoms, box_size=500.0):
    """Generates a random cluster of atoms in a massive box (effectively non-PBC)."""
    rng = np.random.default_rng(42)
    spread = 2.0 * (n_atoms ** (1 / 3))
    # Place atoms in the center of the massive box
    positions = (rng.random((n_atoms, 3)) - 0.5) * spread + box_size / 2
    symbols = rng.choice(["C", "H", "O", "N"], size=n_atoms)
    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_size] * 3, pbc=True)
    return atoms


def generate_ethanol_system(target_n_atoms, density=1.0):
    """Generates a box of ethanol molecules matching target atom count and density."""
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


def main():
    output_dir = "tests/data"
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ("isolated_100", "isolated", 100),
        ("isolated_1000", "isolated", 1000),
        ("isolated_10000", "isolated", 10000),
        ("isolated_20000", "isolated", 20000),
        ("ethanol_1000", "ethanol", 1000),
        ("ethanol_10000", "ethanol", 10000),
        ("ethanol_20000", "ethanol", 20000),
        ("si_bulk", "si_bulk", 8),
        ("isolated_two_clusters", "isolated_two_clusters", 20),
    ]

    for name, dtype, val in configs:
        print(f"Generating {name}...")
        if dtype == "isolated":
            atoms = generate_random_molecule(val)
            # Remove cell info to force plain XYZ format
            atoms.pbc = False
            atoms.cell = None
            filepath = os.path.join(output_dir, f"{name}.xyz")
            from ase.io import write

            write(filepath, atoms, format="xyz")
        elif dtype == "isolated_two_clusters":
            # Two clusters of 10 atoms each, separated by 10.0 Angstroms
            atoms1 = generate_random_molecule(10)
            atoms2 = generate_random_molecule(10)
            # Shift atoms2 by 10.0 in x
            atoms2.translate([10.0, 0.0, 0.0])
            atoms = atoms1 + atoms2
            atoms.pbc = False
            atoms.cell = None
            filepath = os.path.join(output_dir, f"{name}.xyz")
            from ase.io import write

            write(filepath, atoms, format="xyz")
        elif dtype == "ethanol":
            atoms = generate_ethanol_system(val)
            filepath = os.path.join(output_dir, f"{name}.xyz")
            from ase.io import write

            write(filepath, atoms, format="extxyz")
        elif dtype == "si_bulk":
            from ase.build import bulk

            atoms = bulk("Si", "diamond", a=5.43, cubic=True)
            filepath = os.path.join(output_dir, f"{name}.xyz")
            from ase.io import write

            write(filepath, atoms, format="extxyz")

        print(f"Saved to {filepath}")


if __name__ == "__main__":
    main()
