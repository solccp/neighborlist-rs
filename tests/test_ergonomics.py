import pytest
import neighborlist_rs


def test_pycell_repr():
    h = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
    cell = neighborlist_rs.PyCell(h)

    repr_str = repr(cell)
    print(f"Repr: {repr_str}")


def test_pycell_from_ase_mixed_pbc():
    try:
        from ase import Atoms
    except ImportError:
        pytest.skip("ASE not installed")

    atoms = Atoms(
        "H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=[True, True, False]
    )

    # Currently this raises ValueError in extract_ase_data,
    # but we want it to work (or at least PyCell to support it).
    # If we implement Mixed PBC, PyCell needs to know which dims are periodic.
    cell = neighborlist_rs.PyCell.from_ase(atoms)

    # Check that it didn't crash.
    assert "PyCell" in repr(cell)
