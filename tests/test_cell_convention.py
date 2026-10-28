"""Regression tests for the cell convention on an oblique cell.

Every API takes the lattice vectors as rows (ASE's ``atoms.cell``) and returns
shifts with ``r_ij = x_j - x_i + S @ cell``. Orthogonal cells cannot tell rows
from columns, so these tests use an fcc(111) slab with a 60 degree in-plane
angle.
"""

import re

import numpy as np
import pytest

import neighborlist_rs

ase = pytest.importorskip("ase")
from ase.build import fcc111  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

CUTOFF = 31.7506  # 60 Bohr: exceeds the in-plane cell and the vacuum gap.


@pytest.fixture(scope="module")
def slab():
    atoms = fcc111("Al", size=(8, 8, 6), vacuum=10.0)
    atoms.pbc = (True, True, False)
    atoms.rattle(0.03, seed=42)
    return atoms


@pytest.fixture(scope="module")
def ase_full(slab):
    i, j, s = neighbor_list("ijS", slab, CUTOFF)
    return _sorted_rows(np.column_stack([i, j, s]))


def _sorted_rows(a):
    a = np.asarray(a, dtype=np.int64)
    return a[np.lexsort(a.T[::-1])]


def _full_from_half(res):
    ei, s = res["edge_index"], res["shift"]
    fwd = np.column_stack([ei[0], ei[1], s])
    bwd = np.column_stack([ei[1], ei[0], -s])
    return _sorted_rows(np.concatenate([fwd, bwd]))


def _check(res, slab, ase_full):
    full = _full_from_half(res)
    np.testing.assert_array_equal(full, ase_full)
    # The shift meaning, independently of ASE's pair set.
    ei, s = res["edge_index"], res["shift"]
    pos = slab.get_positions()
    d = pos[ei[1]] - pos[ei[0]] + s @ slab.cell.array
    assert np.linalg.norm(d, axis=1).max() < CUTOFF


def test_cell_is_oblique(slab):
    cell = slab.cell.array
    assert not np.allclose(cell, cell.T)


def test_build_from_ase(slab, ase_full):
    _check(neighborlist_rs.build_from_ase(slab, CUTOFF), slab, ase_full)


@pytest.mark.parametrize("parallel", [True, False])
def test_pycell_takes_rows(slab, ase_full, parallel):
    cell = neighborlist_rs.PyCell(slab.cell.tolist(), pbc=list(slab.pbc))
    res = neighborlist_rs.build_neighborlists(
        cell, slab.get_positions(), CUTOFF, parallel=parallel
    )
    _check(res, slab, ase_full)


def test_pycell_repr_round_trips_rows(slab):
    cell = neighborlist_rs.PyCell(slab.cell.tolist())
    h_text = repr(cell).split("pbc=")[0]
    h = np.array(re.findall(r"-?\d+(?:\.\d*)?(?:e-?\d+)?", h_text), float)
    np.testing.assert_allclose(h.reshape(3, 3), slab.cell.array)


def test_batch_takes_rows():
    # Batch cells carry no pbc flag, so use a fully periodic oblique bulk.
    atoms = fcc111("Al", size=(4, 4, 3), vacuum=0.0, periodic=True)
    atoms.rattle(0.03, seed=7)
    cutoff = 12.0
    i, j, s = neighbor_list("ijS", atoms, cutoff)
    expected = _sorted_rows(np.column_stack([i, j, s]))
    res = neighborlist_rs.build_neighborlists_batch(
        atoms.get_positions(),
        np.zeros(len(atoms), dtype=np.int32),
        cells=atoms.cell.array[None],
        cutoff=cutoff,
    )
    np.testing.assert_array_equal(_full_from_half(res), expected)
