"""Regression tests for self-image pairs when the cutoff exceeds a cell length.

A half list must contain each self-image pair (i, i, S) exactly once, not both
(i, i, S) and (i, i, -S), so that symmetrising it reproduces ASE's full list.
"""

from collections import Counter

import numpy as np
import pytest

import neighborlist_rs

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402


def _full_from_half(edge_index, shift):
    """Symmetrises a half list into a multiset of (i, j, sx, sy, sz)."""
    full = Counter()
    for i, j, s in zip(edge_index[0], edge_index[1], shift):
        full[(int(i), int(j), *map(int, s))] += 1
        full[(int(j), int(i), *map(int, -s))] += 1
    return full


def _ase_full(atoms, cutoff):
    """Returns ASE's full neighbor list as a multiset of (i, j, sx, sy, sz)."""
    i, j, s = neighbor_list("ijS", atoms, cutoff)
    return Counter((int(a), int(b), *map(int, c)) for a, b, c in zip(i, j, s))


def _assert_matches_ase(res, atoms, cutoff):
    edge_index, shift = res["edge_index"], res["shift"]
    self_mask = edge_index[0] == edge_index[1]
    assert self_mask.any(), "test system must contain self-image pairs"
    # Canonical sign: first non-zero shift component of a self pair is positive.
    for s in shift[self_mask]:
        assert s[np.flatnonzero(s)[0]] > 0
    assert _full_from_half(edge_index, shift) == _ase_full(atoms, cutoff)


def _si_bulk(reps):
    return bulk("Si", cubic=True).repeat(reps)


def _si_slab():
    atoms = _si_bulk((2, 2, 2))
    atoms.center(vacuum=5.0, axis=2)
    atoms.pbc = (True, True, False)
    return atoms


SYSTEMS = [
    pytest.param(_si_bulk((1, 1, 1)), 7.0, id="si8-7A"),
    pytest.param(_si_bulk((2, 2, 1)), 7.0, id="si32-7A"),
    pytest.param(_si_bulk((4, 4, 3)), 20.0, id="si384-20A"),
    pytest.param(_si_slab(), 12.0, id="slab-TTF-12A"),
    pytest.param(
        Atoms("H", positions=[[0.1, 0.2, 0.3]], cell=[2.0, 2.5, 3.0], pbc=True),
        6.5,
        id="single-atom",
    ),
]


@pytest.mark.parametrize("atoms,cutoff", SYSTEMS)
def test_single_matches_ase(atoms, cutoff):
    res = neighborlist_rs.build_from_ase(atoms, cutoff)
    _assert_matches_ase(res, atoms, cutoff)


@pytest.mark.parametrize("atoms,cutoff", SYSTEMS)
def test_single_serial_matches_ase(atoms, cutoff):
    cell = neighborlist_rs.PyCell.from_ase(atoms)
    res = neighborlist_rs.build_neighborlists(
        cell, atoms.get_positions(), cutoff, parallel=False
    )
    _assert_matches_ase(res, atoms, cutoff)


@pytest.mark.parametrize("disjoint", [False, True])
def test_multi_matches_ase(disjoint):
    atoms = _si_bulk((2, 2, 1))
    cutoffs = [3.0, 7.0, 9.0]
    res = neighborlist_rs.build_multi_from_ase(atoms, cutoffs, disjoint=disjoint)
    cumulative = Counter()
    for k, rc in enumerate(cutoffs):
        full = _full_from_half(res[k]["edge_index"], res[k]["shift"])
        if disjoint:
            # Disjoint shells sum to the cumulative list.
            cumulative += full
            full = cumulative
        assert full == _ase_full(atoms, rc), f"cutoff {rc}"


def _slab_outside_box():
    # Atoms straddle both z faces of a non-periodic axis.
    atoms = _si_slab()
    atoms.positions[:, 2] -= 8.0
    return atoms


def _peer_slab():
    from ase.build import diamond100

    atoms = diamond100("Si", size=(4, 4, 8), vacuum=12.0)
    atoms.pbc = (True, True, False)
    return atoms


NON_PERIODIC = [
    pytest.param(_si_slab(), 25.0, id="slab-cutoff-gt-vacuum"),
    pytest.param(_slab_outside_box(), 12.0, id="slab-atoms-outside-box"),
    pytest.param(_peer_slab(), 31.7506, id="si100-slab-60bohr"),
]


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("atoms,cutoff", NON_PERIODIC)
def test_no_images_along_non_periodic_axis(atoms, cutoff, parallel):
    cell = neighborlist_rs.PyCell.from_ase(atoms)
    res = neighborlist_rs.build_neighborlists(
        cell, atoms.get_positions(), cutoff, parallel=parallel
    )
    assert not res["shift"][:, 2].any()
    _assert_matches_ase(res, atoms, cutoff)


def test_batch_matches_ase():
    systems = [_si_bulk((1, 1, 1)), _si_bulk((2, 2, 1))]
    cutoff = 7.0
    positions = np.concatenate([a.get_positions() for a in systems])
    batch = np.concatenate(
        [np.full(len(a), k, dtype=np.int32) for k, a in enumerate(systems)]
    )
    cells = np.stack([np.asarray(a.cell) for a in systems])
    res = neighborlist_rs.build_neighborlists_batch(
        positions, batch, cells=cells, cutoff=cutoff
    )
    expected = Counter()
    offset = 0
    for a in systems:
        for (i, j, *s), n in _ase_full(a, cutoff).items():
            expected[(i + offset, j + offset, *s)] = n
        offset += len(a)
    assert _full_from_half(res["edge_index"], res["shift"]) == expected
