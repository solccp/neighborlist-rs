import neighborlist_rs
import pytest
from ase.build import bulk

try:
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_disjoint_logic():
    atoms = bulk("Cu", "fcc", a=3.6) * (3, 3, 3)
    pos = atoms.get_positions()
    cell = neighborlist_rs.PyCell(atoms.get_cell().tolist())

    # Nearest neighbor ~2.55, Second ~3.6, Third ~4.4
    cutoffs = [3.0, 4.0, 5.0]

    # 1. Standard (Cumulative)
    res_cum = neighborlist_rs.build_neighborlists_multi(
        cell, pos, cutoffs, disjoint=False
    )
    n0 = res_cum[0]["edge_index"].shape[1]
    n1 = res_cum[1]["edge_index"].shape[1]
    n2 = res_cum[2]["edge_index"].shape[1]

    assert n0 < n1 < n2

    # 2. Disjoint (Shells)
    res_dis = neighborlist_rs.build_neighborlists_multi(
        cell, pos, cutoffs, disjoint=True
    )
    d0 = res_dis[0]["edge_index"].shape[1]
    d1 = res_dis[1]["edge_index"].shape[1]
    d2 = res_dis[2]["edge_index"].shape[1]

    # d0 should be same as n0 (smallest shell is same as cumulative)
    assert d0 == n0
    # d1 should be (n1 - n0)
    assert d1 == n1 - n0
    # d2 should be (n2 - n1)
    assert d2 == n2 - n1
    # Total sum should be n2
    assert d0 + d1 + d2 == n2

    # 3. Check that edge indices are unique across shells in disjoint mode
    def get_pairs(ei):
        return set(tuple(sorted(p)) for p in ei.T)

    p0 = get_pairs(res_dis[0]["edge_index"])
    p1 = get_pairs(res_dis[1]["edge_index"])
    p2 = get_pairs(res_dis[2]["edge_index"])

    assert p0.isdisjoint(p1)
    assert p1.isdisjoint(p2)
    assert p0.isdisjoint(p2)


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_unsorted_cutoffs():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    pos = atoms.get_positions()
    cell = neighborlist_rs.PyCell(atoms.get_cell().tolist())

    cutoffs = [5.0, 3.0]  # Unsorted
    res = neighborlist_rs.build_neighborlists_multi(cell, pos, cutoffs, disjoint=True)

    # index 1 (3.0) is the smaller one
    # index 0 (5.0) is the larger one

    n_small = res[1]["edge_index"].shape[1]
    n_large = res[0]["edge_index"].shape[1]

    # Single call references
    ref_small = neighborlist_rs.build_neighborlists(cell, pos, 3.0)["edge_index"].shape[
        1
    ]
    ref_total = neighborlist_rs.build_neighborlists(cell, pos, 5.0)["edge_index"].shape[
        1
    ]

    assert n_small == ref_small
    assert n_large == ref_total - ref_small
