use neighborlist_rs::build_neighborlists;

#[test]
fn test_build_neighborlists_simple() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let cutoff = 1.5;
    let parallel = false;

    let result = build_neighborlists(&positions, cutoff, None, parallel).unwrap();

    // Check edge_index: should have 1 edge (0->1) since i < j is enforced
    assert_eq!(result.edge_index.len(), 2);
    assert_eq!(result.edge_index[0], 0);
    assert_eq!(result.edge_index[1], 1);

    // Check shifts: should be all zeros for non-PBC
    assert_eq!(result.shifts.len(), 3);
    for &s in &result.shifts {
        assert_eq!(s, 0);
    }
}

#[test]
fn test_build_neighborlists_pbc() {
    let positions = [[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]];
    let cutoff = 2.5;
    let h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    let pbc = [true, true, true];
    let parallel = false;

    let result = build_neighborlists(&positions, cutoff, Some((&h, pbc)), parallel).unwrap();

    // Neighbors: (0, 1) via PBC. Dist is 2.0.
    // r_j_image = r_j + shift * cell.
    // 9.0 + (-1 * 10.0) = -1.0. Distance to 1.0 is 2.0.
    assert_eq!(result.edge_index.len(), 2);
    assert_eq!(result.edge_index[0], 0);
    assert_eq!(result.edge_index[1], 1);

    assert_eq!(result.shifts.len(), 3);
    assert_eq!(result.shifts[0], -1); // x shift
    assert_eq!(result.shifts[1], 0);
    assert_eq!(result.shifts[2], 0);
}

/// Displacement `x_j - x_i + S · h` with `h` holding the lattice vectors as rows.
fn displacement(pos: &[[f64; 3]], h: &[[f64; 3]; 3], i: usize, j: usize, s: &[i32]) -> f64 {
    let d: Vec<f64> = (0..3)
        .map(|k| {
            pos[j][k] - pos[i][k]
                + s[0] as f64 * h[0][k]
                + s[1] as f64 * h[1][k]
                + s[2] as f64 * h[2][k]
        })
        .collect();
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[test]
fn test_build_neighborlists_oblique_cell_rows() {
    // Triangular lattice, a = 3: rows are the lattice vectors. Read as columns
    // the lattice would be different (|a| = 3.35, |b| = 2.6) and the six
    // nearest images at 3.0 would not be found.
    let a = 3.0;
    let h = [
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * 3f64.sqrt() * a, 0.0],
        [0.0, 0.0, 10.0],
    ];
    let positions = [[0.2, 0.1, 5.0]];
    let cutoff = 3.2;

    for parallel in [false, true] {
        let result =
            build_neighborlists(&positions, cutoff, Some((&h, [true, true, true])), parallel)
                .unwrap();
        let n_edges = result.edge_index.len() / 2;
        assert_eq!(n_edges, 3, "six images at 3.0 -> three half-list pairs");
        for (e, s) in result.shifts.chunks(3).enumerate() {
            let (i, j) = (
                result.edge_index[e] as usize,
                result.edge_index[n_edges + e] as usize,
            );
            assert!((displacement(&positions, &h, i, j, s) - a).abs() < 1e-12);
        }
    }
}
