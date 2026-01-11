use neighborlist_rs::build_neighborlists_batch;

#[test]
fn test_build_neighborlists_batch_mixed() {
    // System 0: Isolated, [0,0,0], [1,0,0]
    // System 1: PBC, [1,1,1], [9,1,1] in 10x10x10 box
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [9.0, 1.0, 1.0],
    ];
    let batch = [0, 0, 1, 1];
    let cutoff = 2.5;

    let h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    let pbc = [true, true, true];

    // Mixed cells: first None, second Some
    let cells_data = [None, Some((h, pbc))];

    let result =
        build_neighborlists_batch(&positions, &batch, cutoff, Some(&cells_data), true).unwrap();

    let n_edges = result.edge_index.len() / 2;
    assert_eq!(n_edges, 2);

    let src = &result.edge_index[0..n_edges];
    let dst = &result.edge_index[n_edges..];

    // Check if (0,1) exists
    let idx0 = src
        .iter()
        .position(|&x| x == 0)
        .expect("Edge 0->1 not found");
    assert_eq!(dst[idx0], 1);
    assert_eq!(result.shifts[3 * idx0], 0);
    assert_eq!(result.shifts[3 * idx0 + 1], 0);
    assert_eq!(result.shifts[3 * idx0 + 2], 0);

    // Check if (2,3) exists
    let idx1 = src
        .iter()
        .position(|&x| x == 2)
        .expect("Edge 2->3 not found");
    assert_eq!(dst[idx1], 3);
    assert_eq!(result.shifts[3 * idx1], -1); // x shift
    assert_eq!(result.shifts[3 * idx1 + 1], 0);
    assert_eq!(result.shifts[3 * idx1 + 2], 0);
}
