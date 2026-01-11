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
