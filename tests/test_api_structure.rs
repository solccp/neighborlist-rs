use neighborlist_rs::NeighborList;

#[test]
fn test_neighborlist_struct_exists() {
    // This test is just to verify that the struct exists and can be instantiated (conceptually)
    // We can't really instantiate it easily without the constructor or pub fields,
    // but just referencing it verifies its visibility.
    let _list: Option<NeighborList> = None;
}
