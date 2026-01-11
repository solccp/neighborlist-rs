use crate::batch;
use crate::single;
use nalgebra::{Matrix3, Vector3};

/// A container for neighbor list results.
///
/// This struct holds the edge indices and shift vectors for a neighbor list.
/// It is designed to be compatible with PyTorch Geometric (PyG) conventions.
#[derive(Debug, Clone)]
pub struct NeighborList {
    /// Edge indices: [source_0, source_1, ..., target_0, target_1, ...]
    /// Length is 2 * num_edges.
    pub edge_index: Vec<i64>,
    /// Shift vectors: [x_0, y_0, z_0, x_1, y_1, z_1, ...]
    /// Length is 3 * num_edges.
    pub shifts: Vec<i32>,
}

/// Build neighbor lists for a single system.
pub fn build_neighborlists(
    positions: &[[f64; 3]],
    cutoff: f64,
    cell: Option<(&[[f64; 3]; 3], [bool; 3])>,
    parallel: bool,
) -> Result<NeighborList, String> {
    let pos_vec = convert_positions(positions);
    let cell_info = cell.map(|(h, pbc)| (convert_cell(h), Vector3::new(pbc[0], pbc[1], pbc[2])));

    let (mut edge_i, edge_j, shifts) =
        single::search_single(&pos_vec, cell_info, cutoff, parallel)?;

    edge_i.extend(edge_j);

    Ok(NeighborList {
        edge_index: edge_i,
        shifts,
    })
}

/// Build neighbor lists for a batch of systems.
type CellInfo = ([[f64; 3]; 3], [bool; 3]);

pub fn build_neighborlists_batch(
    positions: &[[f64; 3]],
    batch: &[i32],
    cutoff: f64,
    cells: Option<&[Option<CellInfo>]>,
    parallel: bool,
) -> Result<NeighborList, String> {
    let pos_vec = convert_positions(positions);

    let n_systems = if batch.is_empty() {
        0
    } else {
        let mut count = 1;
        let mut curr = batch[0];
        for &b in batch.iter().skip(1) {
            if b != curr {
                count += 1;
                curr = b;
            }
        }
        count
    };

    let cells_vec: Vec<Option<(Matrix3<f64>, Vector3<bool>)>> = if let Some(c_slice) = cells {
        c_slice
            .iter()
            .map(|opt| opt.map(|(h, pbc)| (convert_cell(&h), Vector3::new(pbc[0], pbc[1], pbc[2]))))
            .collect()
    } else {
        vec![None; n_systems]
    };

    let (mut edge_i, edge_j, shifts) =
        batch::search_batch(&pos_vec, batch, &cells_vec, cutoff, parallel)?;

    edge_i.extend(edge_j);

    Ok(NeighborList {
        edge_index: edge_i,
        shifts,
    })
}

fn convert_positions(positions: &[[f64; 3]]) -> Vec<Vector3<f64>> {
    positions
        .iter()
        .map(|p| Vector3::new(p[0], p[1], p[2]))
        .collect()
}

fn convert_cell(cell: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
        cell[2][1], cell[2][2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_convert_positions() {
        let input = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let output = convert_positions(&input);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(output[1], Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_convert_cell() {
        let input = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let output = convert_cell(&input);
        assert_eq!(output[(0, 0)], 1.0);
        assert_eq!(output[(2, 2)], 1.0);
    }

    #[test]
    fn test_api_batch_empty() {
        let res = build_neighborlists_batch(&[], &[], 1.0, None, false).unwrap();
        assert!(res.edge_index.is_empty());
    }

    #[test]
    fn test_api_batch_no_cells() {
        // 2 systems, no cells provided (should use None -> auto-box)
        let positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let batch = [0, 1];
        let res = build_neighborlists_batch(&positions, &batch, 1.0, None, false).unwrap();
        assert!(res.edge_index.is_empty());
    }
}
