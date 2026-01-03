use pyo3::prelude::*;

pub mod cell;
pub mod search;

#[pymodule]
fn neighborlist_rs(_py: Python, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
