use nalgebra::{Matrix3, Vector3};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CellError {
    #[error("Cell matrix is not invertible")]
    NotInvertible,
}

pub struct Cell {
    h: Matrix3<f64>,
    h_inv: Matrix3<f64>,
}

impl Cell {
    pub fn new(h: Matrix3<f64>) -> Result<Self, CellError> {
        let h_inv = h.try_inverse().ok_or(CellError::NotInvertible)?;
        Ok(Self { h, h_inv })
    }

    pub fn to_fractional(&self, cart: &Vector3<f64>) -> Vector3<f64> {
        self.h_inv * cart
    }

    pub fn to_cartesian(&self, frac: &Vector3<f64>) -> Vector3<f64> {
        self.h * frac
    }

    pub fn h(&self) -> &Matrix3<f64> {
        &self.h
    }

    pub fn h_inv(&self) -> &Matrix3<f64> {
        &self.h_inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_coordinate_transformation() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();

        let cart = Vector3::new(5.0, 2.0, 8.0);
        let frac = cell.to_fractional(&cart);

        assert_relative_eq!(frac.x, 0.5);
        assert_relative_eq!(frac.y, 0.2);
        assert_relative_eq!(frac.z, 0.8);

        let cart_back = cell.to_cartesian(&frac);
        assert_relative_eq!(cart_back.x, cart.x);
        assert_relative_eq!(cart_back.y, cart.y);
        assert_relative_eq!(cart_back.z, cart.z);
    }

    #[test]
    fn test_triclinic_cell() {
        // A simple triclinic cell
        let h = Matrix3::new(
            10.0, 2.0, 1.0,
            0.0, 10.0, 0.5,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();

        let cart = Vector3::new(13.0, 10.5, 10.0);
        let frac = cell.to_fractional(&cart);
        
        // r_cart = H * r_frac => r_frac = [1, 1, 1]
        assert_relative_eq!(frac.x, 1.0);
        assert_relative_eq!(frac.y, 1.0);
        assert_relative_eq!(frac.z, 1.0);
    }

        #[test]

        fn test_invalid_cell() {

            let h = Matrix3::zeros();

            let cell = Cell::new(h);

            assert!(cell.is_err());

        }

    

        #[test]

        fn test_getters() {

            let h = Matrix3::identity();

            let cell = Cell::new(h).unwrap();

            assert_eq!(cell.h(), &h);

            assert_eq!(cell.h_inv(), &h);

        }

    }

    