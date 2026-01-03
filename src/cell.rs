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

    pub fn wrap(&self, cart: &Vector3<f64>) -> Vector3<f64> {
        let frac = self.to_fractional(cart);
        let wrapped_frac = Vector3::new(
            frac.x - frac.x.floor(),
            frac.y - frac.y.floor(),
            frac.z - frac.z.floor(),
        );
        self.to_cartesian(&wrapped_frac)
    }

    pub fn get_shift_and_displacement(&self, r_i: &Vector3<f64>, r_j: &Vector3<f64>) -> (Vector3<i32>, Vector3<f64>) {
        let d_frac = self.to_fractional(&(r_j - r_i));
        let shift_frac = Vector3::new(
            -d_frac.x.round(),
            -d_frac.y.round(),
            -d_frac.z.round(),
        );
        let shift = Vector3::new(
            shift_frac.x as i32,
            shift_frac.y as i32,
            shift_frac.z as i32,
        );
        let r_j_img = r_j + self.h * shift_frac;
        let disp = r_j_img - r_i;
        (shift, disp)
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

    #[test]
    fn test_wrapping() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();

        let cart = Vector3::new(15.0, -2.0, 8.0);
        let wrapped = cell.wrap(&cart);

        assert_relative_eq!(wrapped.x, 5.0);
        assert_relative_eq!(wrapped.y, 8.0);
        assert_relative_eq!(wrapped.z, 8.0);
    }

    #[test]
    fn test_minimum_image() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();

        let r_i = Vector3::new(1.0, 1.0, 1.0);
        let r_j = Vector3::new(9.0, 9.0, 9.0);

        let (shift, disp) = cell.get_shift_and_displacement(&r_i, &r_j);

        // r_j_img = r_j + shift * H = [9, 9, 9] + [-1, -1, -1] * 10 = [-1, -1, -1]
        // disp = r_j_img - r_i = [-1, -1, -1] - [1, 1, 1] = [-2, -2, -2]
        assert_eq!(shift.x, -1);
        assert_eq!(shift.y, -1);
        assert_eq!(shift.z, -1);

        assert_relative_eq!(disp.x, -2.0);
        assert_relative_eq!(disp.y, -2.0);
        assert_relative_eq!(disp.z, -2.0);
    }
}