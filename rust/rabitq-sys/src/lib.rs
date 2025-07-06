#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fht_kac_rotator_binding() {
        let dim: usize = 128;
        let padded_dim: usize = (dim + 15) / 16 * 16;

        let mut original_vec: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        original_vec.resize(padded_dim, 0.0);

        let mut rotated_vec: Vec<f32> = vec![0.0; padded_dim];

        let rotator = unsafe {
            rabitq_rotator_new(dim, padded_dim)
        };

        assert!(!rotator.is_null(), "Rotator creation failed");

        unsafe {
            rabitq_rotator_rotate(rotator, original_vec.as_ptr(), rotated_vec.as_mut_ptr());
        }

        println!("Original vector: {:?}", &original_vec[..dim]);
        println!("Rotated vector:  {:?}", &rotated_vec[..dim]);

        assert_ne!(
            &original_vec[..dim],
            &rotated_vec[..dim],
            "Rotated vector should not be the same as the original vector"
        );
        
        let is_any_nonzero = rotated_vec.iter().any(|&x| x != 0.0);
        assert!(is_any_nonzero, "Rotated vector should not be all zeros");

        unsafe {
            rabitq_rotator_free(rotator);
        }
    }
}
