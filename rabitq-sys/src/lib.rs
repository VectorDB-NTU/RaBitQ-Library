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
        // 根据 rabitqlib 的要求，padded_dim 通常需要是某个值的倍数，例如 16 或 32
        let padded_dim: usize = (dim + 15) / 16 * 16;

        // 创建原始向量
        let mut original_vec: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        original_vec.resize(padded_dim, 0.0);

        // 创建一个用于存放旋转后向量的缓冲区
        let mut rotated_vec: Vec<f32> = vec![0.0; padded_dim];

        // 调用 C API 创建旋转器
        let rotator = unsafe {
            rabitq_rotator_new(dim, padded_dim)
        };

        // 确认旋转器指针不是空的
        assert!(!rotator.is_null(), "Rotator creation failed");

        // 调用 C API 执行旋转
        unsafe {
            rabitq_rotator_rotate(rotator, original_vec.as_ptr(), rotated_vec.as_mut_ptr());
        }

        println!("Original vector: {:?}", &original_vec[..dim]);
        println!("Rotated vector:  {:?}", &rotated_vec[..dim]);

        // 检查旋转后的向量是否与原始向量不同
        // 这是一个基本的健全性检查。FHT 是一种正交变换，它会改变向量的值。
        assert_ne!(
            &original_vec[..dim],
            &rotated_vec[..dim],
            "Rotated vector should not be the same as the original vector"
        );
        
        // 检查旋转后的向量是否包含非零值
        let is_any_nonzero = rotated_vec.iter().any(|&x| x != 0.0);
        assert!(is_any_nonzero, "Rotated vector should not be all zeros");

        // 释放旋转器内存
        unsafe {
            rabitq_rotator_free(rotator);
        }
    }
}
