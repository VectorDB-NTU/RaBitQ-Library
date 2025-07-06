use rabitq_sys as ffi;

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum MetricType {
    L2 = 0,
    IP = 1,
}

pub struct RabitqConfig {
    pub ptr: *mut ffi::RabitqConfig,
}

impl RabitqConfig {
    pub fn new() -> Self {
        let ptr = unsafe { ffi::rabitq_config_new() };
        Self { ptr }
    }

    pub fn faster_config(dim: usize, total_bits: usize) -> Self {
        let ptr = unsafe { ffi::rabitq_faster_config(dim, total_bits) };
        Self { ptr }
    }
}

impl Drop for RabitqConfig {
    fn drop(&mut self) {
        unsafe { ffi::rabitq_config_free(self.ptr) };
    }
}

pub fn quantize_full_single(
    data: &[f32],
    total_bits: usize,
    metric_type: MetricType,
    config: &RabitqConfig,
) -> (Vec<u8>, f32, f32, f32) {
    let dim = data.len();
    let mut total_code = vec![0u8; dim];
    let mut f_add = 0.0;
    let mut f_rescale = 0.0;
    let mut f_error = 0.0;

    unsafe {
        ffi::rabitq_quantize_full_single(
            data.as_ptr(),
            dim,
            total_bits,
            total_code.as_mut_ptr(),
            &mut f_add,
            &mut f_rescale,
            &mut f_error,
            metric_type as u32,
            config.ptr,
        );
    }
    (total_code, f_add, f_rescale, f_error)
}

pub fn quantize_split_single(
    data: &[f32],
    centroid: &[f32],
    ex_bits: usize,
    metric_type: MetricType,
    config: &RabitqConfig,
) -> (Vec<u8>, Vec<u8>) {
    let padded_dim = data.len();
    let mut bin_codes = vec![0u8; padded_dim / 8 + 12];
    let mut ex_codes = vec![0u8; padded_dim * ex_bits / 8 + 8];

    unsafe {
        ffi::rabitq_quantize_split_single(
            data.as_ptr(),
            centroid.as_ptr(),
            padded_dim,
            ex_bits,
            bin_codes.as_mut_ptr() as *mut i8,
            ex_codes.as_mut_ptr() as *mut i8,
            metric_type as u32,
            config.ptr,
        );
    }
    (bin_codes, ex_codes)
}

pub fn reconstruct_vec(quantized_vec: &[u8], delta: f32, vl: f32) -> Vec<f32> {
    let dim = quantized_vec.len();
    let mut results = vec![0.0f32; dim];
    unsafe {
        ffi::rabitq_reconstruct_vec(quantized_vec.as_ptr(), delta, vl, dim, results.as_mut_ptr());
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_and_reconstruct() {
        let dim = 128;
        let total_bits = 5;
        let data: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let config = RabitqConfig::faster_config(dim, total_bits);

        let (quantized, delta, vl, _) =
            quantize_full_single(&data, total_bits, MetricType::L2, &config);
        let reconstructed = reconstruct_vec(&quantized, delta, vl);

        assert_eq!(reconstructed.len(), dim);
        // Simple check, a more robust test would check the error
        assert!(reconstructed.iter().sum::<f32>() > 0.0);
    }
}
