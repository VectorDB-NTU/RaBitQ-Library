use rabitq_sys::{MetricType, SplitBatchQuery, rabitq_split_batch_query_new, rabitq_split_batch_query_free, rabitq_split_batch_query_set_g_add, rabitq_split_batch_estdist, rabitq_split_distance_boosting_with_batch_query, rabitq_select_excode_ipfunc, ex_ipfunc};

pub fn select_excode_ipfunc(ex_bits: usize) -> ex_ipfunc {
    unsafe { rabitq_select_excode_ipfunc(ex_bits) }
}

pub struct Estimator {
    ptr: *mut SplitBatchQuery,
    padded_dim: usize,
}

impl Estimator {
    pub fn new(rotated_query: &[f32], padded_dim: usize, ex_bits: usize, metric_type: MetricType, use_hacc: bool) -> Self {
        let ptr = unsafe {
            rabitq_split_batch_query_new(rotated_query.as_ptr(), padded_dim, ex_bits, metric_type, use_hacc)
        };
        Self { ptr, padded_dim }
    }

    pub fn set_g_add(&mut self, norm: f32, ip: f32) {
        unsafe {
            rabitq_split_batch_query_set_g_add(self.ptr, norm, ip);
        }
    }

    pub fn estdist(&self, batch_data: &[u8], padded_dim: usize, use_hacc: bool) -> (f32, f32, f32) {
        let mut est_distance = 0.0;
        let mut low_distance = 0.0;
        let mut ip_x0_qr = 0.0;
        unsafe {
            rabitq_split_batch_estdist(
                batch_data.as_ptr() as *const i8,
                self.ptr,
                padded_dim,
                &mut est_distance,
                &mut low_distance,
                &mut ip_x0_qr,
                use_hacc,
            );
        }
        (est_distance, low_distance, ip_x0_qr)
    }

    pub fn distance_boosting(
        &self,
        ex_data: &[u8],
        ip_func: Option<unsafe extern "C" fn(*const f32, *const u8, usize) -> f32>,
        ex_bits: usize,
        ip_x0_qr: f32,
    ) -> f32 {
        unsafe {
            rabitq_split_distance_boosting_with_batch_query(
                ex_data.as_ptr() as *const i8,
                ip_func,
                self.ptr,
                self.padded_dim,
                ex_bits,
                ip_x0_qr,
            )
        }
    }
}

impl Drop for Estimator {
    fn drop(&mut self) {
        unsafe {
            rabitq_split_batch_query_free(self.ptr);
        }
    }
}