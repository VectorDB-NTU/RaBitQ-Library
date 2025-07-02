use rabitq_sys::{
    MetricType, SplitBatchQuery, SplitSingleQuery, ex_ipfunc, rabitq_select_excode_ipfunc,
    rabitq_split_batch_estdist, rabitq_split_batch_query_free, rabitq_split_batch_query_new,
    rabitq_split_batch_query_set_g_add, rabitq_split_distance_boosting_with_batch_query,
    rabitq_split_distance_boosting_with_single_query, rabitq_split_single_estdist,
    rabitq_split_single_query_delta, rabitq_split_single_query_free, rabitq_split_single_query_new,
    rabitq_split_single_query_query_bin, rabitq_split_single_query_set_g_add,
    rabitq_split_single_query_vl,
};

use crate::RabitqConfig;

pub fn select_excode_ipfunc(ex_bits: usize) -> ex_ipfunc {
    unsafe { rabitq_select_excode_ipfunc(ex_bits) }
}

pub struct BatchEstimator {
    ptr: *mut SplitBatchQuery,
    padded_dim: usize,
}

impl BatchEstimator {
    pub fn new(
        rotated_query: &[f32],
        padded_dim: usize,
        ex_bits: usize,
        metric_type: MetricType,
        use_hacc: bool,
    ) -> Self {
        let ptr = unsafe {
            rabitq_split_batch_query_new(
                rotated_query.as_ptr(),
                padded_dim,
                ex_bits,
                metric_type,
                use_hacc,
            )
        };
        Self { ptr, padded_dim }
    }

    pub fn set_g_add(&mut self, norm: f32, ip: f32) {
        unsafe {
            rabitq_split_batch_query_set_g_add(self.ptr, norm, ip);
        }
    }

    pub fn estdist(&self, batch_data: &[u8], use_hacc: bool) -> (f32, f32, f32) {
        let mut est_distance = 0.0;
        let mut low_distance = 0.0;
        let mut ip_x0_qr = 0.0;
        unsafe {
            rabitq_split_batch_estdist(
                batch_data.as_ptr() as *const i8,
                self.ptr,
                self.padded_dim,
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

impl Drop for BatchEstimator {
    fn drop(&mut self) {
        unsafe {
            rabitq_split_batch_query_free(self.ptr);
        }
    }
}

pub struct SingleEstimator {
    ptr: *mut SplitSingleQuery,
    padded_dim: usize,
    ex_bits: usize,
}

impl SingleEstimator {
    pub fn new(
        rotated_query: &[f32],
        padded_dim: usize,
        ex_bits: usize,
        config: &RabitqConfig,
        metric_type: MetricType,
    ) -> Self {
        let ptr = unsafe {
            rabitq_split_single_query_new(
                rotated_query.as_ptr(),
                padded_dim,
                ex_bits,
                config.ptr as *const _,
                metric_type,
            )
        };
        Self {
            ptr,
            padded_dim,
            ex_bits,
        }
    }

    pub fn query_bin(&self) -> &[u64] {
        let data = unsafe { rabitq_split_single_query_query_bin(self.ptr) };
        let len = self.padded_dim / 8; // a u64 is 8 bytes
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    pub fn delta(&self) -> f32 {
        unsafe { rabitq_split_single_query_delta(self.ptr) }
    }

    pub fn vl(&self) -> f32 {
        unsafe { rabitq_split_single_query_vl(self.ptr) }
    }

    pub fn set_g_add(&mut self, norm: f32, ip: f32) {
        unsafe { rabitq_split_single_query_set_g_add(self.ptr, norm, ip) };
    }

    pub fn estdist(&self, bin_data: &[u8], g_add: f32, g_error: f32) -> (f32, f32, f32) {
        let mut est_dist = 0.0;
        let mut ip_x0_qr: f32 = 0.0;
        let mut low_dist: f32 = 0.0;
        unsafe {
            rabitq_split_single_estdist(
                bin_data.as_ptr() as *const i8,
                self.ptr,
                self.padded_dim,
                &mut ip_x0_qr,
                &mut est_dist,
                &mut low_dist,
                g_add,
                g_error,
            )
        }
        (est_dist, low_dist, ip_x0_qr)
    }

    pub fn distance_boosting(
        &self,
        ex_data: &[u8],
        ip_func: Option<unsafe extern "C" fn(*const f32, *const u8, usize) -> f32>,
        ip_x0_qr: f32,
    ) -> f32 {
        unsafe {
            rabitq_split_distance_boosting_with_single_query(
                ex_data.as_ptr() as *const i8,
                ip_func,
                self.ptr,
                self.padded_dim,
                self.ex_bits,
                ip_x0_qr,
            )
        }
    }
}

impl Drop for SingleEstimator {
    fn drop(&mut self) {
        unsafe {
            rabitq_split_single_query_free(self.ptr);
        }
    }
}
