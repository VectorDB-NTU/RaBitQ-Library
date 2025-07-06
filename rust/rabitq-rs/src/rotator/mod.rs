use rabitq_sys as ffi;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;

pub const ALIGNED_DIM: usize = 16;
/// A wrapper Rotator
pub struct Rotator {
    ptr: *mut ffi::Rotator,
}

impl Rotator {
    /// Create a new Rotator
    pub fn new(dim: usize, padded_dim: usize) -> Option<Self> {
        let ptr = unsafe { ffi::rabitq_rotator_new(dim, padded_dim) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Rotate a raw vector
    pub fn rotate(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(y.len(), self.padded_dim());
        unsafe {
            ffi::rabitq_rotator_rotate(self.ptr, x.as_ptr(), y.as_mut_ptr());
        }
    }

    /// Load rotator matrix param from file
    pub fn load(&mut self, path: &Path) -> Result<(), std::io::Error> {
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let result =
            unsafe { ffi::rabitq_rotator_load(self.ptr, c_path.as_ptr() as *const c_char) };
        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to load rotator",
            ))
        }
    }

    /// Save the rotator matrix param to file
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let result =
            unsafe { ffi::rabitq_rotator_save(self.ptr, c_path.as_ptr() as *const c_char) };
        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to save rotator",
            ))
        }
    }

    /// Get the padded dimension
    pub fn padded_dim(&self) -> usize {
        unsafe { ffi::rabitq_rotator_size(self.ptr) }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        unsafe { ffi::rabitq_rotator_dim(self.ptr) }
    }
}

impl Drop for Rotator {
    fn drop(&mut self) {
        unsafe {
            ffi::rabitq_rotator_free(self.ptr);
        }
    }
}

unsafe impl Send for Rotator {}
unsafe impl Sync for Rotator {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_rotator_creation() {
        let rotator = Rotator::new(128, 128);
        assert!(rotator.is_some());
        let rotator = rotator.unwrap();
        assert_eq!(rotator.padded_dim(), 128);
    }

    #[test]
    fn test_rotator_rotate() {
        let rotator = Rotator::new(128, 128).unwrap();
        let x = vec![1.0f32; 128];
        let mut y = vec![0.0f32; 128];
        rotator.rotate(&x, &mut y);
        // Simple sanity test
        assert!(y.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_rotator_save_load() {
        let rotator = Rotator::new(128, 128).unwrap();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let save_result = rotator.save(path);
        assert!(save_result.is_ok());

        let mut new_rotator = Rotator::new(128, 128).unwrap();
        let load_result = new_rotator.load(path);
        assert!(load_result.is_ok());
    }
}
