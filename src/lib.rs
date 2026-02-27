use std::ffi::CStr;
use std::fmt;
use std::mem::MaybeUninit;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::NonNull;

pub type Result<T> = std::result::Result<T, UmapError>;

#[derive(Debug)]
pub enum UmapError {
    InvalidInput(&'static str),
    Ffi(String),
}

impl fmt::Display for UmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UmapError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            UmapError::Ffi(msg) => write!(f, "umappp ffi error: {}", msg),
        }
    }
}

impl std::error::Error for UmapError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializeMethod {
    Spectral,
    Random,
    None,
}

impl InitializeMethod {
    fn to_raw(self) -> u8 {
        match self {
            InitializeMethod::Spectral => 0,
            InitializeMethod::Random => 1,
            InitializeMethod::None => 2,
        }
    }

    fn from_raw(value: u8) -> Option<Self> {
        match value {
            0 => Some(InitializeMethod::Spectral),
            1 => Some(InitializeMethod::Random),
            2 => Some(InitializeMethod::None),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UmapOptions {
    pub local_connectivity: f64,
    pub bandwidth: f64,
    pub mix_ratio: f64,
    pub spread: f64,
    pub min_dist: f64,
    pub a: Option<f64>,
    pub b: Option<f64>,
    pub repulsion_strength: f64,
    pub initialize_method: InitializeMethod,
    pub initialize_random_on_spectral_fail: bool,
    pub initialize_spectral_scale: f64,
    pub initialize_spectral_jitter: bool,
    pub initialize_spectral_jitter_sd: f64,
    pub initialize_random_scale: f64,
    pub initialize_seed: u64,
    pub num_epochs: Option<i32>,
    pub learning_rate: f64,
    pub negative_sample_rate: f64,
    pub num_neighbors: i32,
    pub optimize_seed: u64,
    pub num_threads: i32,
    pub parallel_optimization: bool,
}

impl Default for UmapOptions {
    fn default() -> Self {
        let mut raw = MaybeUninit::<RawOptions>::uninit();
        unsafe {
            umappp_default_options(raw.as_mut_ptr());
            UmapOptions::from_raw(raw.assume_init())
        }
    }
}

impl UmapOptions {
    fn from_raw(raw: RawOptions) -> Self {
        UmapOptions {
            local_connectivity: raw.local_connectivity,
            bandwidth: raw.bandwidth,
            mix_ratio: raw.mix_ratio,
            spread: raw.spread,
            min_dist: raw.min_dist,
            a: if raw.has_a != 0 { Some(raw.a) } else { None },
            b: if raw.has_b != 0 { Some(raw.b) } else { None },
            repulsion_strength: raw.repulsion_strength,
            initialize_method: InitializeMethod::from_raw(raw.initialize_method)
                .unwrap_or(InitializeMethod::Spectral),
            initialize_random_on_spectral_fail: raw.initialize_random_on_spectral_fail != 0,
            initialize_spectral_scale: raw.initialize_spectral_scale,
            initialize_spectral_jitter: raw.initialize_spectral_jitter != 0,
            initialize_spectral_jitter_sd: raw.initialize_spectral_jitter_sd,
            initialize_random_scale: raw.initialize_random_scale,
            initialize_seed: raw.initialize_seed,
            num_epochs: if raw.has_num_epochs != 0 {
                Some(raw.num_epochs)
            } else {
                None
            },
            learning_rate: raw.learning_rate,
            negative_sample_rate: raw.negative_sample_rate,
            num_neighbors: raw.num_neighbors,
            optimize_seed: raw.optimize_seed,
            num_threads: raw.num_threads,
            parallel_optimization: raw.parallel_optimization != 0,
        }
    }

    fn to_raw(&self) -> RawOptions {
        RawOptions {
            local_connectivity: self.local_connectivity,
            bandwidth: self.bandwidth,
            mix_ratio: self.mix_ratio,
            spread: self.spread,
            min_dist: self.min_dist,
            a: self.a.unwrap_or(0.0),
            b: self.b.unwrap_or(0.0),
            has_a: if self.a.is_some() { 1 } else { 0 },
            has_b: if self.b.is_some() { 1 } else { 0 },
            repulsion_strength: self.repulsion_strength,
            initialize_method: self.initialize_method.to_raw(),
            initialize_random_on_spectral_fail: if self.initialize_random_on_spectral_fail {
                1
            } else {
                0
            },
            initialize_spectral_scale: self.initialize_spectral_scale,
            initialize_spectral_jitter: if self.initialize_spectral_jitter {
                1
            } else {
                0
            },
            initialize_spectral_jitter_sd: self.initialize_spectral_jitter_sd,
            initialize_random_scale: self.initialize_random_scale,
            initialize_seed: self.initialize_seed,
            num_epochs: self.num_epochs.unwrap_or(0),
            has_num_epochs: if self.num_epochs.is_some() { 1 } else { 0 },
            learning_rate: self.learning_rate,
            negative_sample_rate: self.negative_sample_rate,
            num_neighbors: self.num_neighbors,
            optimize_seed: self.optimize_seed,
            num_threads: self.num_threads,
            parallel_optimization: if self.parallel_optimization { 1 } else { 0 },
        }
    }
}

pub struct UmapStatus {
    handle: NonNull<c_void>,
    num_dim: usize,
    num_obs: usize,
}

impl UmapStatus {
    pub fn run(&mut self, embedding: &mut [f64], epoch_limit: Option<i32>) -> Result<()> {
        self.ensure_embedding_len(embedding)?;
        let rc = unsafe {
            match epoch_limit {
                Some(limit) => {
                    umappp_status_run(self.handle.as_ptr(), embedding.as_mut_ptr(), limit)
                }
                None => umappp_status_run_full(self.handle.as_ptr(), embedding.as_mut_ptr()),
            }
        };
        if rc != 0 {
            return Err(UmapError::Ffi(take_last_error()));
        }
        Ok(())
    }

    pub fn epoch(&self) -> Result<i32> {
        let value = unsafe { umappp_status_epoch(self.handle.as_ptr()) };
        if value < 0 {
            return Err(UmapError::Ffi(take_last_error()));
        }
        Ok(value)
    }

    pub fn num_epochs(&self) -> Result<i32> {
        let value = unsafe { umappp_status_num_epochs(self.handle.as_ptr()) };
        if value < 0 {
            return Err(UmapError::Ffi(take_last_error()));
        }
        Ok(value)
    }

    pub fn num_dimensions(&self) -> usize {
        self.num_dim
    }

    pub fn num_observations(&self) -> usize {
        self.num_obs
    }

    fn ensure_embedding_len(&self, embedding: &[f64]) -> Result<()> {
        let expected = self
            .num_dim
            .checked_mul(self.num_obs)
            .ok_or(UmapError::InvalidInput("embedding size overflow"))?;
        if embedding.len() != expected {
            return Err(UmapError::InvalidInput("embedding length mismatch"));
        }
        Ok(())
    }
}

impl Drop for UmapStatus {
    fn drop(&mut self) {
        unsafe {
            umappp_status_free(self.handle.as_ptr());
        }
    }
}

pub fn initialize(
    data_dim: usize,
    num_obs: usize,
    data: &[f64],
    num_dim: usize,
    embedding: &mut [f64],
    options: &UmapOptions,
) -> Result<UmapStatus> {
    if data_dim == 0 || num_dim == 0 {
        return Err(UmapError::InvalidInput(
            "data_dim and num_dim must be positive",
        ));
    }
    if num_obs == 0 {
        return Err(UmapError::InvalidInput("num_obs must be positive"));
    }
    let num_obs_i32 =
        i32::try_from(num_obs).map_err(|_| UmapError::InvalidInput("num_obs too large"))?;
    let data_len = data_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("data size overflow"))?;
    if data.len() != data_len {
        return Err(UmapError::InvalidInput("data length mismatch"));
    }
    let embed_len = num_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("embedding size overflow"))?;
    if embedding.len() != embed_len {
        return Err(UmapError::InvalidInput("embedding length mismatch"));
    }

    let raw = options.to_raw();
    let handle = unsafe {
        umappp_initialize(
            data.as_ptr(),
            data_dim,
            num_obs_i32,
            num_dim,
            embedding.as_mut_ptr(),
            &raw,
        )
    };
    let handle = NonNull::new(handle).ok_or_else(|| UmapError::Ffi(take_last_error()))?;
    Ok(UmapStatus {
        handle,
        num_dim,
        num_obs,
    })
}

pub fn fit(
    data_dim: usize,
    num_obs: usize,
    data: &[f64],
    num_dim: usize,
    options: &UmapOptions,
) -> Result<Vec<f64>> {
    let mut embedding = vec![
        0.0;
        num_dim.checked_mul(num_obs).ok_or_else(|| {
            UmapError::InvalidInput("embedding size overflow")
        })?
    ];
    let mut status = initialize(data_dim, num_obs, data, num_dim, &mut embedding, options)?;
    status.run(&mut embedding, None)?;
    Ok(embedding)
}

pub fn fit_rowmajor(
    data_dim: usize,
    num_obs: usize,
    data_rowmajor: &[f64],
    num_dim: usize,
    options: &UmapOptions,
) -> Result<Vec<f64>> {
    if data_dim == 0 || num_dim == 0 {
        return Err(UmapError::InvalidInput(
            "data_dim and num_dim must be positive",
        ));
    }
    if num_obs == 0 {
        return Err(UmapError::InvalidInput("num_obs must be positive"));
    }
    let num_obs_i32 =
        i32::try_from(num_obs).map_err(|_| UmapError::InvalidInput("num_obs too large"))?;
    let data_len = data_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("data size overflow"))?;
    if data_rowmajor.len() != data_len {
        return Err(UmapError::InvalidInput("data length mismatch"));
    }

    let embed_len = num_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("embedding size overflow"))?;
    let mut embedding_rowmajor = vec![0.0f64; embed_len];

    let raw = options.to_raw();
    let rc = unsafe {
        umappp_fit_rowmajor(
            data_rowmajor.as_ptr(),
            data_dim,
            num_obs_i32,
            num_dim,
            &raw,
            embedding_rowmajor.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(UmapError::Ffi(take_last_error()));
    }
    Ok(embedding_rowmajor)
}

pub fn fit_from_knn(
    num_obs: usize,
    k: usize,
    indices: &[u32],
    distances: &[f64],
    num_dim: usize,
    options: &UmapOptions,
) -> Result<Vec<f64>> {
    if num_dim == 0 || k == 0 {
        return Err(UmapError::InvalidInput("num_dim and k must be positive"));
    }
    if num_obs == 0 {
        return Err(UmapError::InvalidInput("num_obs must be positive"));
    }
    let num_obs_i32 =
        i32::try_from(num_obs).map_err(|_| UmapError::InvalidInput("num_obs too large"))?;
    let expected = num_obs
        .checked_mul(k)
        .ok_or(UmapError::InvalidInput("knn size overflow"))?;
    if indices.len() != expected || distances.len() != expected {
        return Err(UmapError::InvalidInput(
            "knn indices/distances length mismatch",
        ));
    }

    let embed_len = num_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("embedding size overflow"))?;
    let mut embedding_rowmajor = vec![0.0f64; embed_len];

    let raw = options.to_raw();
    let rc = unsafe {
        umappp_fit_from_knn(
            indices.as_ptr(),
            distances.as_ptr(),
            k,
            num_obs_i32,
            num_dim,
            &raw,
            embedding_rowmajor.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(UmapError::Ffi(take_last_error()));
    }
    Ok(embedding_rowmajor)
}

pub fn fit_from_knn_f32(
    num_obs: usize,
    k: usize,
    indices: &[u32],
    distances: &[f32],
    num_dim: usize,
    options: &UmapOptions,
) -> Result<Vec<f32>> {
    if num_dim == 0 || k == 0 {
        return Err(UmapError::InvalidInput("num_dim and k must be positive"));
    }
    if num_obs == 0 {
        return Err(UmapError::InvalidInput("num_obs must be positive"));
    }
    let num_obs_i32 =
        i32::try_from(num_obs).map_err(|_| UmapError::InvalidInput("num_obs too large"))?;
    let expected = num_obs
        .checked_mul(k)
        .ok_or(UmapError::InvalidInput("knn size overflow"))?;
    if indices.len() != expected || distances.len() != expected {
        return Err(UmapError::InvalidInput(
            "knn indices/distances length mismatch",
        ));
    }

    let embed_len = num_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("embedding size overflow"))?;
    let mut embedding_rowmajor = vec![0.0f32; embed_len];

    let raw = options.to_raw();
    let rc = unsafe {
        umappp_fit_from_knn_f32(
            indices.as_ptr(),
            distances.as_ptr(),
            k,
            num_obs_i32,
            num_dim,
            &raw,
            embedding_rowmajor.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(UmapError::Ffi(take_last_error()));
    }
    Ok(embedding_rowmajor)
}

pub fn transform_from_knn(
    num_train: usize,
    num_obs: usize,
    k: usize,
    indices: &[u32],
    distances: &[f64],
    num_dim: usize,
    train_embedding_rowmajor: &[f64],
    options: &UmapOptions,
) -> Result<Vec<f64>> {
    if num_dim == 0 || k == 0 {
        return Err(UmapError::InvalidInput("num_dim and k must be positive"));
    }
    if num_obs == 0 || num_train == 0 {
        return Err(UmapError::InvalidInput(
            "num_obs and num_train must be positive",
        ));
    }
    let num_obs_i32 =
        i32::try_from(num_obs).map_err(|_| UmapError::InvalidInput("num_obs too large"))?;
    let num_train_i32 =
        i32::try_from(num_train).map_err(|_| UmapError::InvalidInput("num_train too large"))?;

    let expected = num_obs
        .checked_mul(k)
        .ok_or(UmapError::InvalidInput("knn size overflow"))?;
    if indices.len() != expected || distances.len() != expected {
        return Err(UmapError::InvalidInput(
            "knn indices/distances length mismatch",
        ));
    }

    let train_len = num_train
        .checked_mul(num_dim)
        .ok_or(UmapError::InvalidInput("train embedding size overflow"))?;
    if train_embedding_rowmajor.len() != train_len {
        return Err(UmapError::InvalidInput("train embedding length mismatch"));
    }

    let embed_len = num_dim
        .checked_mul(num_obs)
        .ok_or(UmapError::InvalidInput("embedding size overflow"))?;
    let mut embedding_rowmajor = vec![0.0f64; embed_len];

    let raw = options.to_raw();
    let rc = unsafe {
        umappp_transform_from_knn(
            indices.as_ptr(),
            distances.as_ptr(),
            k,
            num_obs_i32,
            num_train_i32,
            num_dim,
            train_embedding_rowmajor.as_ptr(),
            &raw,
            embedding_rowmajor.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(UmapError::Ffi(take_last_error()));
    }

    Ok(embedding_rowmajor)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawOptions {
    local_connectivity: f64,
    bandwidth: f64,
    mix_ratio: f64,
    spread: f64,
    min_dist: f64,
    a: f64,
    b: f64,
    has_a: u8,
    has_b: u8,
    repulsion_strength: f64,
    initialize_method: u8,
    initialize_random_on_spectral_fail: u8,
    initialize_spectral_scale: f64,
    initialize_spectral_jitter: u8,
    initialize_spectral_jitter_sd: f64,
    initialize_random_scale: f64,
    initialize_seed: u64,
    num_epochs: i32,
    has_num_epochs: u8,
    learning_rate: f64,
    negative_sample_rate: f64,
    num_neighbors: i32,
    optimize_seed: u64,
    num_threads: i32,
    parallel_optimization: u8,
}

unsafe extern "C" {
    fn umappp_default_options(out: *mut RawOptions);
    fn umappp_initialize(
        data: *const f64,
        data_dim: usize,
        num_obs: i32,
        num_dim: usize,
        embedding: *mut f64,
        options: *const RawOptions,
    ) -> *mut c_void;
    fn umappp_status_run(status: *mut c_void, embedding: *mut f64, epoch_limit: c_int) -> c_int;
    fn umappp_status_run_full(status: *mut c_void, embedding: *mut f64) -> c_int;
    fn umappp_status_epoch(status: *mut c_void) -> c_int;
    fn umappp_status_num_epochs(status: *mut c_void) -> c_int;
    fn umappp_status_free(status: *mut c_void);
    fn umappp_last_error() -> *const c_char;
    fn umappp_clear_last_error();
    fn umappp_fit_rowmajor(
        data_rowmajor: *const f64,
        data_dim: usize,
        num_obs: i32,
        num_dim: usize,
        options: *const RawOptions,
        embedding_rowmajor: *mut f64,
    ) -> c_int;
    fn umappp_fit_from_knn(
        indices: *const u32,
        distances: *const f64,
        k: usize,
        num_obs: i32,
        num_dim: usize,
        options: *const RawOptions,
        embedding_rowmajor: *mut f64,
    ) -> c_int;
    fn umappp_fit_from_knn_f32(
        indices: *const u32,
        distances: *const f32,
        k: usize,
        num_obs: i32,
        num_dim: usize,
        options: *const RawOptions,
        embedding_rowmajor: *mut f32,
    ) -> c_int;
    fn umappp_transform_from_knn(
        indices: *const u32,
        distances: *const f64,
        k: usize,
        num_obs: i32,
        num_train: i32,
        num_dim: usize,
        train_embedding_rowmajor: *const f64,
        options: *const RawOptions,
        embedding_rowmajor: *mut f64,
    ) -> c_int;
}

fn take_last_error() -> String {
    unsafe {
        let ptr = umappp_last_error();
        let message = if ptr.is_null() {
            "unknown umappp error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        umappp_clear_last_error();
        message
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::fs;
    use std::io::Write;
    use std::process::Command;

    unsafe extern "C" {
        fn umappp_run_reference(
            data: *const f64,
            data_dim: usize,
            num_obs: i32,
            num_dim: usize,
            options: *const RawOptions,
            embedding: *mut f64,
        ) -> c_int;
    }

    fn make_data(data_dim: usize, num_obs: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut data = vec![0.0; data_dim * num_obs];
        for obs in 0..num_obs {
            for dim in 0..data_dim {
                data[dim + obs * data_dim] = rng.gen_range(-1.0..1.0);
            }
        }
        data
    }

    fn test_options() -> UmapOptions {
        let mut options = UmapOptions::default();
        options.initialize_method = InitializeMethod::Random;
        options.initialize_seed = 123;
        options.optimize_seed = 123;
        options.num_epochs = Some(50);
        options.num_neighbors = 10;
        options.num_threads = 1;
        options.parallel_optimization = false;
        options
    }

    fn build_knn_rowmajor(
        data_colmajor: &[f64],
        data_dim: usize,
        num_obs: usize,
        k: usize,
    ) -> (Vec<u32>, Vec<f64>) {
        let mut indices = vec![0u32; num_obs * k];
        let mut distances = vec![0.0f64; num_obs * k];

        for i in 0..num_obs {
            let mut dists = Vec::with_capacity(num_obs.saturating_sub(1));
            for j in 0..num_obs {
                if i == j {
                    continue;
                }
                let mut sum = 0.0f64;
                for d in 0..data_dim {
                    let left = data_colmajor[d + i * data_dim];
                    let right = data_colmajor[d + j * data_dim];
                    let delta = left - right;
                    sum += delta * delta;
                }
                dists.push((j as u32, sum.sqrt()));
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("invalid distance ordering"));
            for (offset, (j, dist)) in dists.into_iter().take(k).enumerate() {
                let out = i * k + offset;
                indices[out] = j;
                distances[out] = dist;
            }
        }

        (indices, distances)
    }

    fn lowdim_knn_rowmajor(
        embedding: &[f64],
        num_dim: usize,
        num_obs: usize,
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut out = vec![vec![0usize; k]; num_obs];
        for i in 0..num_obs {
            let mut dists = Vec::with_capacity(num_obs.saturating_sub(1));
            for j in 0..num_obs {
                if i == j {
                    continue;
                }
                let mut sum = 0.0f64;
                for d in 0..num_dim {
                    let left = embedding[i * num_dim + d];
                    let right = embedding[j * num_dim + d];
                    let delta = left - right;
                    sum += delta * delta;
                }
                dists.push((j, sum));
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("invalid distance ordering"));
            for (slot, (j, _dist)) in dists.into_iter().take(k).enumerate() {
                out[i][slot] = j;
            }
        }
        out
    }

    fn knn_overlap(reference: &[Vec<usize>], observed: &[Vec<usize>]) -> f64 {
        assert_eq!(reference.len(), observed.len());
        let n = reference.len();
        if n == 0 {
            return 0.0;
        }
        let k = reference[0].len();
        if k == 0 {
            return 0.0;
        }
        let mut total = 0usize;
        for i in 0..n {
            let r = &reference[i];
            let o = &observed[i];
            for idx in o {
                if r.contains(idx) {
                    total += 1;
                }
            }
        }
        total as f64 / (n * k) as f64
    }

    #[test]
    fn compare_cpp_reference() -> Result<()> {
        let data_dim = 5;
        let num_obs = 40;
        let num_dim = 2;
        let data = make_data(data_dim, num_obs);
        let options = test_options();

        let mut embedding = vec![0.0; num_dim * num_obs];
        let mut status = initialize(data_dim, num_obs, &data, num_dim, &mut embedding, &options)?;
        status.run(&mut embedding, None)?;

        let mut reference = vec![0.0; num_dim * num_obs];
        let raw = options.to_raw();
        let rc = unsafe {
            umappp_run_reference(
                data.as_ptr(),
                data_dim,
                num_obs as i32,
                num_dim,
                &raw,
                reference.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(UmapError::Ffi(take_last_error()));
        }

        for (idx, (left, right)) in embedding.iter().zip(reference.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff < 1e-10,
                "embedding mismatch at {}: {} vs {} (diff {})",
                idx,
                left,
                right,
                diff
            );
        }

        Ok(())
    }

    #[test]
    fn compare_python_umap_optional() {
        if std::env::var("UMAPPP_PYTHON_COMPARE").ok().as_deref() != Some("1") {
            return;
        }

        let python = match find_python() {
            Some(binary) => binary,
            None => panic!("python3 or python not found in PATH"),
        };

        if !python_has_umap(&python) {
            panic!("python umap-learn not available in the selected interpreter");
        }

        let data_dim = 5;
        let num_obs = 40;
        let num_dim = 2;
        let data = make_data(data_dim, num_obs);
        let options = test_options();

        let mut embedding = vec![0.0; num_dim * num_obs];
        let mut status =
            initialize(data_dim, num_obs, &data, num_dim, &mut embedding, &options).unwrap();
        status.run(&mut embedding, None).unwrap();

        let row_major = to_row_major(&data, data_dim, num_obs);
        let data_file = tempfile::NamedTempFile::new().expect("tempfile create failed");
        let embed_file = tempfile::NamedTempFile::new().expect("tempfile create failed");
        write_csv(&data_file.path(), data_dim, num_obs, &row_major);

        run_python_umap(
            &python,
            data_file.path(),
            embed_file.path(),
            &options,
            num_dim,
        );

        let python_embedding = read_csv(embed_file.path());
        let rust_dist = pairwise_distances_col_major(&embedding, num_dim, num_obs);
        let python_dist = pairwise_distances_row_major(&python_embedding, num_dim, num_obs);
        let corr = pearson_corr(&rust_dist, &python_dist);

        assert!(
            corr > 0.85,
            "pairwise distance correlation too low: {}",
            corr
        );
    }

    #[test]
    fn fit_from_knn_f32_tracks_f64() -> Result<()> {
        let data_dim = 6;
        let num_obs = 48;
        let num_dim = 3;
        let k = 10;
        let data = make_data(data_dim, num_obs);
        let options = test_options();

        let (indices, distances_f64) = build_knn_rowmajor(&data, data_dim, num_obs, k);
        let embedding_f64 = fit_from_knn(num_obs, k, &indices, &distances_f64, num_dim, &options)?;
        let distances_f32 = distances_f64.iter().map(|v| *v as f32).collect::<Vec<_>>();
        let embedding_f32 =
            fit_from_knn_f32(num_obs, k, &indices, &distances_f32, num_dim, &options)?;

        assert_eq!(embedding_f64.len(), embedding_f32.len());
        let reference_knn = indices
            .chunks(k)
            .map(|chunk| chunk.iter().map(|idx| *idx as usize).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let f64_knn = lowdim_knn_rowmajor(&embedding_f64, num_dim, num_obs, k);
        let embedding_f32_as_f64 = embedding_f32.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let f32_knn = lowdim_knn_rowmajor(&embedding_f32_as_f64, num_dim, num_obs, k);

        let f64_overlap = knn_overlap(&reference_knn, &f64_knn);
        let f32_overlap = knn_overlap(&reference_knn, &f32_knn);
        assert!(
            (f64_overlap - f32_overlap).abs() < 0.05,
            "f32/f64 quality drift too large: f64_overlap={} f32_overlap={}",
            f64_overlap,
            f32_overlap
        );

        Ok(())
    }

    fn to_row_major(data: &[f64], data_dim: usize, num_obs: usize) -> Vec<f64> {
        let mut output = vec![0.0; data_dim * num_obs];
        for obs in 0..num_obs {
            for dim in 0..data_dim {
                output[obs * data_dim + dim] = data[dim + obs * data_dim];
            }
        }
        output
    }

    fn write_csv(path: &std::path::Path, data_dim: usize, num_obs: usize, data: &[f64]) {
        let mut file = fs::File::create(path).expect("csv create failed");
        for obs in 0..num_obs {
            let start = obs * data_dim;
            let row = &data[start..start + data_dim];
            for (idx, value) in row.iter().enumerate() {
                if idx > 0 {
                    write!(file, ",").expect("csv write failed");
                }
                write!(file, "{}", value).expect("csv write failed");
            }
            writeln!(file).expect("csv write failed");
        }
    }

    fn read_csv(path: &std::path::Path) -> Vec<f64> {
        let content = fs::read_to_string(path).expect("csv read failed");
        let mut values = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            for item in trimmed.split(',') {
                values.push(item.parse::<f64>().expect("csv parse failed"));
            }
        }
        values
    }

    fn find_python() -> Option<String> {
        for candidate in ["python3", "python"] {
            let ok = Command::new(candidate)
                .arg("-c")
                .arg("import sys")
                .status()
                .map(|status| status.success())
                .unwrap_or(false);
            if ok {
                return Some(candidate.to_string());
            }
        }
        None
    }

    fn python_has_umap(python: &str) -> bool {
        Command::new(python)
            .arg("-c")
            .arg("import umap, numpy")
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn run_python_umap(
        python: &str,
        data_path: &std::path::Path,
        output_path: &std::path::Path,
        options: &UmapOptions,
        num_dim: usize,
    ) {
        let script = r#"
import os
import numpy as np
import umap

data = np.loadtxt(os.environ["UMAPPP_DATA_PATH"], delimiter=",")
model = umap.UMAP(
    n_neighbors=int(os.environ["UMAPPP_NUM_NEIGHBORS"]),
    n_components=int(os.environ["UMAPPP_NUM_DIM"]),
    min_dist=float(os.environ["UMAPPP_MIN_DIST"]),
    spread=float(os.environ["UMAPPP_SPREAD"]),
    n_epochs=int(os.environ["UMAPPP_NUM_EPOCHS"]),
    learning_rate=float(os.environ["UMAPPP_LEARNING_RATE"]),
    repulsion_strength=float(os.environ["UMAPPP_REPULSION_STRENGTH"]),
    negative_sample_rate=float(os.environ["UMAPPP_NEGATIVE_SAMPLE_RATE"]),
    set_op_mix_ratio=float(os.environ["UMAPPP_MIX_RATIO"]),
    local_connectivity=float(os.environ["UMAPPP_LOCAL_CONNECTIVITY"]),
    init="random",
    random_state=int(os.environ["UMAPPP_SEED"]),
    metric="euclidean",
)
embedding = model.fit_transform(data)
np.savetxt(os.environ["UMAPPP_OUT_PATH"], embedding, delimiter=",")
"#;

        let status = Command::new(python)
            .arg("-c")
            .arg(script)
            .env("UMAPPP_DATA_PATH", data_path)
            .env("UMAPPP_OUT_PATH", output_path)
            .env("UMAPPP_NUM_NEIGHBORS", options.num_neighbors.to_string())
            .env("UMAPPP_NUM_DIM", num_dim.to_string())
            .env("UMAPPP_MIN_DIST", options.min_dist.to_string())
            .env("UMAPPP_SPREAD", options.spread.to_string())
            .env(
                "UMAPPP_NUM_EPOCHS",
                options.num_epochs.unwrap_or(0).to_string(),
            )
            .env("UMAPPP_LEARNING_RATE", options.learning_rate.to_string())
            .env(
                "UMAPPP_REPULSION_STRENGTH",
                options.repulsion_strength.to_string(),
            )
            .env(
                "UMAPPP_NEGATIVE_SAMPLE_RATE",
                options.negative_sample_rate.to_string(),
            )
            .env("UMAPPP_MIX_RATIO", options.mix_ratio.to_string())
            .env(
                "UMAPPP_LOCAL_CONNECTIVITY",
                options.local_connectivity.to_string(),
            )
            .env("UMAPPP_SEED", options.initialize_seed.to_string())
            .status()
            .expect("python invocation failed");

        assert!(status.success(), "python umap run failed");
    }

    fn pairwise_distances_col_major(embedding: &[f64], num_dim: usize, num_obs: usize) -> Vec<f64> {
        let mut distances = Vec::new();
        for i in 0..num_obs {
            for j in (i + 1)..num_obs {
                let mut sum = 0.0;
                for d in 0..num_dim {
                    let left = embedding[d + i * num_dim];
                    let right = embedding[d + j * num_dim];
                    let delta = left - right;
                    sum += delta * delta;
                }
                distances.push(sum.sqrt());
            }
        }
        distances
    }

    fn pairwise_distances_row_major(embedding: &[f64], num_dim: usize, num_obs: usize) -> Vec<f64> {
        let mut distances = Vec::new();
        for i in 0..num_obs {
            for j in (i + 1)..num_obs {
                let mut sum = 0.0;
                for d in 0..num_dim {
                    let left = embedding[i * num_dim + d];
                    let right = embedding[j * num_dim + d];
                    let delta = left - right;
                    sum += delta * delta;
                }
                distances.push(sum.sqrt());
            }
        }
        distances
    }

    fn pearson_corr(left: &[f64], right: &[f64]) -> f64 {
        assert_eq!(left.len(), right.len());
        let n = left.len() as f64;
        let mean_left = left.iter().sum::<f64>() / n;
        let mean_right = right.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut denom_left = 0.0;
        let mut denom_right = 0.0;
        for (a, b) in left.iter().zip(right.iter()) {
            let da = a - mean_left;
            let db = b - mean_right;
            num += da * db;
            denom_left += da * da;
            denom_right += db * db;
        }
        num / (denom_left.sqrt() * denom_right.sqrt())
    }
}
