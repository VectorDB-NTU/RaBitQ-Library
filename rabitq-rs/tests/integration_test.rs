use rabitq_rs::quantizer::{quantize_split_single, MetricType};
use rabitq_rs::rotator::Rotator;
use rabitq_rs::estimator::{select_excode_ipfunc};
use ndarray::{Array, Array1, Array2};
use rabitq_rs::{SingleEstimator, RabitqConfig};
use rand::distr::{Distribution, Uniform};

#[test]
fn test_quantization_and_estimation() {
    const DIM: usize = 256;
    const PADDED_DIM: usize = 256;
    const NUM_VECTORS: usize = 100;
    const EX_BITS: usize = 4;

    // 1. Generate random vectors
    let mut rng = rand::rng();
    let unif = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let vectors: Array2<f32> = Array::from_shape_fn((NUM_VECTORS, DIM), |_| unif.sample(&mut rng));
    let query_vec: Array1<f32> = Array::from_shape_fn(DIM, |_| unif.sample(&mut rng));

    // 2. Create a rotator and rotate vectors
    let rotator = Rotator::new(DIM, PADDED_DIM).expect("Failed to create rotator");
    let mut rotated_vectors = Array2::<f32>::zeros((NUM_VECTORS, rotator.padded_dim()));
    for i in 0..NUM_VECTORS {
        let mut rotated_row = rotated_vectors.row_mut(i);
        rotator.rotate(vectors.row(i).as_slice().unwrap(), rotated_row.as_slice_mut().unwrap());
    }
    let mut rotated_query = Array1::<f32>::zeros(rotator.padded_dim());
    rotator.rotate(query_vec.as_slice().unwrap(), rotated_query.as_slice_mut().unwrap());

    // 3. Quantize with centroid at origin
    let centroid = vec![0.0f32; rotator.padded_dim()];

    let (bin_codes, ex_codes) = quantize_split_single(
        rotated_vectors.row(0).as_slice().unwrap(),
        &centroid,
        EX_BITS,
        MetricType::L2,
        &RabitqConfig::new(),
    );

    // 4. Query
    let mut query = SingleEstimator::new(rotated_query.as_slice().unwrap(), rotator.padded_dim(), EX_BITS, &RabitqConfig::new(), 0);

    // 4.1 estimate using 1-bit encoding
    println!("rotator dim: {:?}", bin_codes.len());
    let g_add = rotated_query.pow2().sum();
    let g_err = rotated_query.pow2().sum().sqrt();
    let (dist, low_dist, ip) = query.estdist(&bin_codes, g_add, g_err);
    // 计算查询与第一个向量的l2距离
    let l2_dist = (query_vec - rotated_vectors.row(0)).pow2().sum().sqrt();
    println!("acc dist: {:}", l2_dist);
    println!("dist: {:?}, low_dist: {:?}, ip: {:?}", dist.sqrt(), low_dist.sqrt(), ip);

    let ip_func = select_excode_ipfunc(EX_BITS).expect("Failed to get ip function");

    query.set_g_add(rotated_query.pow2().sum().sqrt(), 0.0);
    let estimated_dist = query.distance_boosting(
        &ex_codes,
        Some(ip_func),
        ip,
    );

    println!("estimated dist: {:}", estimated_dist.sqrt());
}