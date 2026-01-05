use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use nalgebra::Vector3;
use wide::{f64x4, CmpLt};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("tuned_constants.rs");
    let mut f = File::create(&dest_path).unwrap();

    // -------------------------------------------------------------------------
    // Auto-Tuning Strategy: Heuristic based on Brute Force Speed
    // -------------------------------------------------------------------------
    // We measure the raw speed of the SIMD kernel.
    // If it's fast (modern CPU), we can afford a higher brute-force threshold
    // because O(N^2) stays cheap for longer.
    // -------------------------------------------------------------------------

    let bf_threshold;
    let parallel_threshold;

    // Perform a quick measurement
    let n_test = 500;
    let mut pos = Vec::with_capacity(n_test);
    for i in 0..n_test {
        pos.push(Vector3::new(i as f64, i as f64, i as f64));
    }
    
    let cutoff = 5.0;
    
    // Average over 5 runs to reduce noise
    let mut total_duration = std::time::Duration::new(0, 0);
    for _ in 0..5 {
        let start = Instant::now();
        let _ = dummy_brute_force_simd(&pos, cutoff);
        total_duration += start.elapsed();
    }
    let duration = total_duration / 5;
    
    println!("cargo:warning=Calibration average duration for 500 atoms: {:?} ({} micros)", duration, duration.as_micros());
    
    // Heuristic Thresholds:
    // 500 atoms BF ~ 125k interactions.
    // < 1000 micros (1ms) -> Fast CPU -> Threshold 1000
    // < 15000 micros (15ms) -> Normal CPU/Debug -> Threshold 800
    // > 15000 micros -> Slow -> Threshold 400
    if duration.as_micros() < 1000 {
        bf_threshold = 1000;
        parallel_threshold = 300;
    } else if duration.as_micros() < 15000 {
        bf_threshold = 800;
        parallel_threshold = 256;
    } else {
        bf_threshold = 400;
        parallel_threshold = 128;
    }

    // Stack threshold capped at 1024 for safety
    let stack_threshold = bf_threshold.min(1024);

    writeln!(f, "pub const BRUTE_FORCE_THRESHOLD: usize = {};", bf_threshold).unwrap();
    writeln!(f, "pub const PARALLEL_THRESHOLD: usize = {};", parallel_threshold).unwrap();
    writeln!(f, "pub const STACK_THRESHOLD: usize = {};", stack_threshold).unwrap();

    println!("cargo:warning=Auto-tuning Result: BRUTE_FORCE_THRESHOLD={}, PARALLEL_THRESHOLD={}, STACK_THRESHOLD={}", 
        bf_threshold, parallel_threshold, stack_threshold);
    
    println!("cargo:rerun-if-changed=build.rs");
}

fn dummy_brute_force_simd(positions: &[Vector3<f64>], cutoff: f64) -> usize {
    let n = positions.len();
    let cutoff_sq = cutoff * cutoff;
    let cutoff_sq_v = f64x4::from(cutoff_sq);
    let mut count = 0;

    let mut px = Vec::with_capacity(n);
    let mut py = Vec::with_capacity(n);
    let mut pz = Vec::with_capacity(n);
    for p in positions {
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
    }

    for i in 0..n {
        let pix = f64x4::from(px[i]);
        let piy = f64x4::from(py[i]);
        let piz = f64x4::from(pz[i]);
        let mut j = i + 1;
        while j + 4 <= n {
            let pjx = f64x4::from(&px[j..j + 4]);
            let pjy = f64x4::from(&py[j..j + 4]);
            let pjz = f64x4::from(&pz[j..j + 4]);
            let dx = pjx - pix;
            let dy = pjy - piy;
            let dz = pjz - piz;
            let d2 = dx * dx + dy * dy + dz * dz;
            let mask = d2.cmp_lt(cutoff_sq_v);
            if mask.any() {
                count += 1;
            }
            j += 4;
        }
        for k in j..n {
            let dx = px[k] - px[i];
            let dy = py[k] - py[i];
            let dz = pz[k] - pz[i];
            if dx*dx + dy*dy + dz*dz < cutoff_sq {
                count += 1;
            }
        }
    }
    count
}
