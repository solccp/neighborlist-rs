use neighborlist_rs::{HalfList, build_half_list, build_neighborlists};
use proptest::prelude::*;

type Pair = (u32, u32, [i32; 3]);
/// Cell rows, pbc, positions and cutoff.
type System = ([[f64; 3]; 3], [bool; 3], Vec<[f64; 3]>, f64);

fn pairs(l: &HalfList) -> Vec<Pair> {
    let mut p: Vec<Pair> = (0..l.len())
        .map(|e| (l.i[e], l.j[e], l.shifts[e]))
        .collect();
    p.sort_unstable();
    p
}

fn inverse(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |r: usize, s: usize| {
        let (r1, r2, s1, s2) = ((r + 1) % 3, (r + 2) % 3, (s + 1) % 3, (s + 2) % 3);
        m[r1][s1] * m[r2][s2] - m[r1][s2] * m[r2][s1]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    std::array::from_fn(|r| std::array::from_fn(|s| c(s, r) / det))
}

/// Independent reference: every image of every pair i <= j, by enumeration.
fn brute_force(pos: &[[f64; 3]], cell: Option<(&[[f64; 3]; 3], [bool; 3])>, rc: f64) -> Vec<Pair> {
    let (h, pbc) = match cell {
        Some((h, pbc)) if pbc.iter().any(|&p| p) => (*h, pbc),
        _ => (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [false; 3],
        ),
    };
    let inv = inverse(&h);
    // Perpendicular width along axis k is 1 / |column k of inv|.
    let perp: [f64; 3] =
        std::array::from_fn(|k| 1.0 / (0..3).map(|a| inv[a][k].powi(2)).sum::<f64>().sqrt());
    let mut out = Vec::new();
    for i in 0..pos.len() {
        for j in i..pos.len() {
            let d: [f64; 3] = std::array::from_fn(|a| pos[j][a] - pos[i][a]);
            let f: [f64; 3] = std::array::from_fn(|k| (0..3).map(|a| d[a] * inv[a][k]).sum());
            let range: [(i32, i32); 3] = std::array::from_fn(|k| {
                if pbc[k] {
                    let c = -f[k].round() as i32;
                    let r = (rc / perp[k]).ceil() as i32 + 1;
                    (c - r, c + r)
                } else {
                    (0, 0)
                }
            });
            for sx in range[0].0..=range[0].1 {
                for sy in range[1].0..=range[1].1 {
                    for sz in range[2].0..=range[2].1 {
                        let s = [sx, sy, sz];
                        if i == j && s <= [0, 0, 0] {
                            continue;
                        }
                        let r: [f64; 3] = std::array::from_fn(|a| {
                            d[a] + (0..3).map(|k| s[k] as f64 * h[k][a]).sum::<f64>()
                        });
                        if r[0] * r[0] + r[1] * r[1] + r[2] * r[2] < rc * rc {
                            out.push((i as u32, j as u32, s));
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// Random cell (rows = lattice vectors), pbc flags, positions and cutoff.
/// Positions may lie outside the cell, and the cutoff ranges from well below
/// the cell size to several times it.
fn system() -> impl Strategy<Value = System> {
    let len = 3.0..9.0f64;
    let tilt = -0.4..0.4f64;
    (
        (len.clone(), len.clone(), len),
        (tilt.clone(), tilt.clone(), tilt),
        any::<[bool; 3]>(),
        prop::collection::vec([-0.3..1.3f64, -0.3..1.3f64, -0.3..1.3f64], 1..24),
        0.5..2.5f64,
    )
        .prop_map(|((a, b, c), (t1, t2, t3), pbc, frac, rc_rel)| {
            let h = [[a, 0.0, 0.0], [t1 * a, b, 0.0], [t2 * a, t3 * b, c]];
            let pos = frac
                .iter()
                .map(|f| std::array::from_fn(|k| (0..3).map(|r| f[r] * h[r][k]).sum()))
                .collect();
            (h, pbc, pos, rc_rel * a.min(b).min(c))
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn matches_brute_force((h, pbc, pos, rc) in system()) {
        let got = pairs(&build_half_list(&pos, rc, Some((&h, pbc))).unwrap());
        prop_assert_eq!(got, brute_force(&pos, Some((&h, pbc)), rc));
    }

    #[test]
    fn isolated_matches_brute_force((_h, _pbc, pos, rc) in system()) {
        let got = pairs(&build_half_list(&pos, rc, None).unwrap());
        prop_assert_eq!(got, brute_force(&pos, None, rc));
    }
}

#[test]
fn matches_build_neighborlists_and_is_thread_count_independent() {
    // Oblique Al(111)-like cell, pbc T,T,F, cutoff larger than the cell.
    let a = 2.86;
    let h = [
        [4.0 * a, 0.0, 0.0],
        [2.0 * a, 4.0 * a * 0.75f64.sqrt(), 0.0],
        [0.0, 0.0, 30.0],
    ];
    let mut pos = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let f = [
                    x as f64 / 4.0 + 0.01 * z as f64,
                    y as f64 / 4.0,
                    0.3 + 0.07 * z as f64,
                ];
                pos.push(std::array::from_fn(|k| {
                    (0..3).map(|r| f[r] * h[r][k]).sum()
                }));
            }
        }
    }
    let rc = 20.0;
    let cell = Some((&h, [true, true, false]));

    let reference = build_neighborlists(&pos, rc, cell, false).unwrap();
    let n = reference.edge_index.len() / 2;
    let mut expected: Vec<Pair> = (0..n)
        .map(|e| {
            let s: [i32; 3] = std::array::from_fn(|k| reference.shifts[3 * e + k]);
            let (i, j) = (
                reference.edge_index[e] as u32,
                reference.edge_index[n + e] as u32,
            );
            if i < j || (i == j && s > [0, 0, 0]) {
                (i, j, s)
            } else {
                (j, i, s.map(|c| -c))
            }
        })
        .collect();
    expected.sort_unstable();

    let runs: Vec<HalfList> = [1, 3, 8]
        .iter()
        .map(|&t| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| build_half_list(&pos, rc, cell).unwrap())
        })
        .collect();
    assert_eq!(pairs(&runs[0]), expected);
    assert!(
        runs.iter().all(|r| *r == runs[0]),
        "output depends on thread count"
    );
}

#[test]
fn edge_cases() {
    let h = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
    assert!(
        build_half_list(&[], 3.0, Some((&h, [true; 3])))
            .unwrap()
            .is_empty()
    );

    // One atom, cutoff 6 in a 5 A box: three self-image pairs.
    let one = build_half_list(&[[0.3, 0.2, 0.1]], 6.0, Some((&h, [true; 3]))).unwrap();
    assert_eq!(
        pairs(&one),
        vec![(0, 0, [0, 0, 1]), (0, 0, [0, 1, 0]), (0, 0, [1, 0, 0])]
    );

    // All atoms in one plane along a non-periodic axis (zero extent).
    let flat: Vec<[f64; 3]> = (0..10)
        .map(|k| [k as f64 * 0.9, 0.5 * k as f64, 2.0])
        .collect();
    let got = pairs(&build_half_list(&flat, 4.0, Some((&h, [true, true, false]))).unwrap());
    assert_eq!(
        got,
        brute_force(&flat, Some((&h, [true, true, false])), 4.0)
    );

    // Sparse isolated system: bins are capped instead of spanning ~1000 A in 2 A steps.
    let sparse = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]];
    assert_eq!(
        pairs(&build_half_list(&sparse, 1.5, None).unwrap()),
        vec![(0, 1, [0, 0, 0])]
    );

    assert!(build_half_list(&[[0.0; 3]], 0.0, None).is_err());
    assert!(build_half_list(&[[0.0; 3]], f64::NAN, None).is_err());
    assert!(build_half_list(&[[f64::NAN, 0.0, 0.0]], 1.0, None).is_err());
    let singular = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(build_half_list(&[[0.0; 3]], 1.0, Some((&singular, [true; 3]))).is_err());
}
