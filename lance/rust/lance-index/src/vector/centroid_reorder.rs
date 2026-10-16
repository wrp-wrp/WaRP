// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Centroid reordering for IVF indices.
//!
//! Given `nlist` k-means centroids for an IVF index, compute a permutation
//! such that centroids close in feature space are close in partition-id. The
//! IVF scan then lays out `auxiliary.idx` partition-by-partition, so
//! reordering centroids turns into physical byte-level locality: a query's
//! probed partitions cluster into a handful of contiguous byte ranges,
//! letting the merge planner issue far fewer object-store requests.
//!
//! The default strategy is a 2-D PCA projection followed by a Hilbert-curve
//! sort — matches the paper's `PCA+Hilbert` reorder variant.

use lance_core::{Error, Result};
use snafu::location;

/// Supported centroid reorder strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentroidReorder {
    /// 2D PCA projection + Hilbert-curve 1-D index sort.
    PcaHilbert,
}

impl CentroidReorder {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hilbert" | "pca_hilbert" | "pca+hilbert" => Some(Self::PcaHilbert),
            _ => None,
        }
    }
}

/// Compute a permutation `perm` of length `nlist` such that
/// `new_centroids[i] = old_centroids[perm[i]]` orders centroids by an
/// n-D Morton (Z-order) curve over their top-k PCA projection. `k` is
/// chosen adaptively to keep ≥35% cumulative variance, capped at 8 to
/// fit a u64 Morton index (8-bit per axis × 8 axes = 64-bit).
///
/// 2-D PCA + Hilbert (the original variant) collapses to ~15-26% of the
/// centroid variance on dim≥96 datasets (sift100m, deep100m, glove),
/// so adjacent partition-IDs after the sort are not actually nearby in
/// feature space, defeating the purpose. Bumping to top-k where
/// cum_var≥35% restores meaningful locality even when no single PC
/// dominates. Z-order replaces Hilbert because it generalises trivially
/// to n-D as bit-interleave; the small locality difference (≈10-20%
/// theoretical) is dwarfed by the 4-8× variance gain from extra axes.
pub fn hilbert_permutation_f32(centroids: &[f32], nlist: usize, dim: usize) -> Result<Vec<usize>> {
    if centroids.len() != nlist * dim {
        return Err(Error::InvalidInput {
            source: format!(
                "hilbert_permutation: centroids.len()={} != nlist*dim={}",
                centroids.len(),
                nlist * dim
            )
            .into(),
            location: location!(),
        });
    }
    if nlist == 0 {
        return Ok(Vec::new());
    }
    const VAR_THRESHOLD: f64 = 0.35;
    const K_MAX: usize = 8;
    let (coords_per_pc, eigenvals, total_var) = project_pca_kd(centroids, nlist, dim, K_MAX);
    // total_var here is trace(cov) = sum of ALL d eigenvalues (not just
    // the top-k_max we extracted). Otherwise cum_var is normalised to
    // the wrong denominator and k is under-selected on high-D datasets.
    let mut cum = 0.0f64;
    let mut k = K_MAX;
    for (i, &ev) in eigenvals.iter().enumerate() {
        cum += ev / total_var;
        if cum >= VAR_THRESHOLD {
            k = (i + 1).max(2).min(K_MAX);
            break;
        }
    }
    let cum_at_k: f64 = eigenvals[..k].iter().sum::<f64>() / total_var;
    eprintln!(
        "[reorder/pca] adaptive k={}/{} cum_var={:.3} (threshold={:.2}, trace={:.3e})",
        k, K_MAX, cum_at_k, VAR_THRESHOLD, total_var
    );
    let bits_per_axis: u32 = ((64 / k).min(32)) as u32;
    let coords_grid = normalize_to_grid_kd(&coords_per_pc[..k], bits_per_axis);
    let morton: Vec<u64> = (0..nlist)
        .map(|i| {
            let pt: Vec<u64> = coords_grid.iter().map(|c| c[i] as u64).collect();
            morton_nd(&pt, bits_per_axis)
        })
        .collect();
    let mut perm: Vec<usize> = (0..nlist).collect();
    perm.sort_by_key(|&i| morton[i]);
    Ok(perm)
}

/// Apply a permutation to a centroid buffer (row-major, nlist × dim).
/// Produces a new Vec<T> where `out[i*dim..(i+1)*dim]` = input's `perm[i]`-th
/// row.
pub fn apply_permutation<T: Copy>(centroids: &[T], perm: &[usize], dim: usize) -> Vec<T> {
    let n = perm.len();
    let mut out = Vec::with_capacity(n * dim);
    for &src in perm {
        let row = &centroids[src * dim..(src + 1) * dim];
        out.extend_from_slice(row);
    }
    out
}

// ---------------------------------------------------------------------------
// k-D PCA via successive deflated power iteration. Returns:
//   - Vec<Vec<f32>> of length k: coords_per_pc[j][i] is centroid i's j-th PC
//   - Vec<f64> of length k: eigenvalues (used to pick adaptive k via cum-var)
// ---------------------------------------------------------------------------
fn project_pca_kd(
    centroids: &[f32],
    n: usize,
    d: usize,
    k_max: usize,
) -> (Vec<Vec<f32>>, Vec<f64>, f64) {
    let k = k_max.min(d);
    // Mean-center in f64 for numerical stability.
    let mut means = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            means[j] += centroids[i * d + j] as f64;
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in means.iter_mut() {
        *m *= inv_n;
    }

    let mut centered = vec![0.0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            centered[i * d + j] = centroids[i * d + j] as f64 - means[j];
        }
    }

    // Covariance (d × d): cov = centered^T @ centered. Skipping the
    // (n-1) divisor — eigenvectors are scale-invariant and the
    // eigenvalues we report are likewise on a relative-variance basis.
    let mut cov = vec![0.0f64; d * d];
    for i in 0..n {
        let row_off = i * d;
        for j1 in 0..d {
            let x = centered[row_off + j1];
            if x == 0.0 {
                continue;
            }
            for j2 in 0..d {
                cov[j1 * d + j2] += x * centered[row_off + j2];
            }
        }
    }

    // trace(cov) = sum of diagonals = sum of ALL d eigenvalues.
    // This is the denominator for the cum-variance threshold; the
    // top-k_max eigenvalues we extract below sum to less than this.
    let total_var: f64 = (0..d).map(|i| cov[i * d + i]).sum();

    let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut eigenvals: Vec<f64> = Vec::with_capacity(k);
    for _ in 0..k {
        let prev_refs: Vec<&Vec<f64>> = eigenvecs.iter().collect();
        let v = power_iteration(&cov, d, 100, &prev_refs);
        // Rayleigh quotient: λ = v^T cov v (v already unit-norm).
        let mut lambda = 0.0f64;
        for i in 0..d {
            let mut row_dot = 0.0f64;
            let row_off = i * d;
            for j in 0..d {
                row_dot += cov[row_off + j] * v[j];
            }
            lambda += v[i] * row_dot;
        }
        eigenvals.push(lambda);
        eigenvecs.push(v);
    }

    let mut coords_per_pc: Vec<Vec<f32>> = Vec::with_capacity(k);
    for v in eigenvecs.iter() {
        let mut proj = Vec::with_capacity(n);
        for i in 0..n {
            let row = &centered[i * d..(i + 1) * d];
            let p: f64 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            proj.push(p as f32);
        }
        coords_per_pc.push(proj);
    }
    (coords_per_pc, eigenvals, total_var)
}

/// Power iteration to extract a dominant eigenvector of symmetric `cov` of
/// size d × d. `deflate_against` lists previously-found eigenvectors to
/// project out from each iterate (Gram–Schmidt deflation) so that successive
/// calls yield the second, third, ... largest eigenvectors.
fn power_iteration(cov: &[f64], d: usize, iters: usize, deflate_against: &[&Vec<f64>]) -> Vec<f64> {
    // Deterministic starting vector: normalised 1/sqrt(i+1) pattern.
    let mut v: Vec<f64> = (0..d).map(|i| 1.0 / ((i + 1) as f64).sqrt()).collect();
    let norm0: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm0 > 0.0 {
        for x in v.iter_mut() {
            *x /= norm0;
        }
    }
    for prev in deflate_against {
        deflate(&mut v, prev);
    }
    renormalise(&mut v);

    for _ in 0..iters {
        // new = cov @ v
        let mut new_v = vec![0.0f64; d];
        for i in 0..d {
            let mut acc = 0.0;
            let row_off = i * d;
            for j in 0..d {
                acc += cov[row_off + j] * v[j];
            }
            new_v[i] = acc;
        }
        for prev in deflate_against {
            deflate(&mut new_v, prev);
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for x in new_v.iter_mut() {
            *x /= norm;
        }
        // Optional early-stop: compare with previous v.
        let mut diff = 0.0f64;
        for (a, b) in new_v.iter().zip(v.iter()) {
            diff = diff.max((a - b).abs());
        }
        v = new_v;
        if diff < 1e-9 {
            break;
        }
    }
    v
}

fn deflate(v: &mut [f64], against: &[f64]) {
    let dot: f64 = v.iter().zip(against.iter()).map(|(a, b)| a * b).sum();
    for (a, b) in v.iter_mut().zip(against.iter()) {
        *a -= dot * b;
    }
}

fn renormalise(v: &mut [f64]) {
    let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ---------------------------------------------------------------------------
// k-D coords → integer grid (bits_per_axis bits per axis). Returns
// k vectors of length n, each with values in [0, 2^bits_per_axis).
// ---------------------------------------------------------------------------
fn normalize_to_grid_kd(coords_per_pc: &[Vec<f32>], bits_per_axis: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::with_capacity(coords_per_pc.len());
    let max_val: u64 = (1u64 << bits_per_axis) - 1;
    for coords in coords_per_pc {
        if coords.is_empty() {
            out.push(Vec::new());
            continue;
        }
        let xmin = coords.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let xmax = coords.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let dx = (xmax - xmin).max(1e-12);
        let scale = max_val as f32;
        let v: Vec<u32> = coords
            .iter()
            .map(|&x| (((x - xmin) / dx) * scale).clamp(0.0, max_val as f32) as u32)
            .collect();
        out.push(v);
    }
    out
}

// ---------------------------------------------------------------------------
// n-D Morton (Z-order) curve: bit-interleave each axis. The result fits
// in u64 as long as bits_per_axis × n_axes ≤ 64.
//
// Using Morton instead of Hilbert in n-D: Hilbert has slightly better
// locality (~10-20% theoretical) but its n-D generalisation (Skilling
// 1991) is non-trivial; Morton is dead simple and the variance gain
// from going 2-D → 4-D PCA dominates the curve-choice difference.
// ---------------------------------------------------------------------------
fn morton_nd(coords: &[u64], bits_per_axis: u32) -> u64 {
    let k = coords.len() as u64;
    let mut result = 0u64;
    for bit in 0..bits_per_axis as u64 {
        for d in 0..k {
            let axis_bit = (coords[d as usize] >> bit) & 1;
            result |= axis_bit << (bit * k + d);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_nd_distinct_on_grid() {
        // Morton (Z-order) curve should map distinct grid points to
        // distinct indices.
        let bits = 16;
        assert_ne!(morton_nd(&[0, 0], bits), morton_nd(&[0, 1], bits));
        assert_ne!(morton_nd(&[0, 0], bits), morton_nd(&[1, 0], bits));
        // 4-D test
        let bits4 = 8;
        assert_ne!(
            morton_nd(&[0, 0, 0, 0], bits4),
            morton_nd(&[0, 0, 0, 1], bits4)
        );
    }

    #[test]
    fn permutation_length_matches_nlist() {
        let d = 4;
        let n = 16;
        let centroids: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let perm = hilbert_permutation_f32(&centroids, n, d).unwrap();
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn apply_permutation_roundtrip() {
        let d = 3;
        let cen: Vec<f32> = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let perm = vec![2, 0, 1];
        let out = apply_permutation(&cen, &perm, d);
        assert_eq!(out, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
