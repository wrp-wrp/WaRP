//! Cumulative sums over the input requests.
//!
//! Several strategies need fast `O(1)` answers to queries like "total
//! useful bytes covered by requests `[lo, hi)`" and "span of the merged
//! request covering `[lo, hi)`". Building prefix-sum tables once up
//! front turns those into two subtractions per query.
//!
//! This module is strategy-agnostic and does not know anything about the
//! cost model — that stays in the strategy files.

use crate::io_plan::IoRequest;

/// Precomputed cumulative statistics over a sorted slice of requests.
#[derive(Debug, Clone)]
pub struct RequestPrefixSums {
    /// `offsets[i]` = `requests[i].offset` (or 0 beyond the array; see
    /// [`RequestPrefixSums::span`]).
    offsets: Vec<i64>,
    /// `ends[i]` = `requests[i].end()`.
    ends: Vec<i64>,
}

impl RequestPrefixSums {
    /// Precompute sums. Requests must be sorted by `offset`; callers are
    /// responsible for ensuring that. The constructor is `O(n)`.
    pub fn new(requests: &[IoRequest]) -> Self {
        let n = requests.len();
        let mut offsets = Vec::with_capacity(n);
        let mut ends = Vec::with_capacity(n);
        for r in requests {
            offsets.push(r.offset);
            ends.push(r.end());
        }
        Self { offsets, ends }
    }

    /// Number of requests covered.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Span of the merged request that covers `requests[lo..hi]`:
    /// `requests[hi-1].end() - requests[lo].offset`. Returns 0 on empty
    /// range.
    #[inline]
    pub fn span(&self, lo: usize, hi: usize) -> i64 {
        if lo >= hi {
            return 0;
        }
        self.ends[hi - 1] - self.offsets[lo]
    }

    /// Start offset of the merged request covering `requests[lo..hi]`.
    ///
    /// Panics if `lo >= self.len()`. Callers should check `lo < len()`.
    #[inline]
    pub fn start_offset(&self, lo: usize) -> i64 {
        self.offsets[lo]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: 0,
            offset,
            length,
        }
    }

    #[test]
    fn prefix_sums_basic() {
        let rs = vec![r(0, 10), r(20, 5), r(100, 1)];
        let ps = RequestPrefixSums::new(&rs);
        assert_eq!(ps.span(0, 3), 101);
        assert_eq!(ps.span(1, 3), 101 - 20);
        assert_eq!(ps.start_offset(1), 20);
        assert_eq!(ps.len(), 3);
    }

    #[test]
    fn empty_range_is_zero() {
        let rs = vec![r(0, 10)];
        let ps = RequestPrefixSums::new(&rs);
        assert_eq!(ps.span(0, 0), 0);
    }
}
