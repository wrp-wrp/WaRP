// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use crate::vector::quantizer::QuantizerStorage;
use arrow::compute::concat_batches;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use deepsize::DeepSizeOf;
use futures::prelude::stream::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result, ROW_ID};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_linalg::distance::DistanceType;
use prost::Message;
use snafu::location;
use std::{any::Any, sync::Arc};

use crate::frag_reuse::FragReuseIndex;
use crate::{
    pb,
    vector::{
        ivf::storage::{IvfModel, IVF_METADATA_KEY},
        quantizer::Quantization,
    },
};

use super::quantizer::{Quantizer, QuantizerMetadata};
use super::DISTANCE_TYPE_KEY;

/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;

    // return the distances of all rows
    // k_hint is a hint that can be used for optimization
    fn distance_all(&self, k_hint: usize) -> Vec<f32>;

    fn prefetch(&self, _id: u32) {}
}

pub const STORAGE_METADATA_KEY: &str = "storage_metadata";

fn prepare_partition_storage_batch(
    batch: RecordBatch,
    compact_for_cache: bool,
) -> Result<RecordBatch> {
    if compact_for_cache {
        Ok(batch.shrink_to_fit()?)
    } else {
        Ok(batch)
    }
}

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory or on-disk, raw vector or quantized vectors.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
///
/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait VectorStore: Send + Sync + Sized + Clone {
    type DistanceCalculator<'a>: DistCalculator
    where
        Self: 'a;

    fn as_any(&self) -> &dyn Any;

    fn schema(&self) -> &SchemaRef;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send>;

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return [DistanceType].
    fn distance_type(&self) -> DistanceType;

    /// Get the lance ROW ID from one vector.
    fn row_id(&self, id: u32) -> u64;

    fn row_ids(&self) -> impl Iterator<Item = &u64>;

    /// Append Raw [RecordBatch] into the Storage.
    /// The storage implement will perform quantization if necessary.
    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self>;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calculator can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef, dist_q_c: f32) -> Self::DistanceCalculator<'_>;

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_>;

    fn dist_between(&self, u: u32, v: u32) -> f32 {
        let dist_cal_u = self.dist_calculator_from_id(u);
        dist_cal_u.distance(v)
    }
}

pub struct StorageBuilder<Q: Quantization> {
    vector_column: String,
    distance_type: DistanceType,
    quantizer: Q,

    frag_reuse_index: Option<Arc<FragReuseIndex>>,
}

impl<Q: Quantization> StorageBuilder<Q> {
    pub fn new(
        vector_column: String,
        distance_type: DistanceType,
        quantizer: Q,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        Ok(Self {
            vector_column,
            distance_type,
            quantizer,
            frag_reuse_index,
        })
    }

    pub fn build(&self, batches: Vec<RecordBatch>) -> Result<Q::Storage> {
        let mut batch = concat_batches(batches[0].schema_ref(), batches.iter())?;

        if batch.column_by_name(self.quantizer.column()).is_none() {
            let vectors = batch
                .column_by_name(&self.vector_column)
                .ok_or(Error::Index {
                    message: format!("Vector column {} not found in batch", self.vector_column),
                    location: location!(),
                })?;
            let codes = self.quantizer.quantize(vectors)?;
            batch = batch.drop_column(&self.vector_column)?.try_with_column(
                arrow_schema::Field::new(self.quantizer.column(), codes.data_type().clone(), true),
                codes,
            )?;
        }

        debug_assert!(batch.column_by_name(ROW_ID).is_some());
        debug_assert!(batch.column_by_name(self.quantizer.column()).is_some());

        Q::Storage::try_from_batch(
            batch,
            &self.quantizer.metadata(None),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    metadata: Q::Metadata,

    ivf: IvfModel,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.ivf.deep_size_of_children(context)
    }
}

impl<Q: Quantization> IvfQuantizationStorage<Q> {
    /// Open a Loader.
    ///
    ///
    pub async fn try_new(
        reader: FileReader,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let schema = reader.schema();

        let distance_type = DistanceType::try_from(
            schema
                .metadata
                .get(DISTANCE_TYPE_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", DISTANCE_TYPE_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;

        let ivf_pos = schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!("{} not found", IVF_METADATA_KEY),
                location: location!(),
            })?
            .parse()
            .map_err(|e| Error::Index {
                message: format!("Failed to decode IVF metadata: {}", e),
                location: location!(),
            })?;
        let ivf_bytes = reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_bytes)?)?;

        let mut metadata: Vec<String> = serde_json::from_str(
            schema
                .metadata
                .get(STORAGE_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", STORAGE_METADATA_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;
        debug_assert_eq!(metadata.len(), 1);
        // for now the metadata is the same for all partitions, so we just store one
        let metadata = metadata.pop().ok_or(Error::Index {
            message: "metadata is empty".to_string(),
            location: location!(),
        })?;
        let mut metadata: Q::Metadata = serde_json::from_str(&metadata)?;
        // we store large metadata (e.g. PQ codebook) in global buffer,
        // and the schema metadata just contains a pointer to the buffer
        if let Some(pos) = metadata.buffer_index() {
            let bytes = reader.read_global_buffer(pos).await?;
            metadata.parse_buffer(bytes)?;
        }

        Ok(Self {
            reader,
            distance_type,
            metadata,
            ivf,
            frag_reuse_index,
        })
    }

    /// Replaces the reader used after storage metadata has been initialized.
    #[doc(hidden)]
    pub fn set_payload_reader(&mut self, reader: FileReader) {
        self.reader = reader;
    }

    pub fn num_rows(&self) -> u64 {
        self.reader.num_rows()
    }

    pub fn partition_size(&self, part_id: usize) -> usize {
        self.ivf.partition_size(part_id)
    }

    pub fn quantizer(&self) -> Result<Quantizer> {
        let metadata = self.metadata();
        Q::from_metadata(metadata, self.distance_type)
    }

    pub fn metadata(&self) -> &Q::Metadata {
        &self.metadata
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.reader.schema().as_ref().into())
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    pub async fn load_partition(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let batch = if range.is_empty() {
            let schema = self.reader.schema();
            let arrow_schema = arrow_schema::Schema::from(schema.as_ref());
            RecordBatch::new_empty(Arc::new(arrow_schema))
        } else {
            let batches = self
                .reader
                .read_stream(
                    ReadBatchParams::Range(range),
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )?
                .try_collect::<Vec<_>>()
                .await?;
            let schema = Arc::new(self.reader.schema().as_ref().into());
            concat_batches(&schema, batches.iter())?
        };
        Q::Storage::try_from_batch(
            batch,
            self.metadata(),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }

    /// Batch-load multiple partitions in ONE `read_stream(Ranges(...))` call.
    ///
    /// Purpose (paper-faithful IVF scan integration): instead of having each
    /// partition issue its own `read_stream(Range(single))` call — which
    /// decomposes to a single-range `FileScheduler::submit_request` the
    /// merge planner cannot act on — we submit every requested partition's
    /// row range in one shot. Lance's v2 encoding groups them by page, so
    /// the FileScheduler sees multi-range submits the merge planner can
    /// meaningfully consolidate.
    ///
    /// The output `Vec<Q::Storage>` matches the order of `part_ids`. Empty
    /// row ranges produce empty storages in place. When `compact_for_cache`
    /// is true, each partition receives compact Arrow buffers so a persistent
    /// cache entry does not retain and charge the whole combined read batch.
    pub async fn load_partitions_batch(
        &self,
        part_ids: &[u32],
        compact_for_cache: bool,
    ) -> Result<Vec<Q::Storage>> {
        let schema: SchemaRef = Arc::new(self.reader.schema().as_ref().into());
        if part_ids.is_empty() {
            return Ok(Vec::new());
        }

        // `read_stream(Ranges([...]))` emits rows in *file-offset* order,
        // independent of submission order. Since IVF probe IDs come from
        // `find_partitions` (distance order), not offset order, we must
        // sort-by-offset before submitting and apply an inverse permutation
        // when distributing results back to caller-requested order.
        let n = part_ids.len();
        let mut indexed: Vec<(usize, std::ops::Range<usize>, usize)> = Vec::with_capacity(n);
        for (i, &pid) in part_ids.iter().enumerate() {
            let r = self.ivf.row_range(pid as usize);
            let len = r.len();
            indexed.push((i, r, len));
        }
        indexed.sort_by_key(|(_, r, _)| r.start);

        // Coalesce adjacent row ranges before handing them to the
        // reader. After Hilbert centroid reorder, feature-space-close
        // partitions land at consecutive partition_ids, so their row
        // ranges are end-to-end (zero-gap) — fusing them collapses
        // many small row spans into a few large ones, which avoids
        // the per-spanned-page byte-range splits that would otherwise
        // bloat the scheduler's input list.
        let mut submit_ranges: Vec<std::ops::Range<u64>> = Vec::with_capacity(n);
        for (_, r, _) in indexed.iter() {
            if r.is_empty() {
                continue;
            }
            let cur = r.start as u64..r.end as u64;
            match submit_ranges.last_mut() {
                Some(last) if last.end == cur.start => last.end = cur.end,
                _ => submit_ranges.push(cur),
            }
        }

        let combined_batch = if submit_ranges.is_empty() {
            RecordBatch::new_empty(schema.clone())
        } else {
            let batches = self
                .reader
                .read_stream(
                    ReadBatchParams::Ranges(submit_ranges.into()),
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )?
                .try_collect::<Vec<_>>()
                .await?;
            concat_batches(&schema, batches.iter())?
        };

        // Walk the concatenated batch in sorted (file-offset) order and
        // place each partition back at its original position in `part_ids`.
        // Persistent cache entries receive compact buffers; query-local
        // entries keep zero-copy slices.
        let mut out: Vec<Option<Q::Storage>> = (0..n).map(|_| None).collect();
        let mut cursor = 0usize;
        for (orig_i, _r, len) in indexed.into_iter() {
            let pid = part_ids[orig_i];
            let batch = if len == 0 {
                RecordBatch::new_empty(schema.clone())
            } else {
                prepare_partition_storage_batch(
                    combined_batch.slice(cursor, len),
                    compact_for_cache,
                )?
            };
            cursor += len;
            let storage = Q::Storage::try_from_batch(
                batch,
                self.metadata(),
                self.distance_type,
                self.frag_reuse_index.clone(),
            )
            .map_err(|e| Error::Index {
                message: format!("failed to build storage for partition {}: {}", pid, e),
                location: location!(),
            })?;
            out[orig_i] = Some(storage);
        }
        Ok(out
            .into_iter()
            .map(|x| x.expect("all slots filled"))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
    use lance_arrow::FixedSizeListArrayExt;

    use super::*;

    #[test]
    fn test_prepare_partition_storage_batch_releases_combined_buffer() {
        const ROWS: usize = 4096;
        const DIMENSION: usize = 128;
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..ROWS as u64));
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from_iter_values((0..ROWS * DIMENSION).map(|value| value as f32)),
                DIMENSION as i32,
            )
            .unwrap(),
        );
        let combined = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids as ArrayRef, false),
            ("flat", vectors as ArrayRef, false),
        ])
        .unwrap();
        let sliced = combined.slice(1024, 16);
        let expected = sliced.clone();
        let retained_size = sliced.get_array_memory_size();

        let prepared = prepare_partition_storage_batch(sliced.clone(), true).unwrap();
        let prepared_size = prepared.get_array_memory_size();
        let query_local = prepare_partition_storage_batch(sliced, false).unwrap();

        assert_eq!(prepared, expected);
        assert_eq!(prepared.num_rows(), 16);
        assert!(
            prepared_size < retained_size / 8,
            "cached storage partition retained the combined batch buffer: \
             prepared={prepared_size}, combined={retained_size}"
        );
        assert_eq!(query_local.get_array_memory_size(), retained_size);
    }
}
