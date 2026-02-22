use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::FileReader;
use datafusion::arrow::ipc::writer::FileWriter;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemTable, Session, TableProvider};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{Expr, TableType};

const CACHE_DIR_NAME: &str = "sail-cache-ipc";

fn build_cache_file_path(prefix: &str) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(CACHE_DIR_NAME);
    std::fs::create_dir_all(&dir).map_err(DataFusionError::IoError)?;
    let mut hasher = DefaultHasher::new();
    prefix.hash(&mut hasher);
    let prefix_hash = hasher.finish();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| DataFusionError::External(Box::new(e)))?
        .as_nanos();
    let pid = std::process::id();
    Ok(dir.join(format!("{prefix_hash:016x}-{pid}-{nanos}.arrowipc")))
}

fn write_batches_to_ipc_file(
    path: &Path,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<()> {
    let file = File::create(path).map_err(DataFusionError::IoError)?;
    let mut writer = FileWriter::try_new(file, schema.as_ref())?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

#[derive(Debug)]
pub struct IpcFileTableProvider {
    schema: SchemaRef,
    path: PathBuf,
}

impl IpcFileTableProvider {
    pub fn new(schema: SchemaRef, path: PathBuf) -> Self {
        Self { schema, path }
    }
}

impl Drop for IpcFileTableProvider {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[async_trait::async_trait]
impl TableProvider for IpcFileTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let file = File::open(&self.path).map_err(DataFusionError::IoError)?;
        let reader = FileReader::try_new(file, None)?;
        let batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
        let table = MemTable::try_new(self.schema.clone(), vec![batches])?;
        table.scan(state, projection, filters, limit).await
    }
}

pub fn create_ipc_file_table_provider(
    prefix: &str,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<Arc<dyn TableProvider>> {
    let path = build_cache_file_path(prefix)?;
    write_batches_to_ipc_file(path.as_path(), schema.clone(), batches)?;
    Ok(Arc::new(IpcFileTableProvider::new(schema, path)))
}
