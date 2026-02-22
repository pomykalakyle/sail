use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::MemTable;
use datafusion::catalog::TableProvider;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion_common::Constraints;
use datafusion_expr::ScalarUDF;
use log::warn;
use sail_common::spec;
use sail_common_datafusion::cache::create_ipc_file_table_provider;
use sail_common_datafusion::catalog::{TableKind, TableStatus};
use sail_common_datafusion::datasource::{SourceInfo, TableFormatRegistry};
use sail_common_datafusion::extension::SessionExtensionAccessor;
use sail_common_datafusion::session::PlanService;

use crate::error::{CatalogError, CatalogResult};
use crate::manager::CatalogManager;
use crate::provider::{
    CreateDatabaseOptions, CreateTableOptions, CreateTemporaryViewOptions, CreateViewOptions,
    DropDatabaseOptions, DropTableOptions, DropTemporaryViewOptions, DropViewOptions,
};
use crate::utils::quote_namespace_if_needed;

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Hash)]
pub enum CatalogCommand {
    CurrentCatalog,
    SetCurrentCatalog {
        catalog: String,
    },
    ListCatalogs {
        pattern: Option<String>,
    },
    CurrentDatabase,
    SetCurrentDatabase {
        database: Vec<String>,
    },
    CreateDatabase {
        database: Vec<String>,
        options: CreateDatabaseOptions,
    },
    DatabaseExists {
        database: Vec<String>,
    },
    GetDatabase {
        database: Vec<String>,
    },
    ListDatabases {
        qualifier: Vec<String>,
        pattern: Option<String>,
    },
    DropDatabase {
        database: Vec<String>,
        options: DropDatabaseOptions,
    },
    CreateTable {
        table: Vec<String>,
        options: CreateTableOptions,
    },
    TableExists {
        table: Vec<String>,
    },
    GetTable {
        table: Vec<String>,
    },
    ListTables {
        database: Vec<String>,
        pattern: Option<String>,
    },
    ListViews {
        database: Vec<String>,
        pattern: Option<String>,
    },
    DropTable {
        table: Vec<String>,
        options: DropTableOptions,
    },
    IsCached {
        table: Vec<String>,
    },
    CacheTable {
        table: Vec<String>,
        lazy: bool,
        storage_level: Option<CatalogStorageLevel>,
    },
    UncacheTable {
        table: Vec<String>,
        if_exists: bool,
    },
    ClearCache,
    RefreshTable {
        table: Vec<String>,
    },
    ListColumns {
        table: Vec<String>,
    },
    FunctionExists {
        function: Vec<String>,
    },
    GetFunction {
        function: Vec<String>,
    },
    ListFunctions {
        database: Vec<String>,
        pattern: Option<String>,
    },
    DropFunction {
        function: Vec<String>,
        if_exists: bool,
        is_temporary: bool,
    },
    RegisterFunction {
        udf: ScalarUDF,
    },
    #[allow(unused)]
    RegisterTableFunction {
        name: String,
        // We have to be explicit about the UDTF types we support.
        // We cannot use `Arc<dyn TableFunctionImpl>` because it does not implement `Eq` and `Hash`.
        udtf: CatalogTableFunction,
    },
    DropTemporaryView {
        view: String,
        is_global: bool,
        options: DropTemporaryViewOptions,
    },
    DropView {
        view: Vec<String>,
        options: DropViewOptions,
    },
    CreateTemporaryView {
        view: String,
        is_global: bool,
        options: CreateTemporaryViewOptions,
    },
    CreateView {
        view: Vec<String>,
        options: CreateViewOptions,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd)]
pub enum CatalogTableFunction {
    // We do not support any kind of table functions yet.
    // PySpark UDTF is registered as a scalar UDF.
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Hash)]
pub struct CatalogStorageLevel {
    use_disk: bool,
    use_memory: bool,
    use_off_heap: bool,
    deserialized: bool,
    replication: usize,
}

impl From<spec::StorageLevel> for CatalogStorageLevel {
    fn from(level: spec::StorageLevel) -> Self {
        Self {
            use_disk: level.use_disk,
            use_memory: level.use_memory,
            use_off_heap: level.use_off_heap,
            deserialized: level.deserialized,
            replication: level.replication,
        }
    }
}
fn cached_table_relation_id(cache_key: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cache_key.hash(&mut hasher);
    format!("__sail_cached_table_{:016x}", hasher.finish())
}

fn validate_table_cache_storage_level(level: Option<&CatalogStorageLevel>) -> CatalogResult<()> {
    let Some(level) = level else {
        return Ok(());
    };
    if !level.use_disk && !level.use_memory {
        return Err(CatalogError::InvalidArgument(
            "CACHE TABLE storage level must use disk and/or memory".to_string(),
        ));
    }
    if level.use_off_heap {
        warn!(
            "CACHE TABLE requested OFF_HEAP storage; Sail does not support off-heap cache and ignores this flag"
        );
    }
    if level.replication > 1 {
        warn!(
            "CACHE TABLE requested replication factor {}; replication is not supported",
            level.replication
        );
    }
    if level.use_disk && level.use_memory {
        warn!(
            "CACHE TABLE MEMORY_AND_DISK requested; Sail does not implement Spark's memory-first spill policy and uses a disk-backed cache implementation"
        );
    }
    Ok(())
}

fn is_disk_backed_storage_level(level: Option<&CatalogStorageLevel>) -> bool {
    matches!(level, Some(level) if level.use_disk)
}
async fn materialize_table_status_batches(
    ctx: &SessionContext,
    status: TableStatus,
) -> CatalogResult<(SchemaRef, Vec<RecordBatch>)> {
    let status_name = status.name;
    let status_database = status.kind.database();
    match status.kind {
        TableKind::Table {
            catalog: _,
            database: _,
            columns,
            comment: _,
            constraints: _,
            location,
            format,
            partition_by,
            sort_by,
            bucket_by,
            options,
            properties: _,
        } => {
            let schema = Schema::new(columns.iter().map(|x| x.field()).collect::<Vec<_>>());
            let info = SourceInfo {
                paths: location.map(|path| vec![path]).unwrap_or_default(),
                schema: Some(schema),
                constraints: Constraints::default(),
                partition_by,
                bucket_by: bucket_by.map(|value| value.into()),
                sort_order: sort_by.into_iter().map(|value| value.into()).collect(),
                options: vec![options.into_iter().collect()],
            };
            let registry = ctx.extension::<TableFormatRegistry>()?;
            let table_format = registry.get(&format).map_err(|e| {
                CatalogError::Internal(format!(
                    "cache table format lookup failed for {}.{} using format {format}: {e}",
                    status_database.join("."),
                    status_name
                ))
            })?;
            let table_provider = table_format
                .create_provider(&ctx.state(), info)
                .await
                .map_err(|e| {
                    CatalogError::Internal(format!(
                        "cache table provider creation failed for {}.{} using format {format}: {e}",
                        status_database.join("."),
                        status_name
                    ))
                })?;
            let filters = vec![];
            let physical = table_provider
                .scan(&ctx.state(), None, filters.as_slice(), None)
                .await
                .map_err(|e| {
                    CatalogError::Internal(format!(
                        "cache table scan planning failed for {}.{}: {e}",
                        status_database.join("."),
                        status_name
                    ))
                })?;
            let schema = physical.schema();
            let batches = collect(physical, ctx.task_ctx()).await.map_err(|e| {
                CatalogError::Internal(format!(
                    "cache table execution failed for {}.{}: {e}",
                    status_database.join("."),
                    status_name
                ))
            })?;
            Ok((schema, batches))
        }
        TableKind::TemporaryView { plan, .. } | TableKind::GlobalTemporaryView { plan, .. } => {
            let physical = ctx
                .state()
                .create_physical_plan(plan.as_ref())
                .await
                .map_err(|e| {
                    CatalogError::Internal(format!(
                        "cache view planning failed for {}.{}: {e}",
                        status_database.join("."),
                        status_name
                    ))
                })?;
            let schema = physical.schema();
            let batches = collect(physical, ctx.task_ctx()).await.map_err(|e| {
                CatalogError::Internal(format!(
                    "cache view execution failed for {}.{}: {e}",
                    status_database.join("."),
                    status_name
                  ))
            })?;
            Ok((schema, batches))
        }
        TableKind::View { .. } => Err(CatalogError::NotSupported("cache view".to_string())),
    }
}

async fn materialize_table_cache(
    ctx: &SessionContext,
    manager: &CatalogManager,
    table: &[String],
    disk_backed: bool,
) -> CatalogResult<(String, Arc<dyn TableProvider>)> {
    let key = manager.cache_key_for_table(table).await?;
    let relation_id = cached_table_relation_id(&key);
    let status = manager.get_table_or_view(table).await?;
    let (schema, batches) = materialize_table_status_batches(ctx, status).await?;
    let provider: Arc<dyn TableProvider> = if disk_backed {
        create_ipc_file_table_provider(relation_id.as_str(), schema, batches.as_slice())?
    } else {
        Arc::new(MemTable::try_new(schema, vec![batches])?)
    };
    Ok((relation_id, provider))
}

impl CatalogCommand {
    pub fn name(&self) -> &str {
        match self {
            CatalogCommand::CurrentCatalog => "CurrentCatalog",
            CatalogCommand::SetCurrentCatalog { .. } => "SetCurrentCatalog",
            CatalogCommand::ListCatalogs { .. } => "ListCatalogs",
            CatalogCommand::CurrentDatabase => "CurrentDatabase",
            CatalogCommand::SetCurrentDatabase { .. } => "SetCurrentDatabase",
            CatalogCommand::CreateDatabase { .. } => "CreateDatabase",
            CatalogCommand::DatabaseExists { .. } => "DatabaseExists",
            CatalogCommand::GetDatabase { .. } => "GetDatabase",
            CatalogCommand::ListDatabases { .. } => "ListDatabases",
            CatalogCommand::DropDatabase { .. } => "DropDatabase",
            CatalogCommand::CreateTable { .. } => "CreateTable",
            CatalogCommand::TableExists { .. } => "TableExists",
            CatalogCommand::GetTable { .. } => "GetTable",
            CatalogCommand::ListTables { .. } => "ListTables",
            CatalogCommand::ListViews { .. } => "ListViews",
            CatalogCommand::DropTable { .. } => "DropTable",
            CatalogCommand::IsCached { .. } => "IsCached",
            CatalogCommand::CacheTable { .. } => "CacheTable",
            CatalogCommand::UncacheTable { .. } => "UncacheTable",
            CatalogCommand::ClearCache => "ClearCache",
            CatalogCommand::RefreshTable { .. } => "RefreshTable",
            CatalogCommand::ListColumns { .. } => "ListColumns",
            CatalogCommand::FunctionExists { .. } => "FunctionExists",
            CatalogCommand::GetFunction { .. } => "GetFunction",
            CatalogCommand::ListFunctions { .. } => "ListFunctions",
            CatalogCommand::RegisterFunction { .. } => "RegisterFunction",
            CatalogCommand::RegisterTableFunction { .. } => "RegisterTableFunction",
            CatalogCommand::DropFunction { .. } => "DropFunction",
            CatalogCommand::DropTemporaryView { .. } => "DropTemporaryView",
            CatalogCommand::DropView { .. } => "DropView",
            CatalogCommand::CreateTemporaryView { .. } => "CreateTemporaryView",
            CatalogCommand::CreateView { .. } => "CreateView",
        }
    }

    pub fn schema(&self, ctx: &SessionContext) -> CatalogResult<SchemaRef> {
        let service = ctx.extension::<PlanService>()?;
        let display = service.catalog_display();
        let schema = match self {
            CatalogCommand::ListCatalogs { .. } => display.catalogs().schema()?,
            CatalogCommand::GetDatabase { .. } | CatalogCommand::ListDatabases { .. } => {
                display.databases().schema()?
            }
            CatalogCommand::GetTable { .. }
            | CatalogCommand::ListTables { .. }
            | CatalogCommand::ListViews { .. } => display.tables().schema()?,
            CatalogCommand::ListColumns { .. } => display.table_columns().schema()?,
            CatalogCommand::GetFunction { .. } | CatalogCommand::ListFunctions { .. } => {
                display.functions().schema()?
            }
            CatalogCommand::SetCurrentCatalog { .. }
            | CatalogCommand::SetCurrentDatabase { .. }
            | CatalogCommand::RegisterFunction { .. }
            | CatalogCommand::RegisterTableFunction { .. }
            | CatalogCommand::CacheTable { .. }
            | CatalogCommand::UncacheTable { .. }
            | CatalogCommand::RefreshTable { .. }
            | CatalogCommand::ClearCache => display.empty().schema()?,
            CatalogCommand::CurrentCatalog | CatalogCommand::CurrentDatabase => {
                display.strings().schema()?
            }
            CatalogCommand::DatabaseExists { .. }
            | CatalogCommand::TableExists { .. }
            | CatalogCommand::IsCached { .. }
            | CatalogCommand::FunctionExists { .. }
            | CatalogCommand::CreateDatabase { .. }
            | CatalogCommand::CreateTable { .. }
            | CatalogCommand::CreateTemporaryView { .. }
            | CatalogCommand::CreateView { .. }
            | CatalogCommand::DropDatabase { .. }
            | CatalogCommand::DropTable { .. }
            | CatalogCommand::DropFunction { .. }
            | CatalogCommand::DropTemporaryView { .. }
            | CatalogCommand::DropView { .. } => display.bools().schema()?,
        };
        Ok(schema)
    }

    pub async fn execute(
        self,
        ctx: &SessionContext,
        manager: &CatalogManager,
    ) -> CatalogResult<RecordBatch> {
        // TODO: make sure we return the same schema as Spark for each command
        let service = ctx.extension::<PlanService>()?;
        let display = service.catalog_display();
        let batch = match self {
            CatalogCommand::CurrentCatalog => {
                let value = manager.default_catalog()?;
                display.strings().to_record_batch(vec![value.to_string()])?
            }
            CatalogCommand::SetCurrentCatalog { catalog } => {
                manager.set_default_catalog(catalog)?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::ListCatalogs { pattern } => {
                let rows = manager
                    .list_catalogs(pattern.as_deref())?
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>();
                display.catalogs().to_record_batch(rows)?
            }
            CatalogCommand::CurrentDatabase => {
                let value = manager.default_database()?;
                let value = quote_namespace_if_needed(&value);
                display.strings().to_record_batch(vec![value])?
            }
            CatalogCommand::SetCurrentDatabase { database } => {
                manager.set_default_database(database).await?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::CreateDatabase { database, options } => {
                manager.create_database(&database, options).await?;
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::DatabaseExists { database } => {
                let value = match manager.get_database(&database).await {
                    Ok(_) => true,
                    Err(CatalogError::NotFound(_, _)) => false,
                    Err(e) => return Err(e),
                };
                display.bools().to_record_batch(vec![value])?
            }
            CatalogCommand::GetDatabase { database } => {
                let rows = match manager.get_database(&database).await {
                    Ok(x) => vec![x],
                    Err(CatalogError::NotFound(_, _)) => vec![],
                    Err(e) => return Err(e),
                };
                display.databases().to_record_batch(rows)?
            }
            CatalogCommand::ListDatabases { qualifier, pattern } => {
                let rows = match manager.list_databases(&qualifier, pattern.as_deref()).await {
                    Ok(rows) => rows,
                    Err(CatalogError::NotFound(_, _)) => vec![],
                    Err(e) => return Err(e),
                };
                display.databases().to_record_batch(rows)?
            }
            CatalogCommand::DropDatabase { database, options } => {
                manager.drop_database(&database, options).await?;
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::CreateTable { table, options } => {
                manager.create_table(&table, options).await?;
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::TableExists { table } => {
                let value = match manager.get_table_or_view(&table).await {
                    Ok(_) => true,
                    Err(CatalogError::NotFound(_, _)) => false,
                    Err(e) => return Err(e),
                };
                display.bools().to_record_batch(vec![value])?
            }
            CatalogCommand::GetTable { table } => {
                // We are supposed to return an error if the table or view does not exist.
                let table = manager.get_table_or_view(&table).await?;
                display.tables().to_record_batch(vec![table])?
            }
            CatalogCommand::ListTables { database, pattern } => {
                let rows = manager
                    .list_tables_and_temporary_views(&database, pattern.as_deref())
                    .await?;
                display.tables().to_record_batch(rows)?
            }
            CatalogCommand::ListViews { database, pattern } => {
                let rows = manager
                    .list_views_and_temporary_views(&database, pattern.as_deref())
                    .await?;
                display.tables().to_record_batch(rows)?
            }
            CatalogCommand::DropTable { table, options } => {
                manager.drop_table(&table, options).await?;
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::IsCached { table } => {
                let value = manager.is_cached_table(&table).await?;
                display.bools().to_record_batch(vec![value])?
            }
            CatalogCommand::CacheTable {
                table,
                lazy,
                storage_level,
            } => {
                validate_table_cache_storage_level(storage_level.as_ref())?;
                let disk_backed = is_disk_backed_storage_level(storage_level.as_ref());
                manager.cache_table(&table, lazy, disk_backed).await?;
                if !lazy {
                    let (relation_id, provider) =
                        match materialize_table_cache(ctx, manager, &table, disk_backed).await {
                            Ok(result) => result,
                            Err(error) => {
                                let _ = manager.uncache_table(&table, true).await;
                                return Err(error);
                            }
                        };
                    manager
                        .set_cached_table_relation(&table, relation_id.clone())
                        .await?;
                    manager.set_cached_table_provider(relation_id, provider)?;
                }
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::UncacheTable { table, if_exists } => {
                let _ = manager.uncache_table(&table, if_exists).await?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::ClearCache => {
                let _ = manager.clear_cached_tables()?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::RefreshTable { table } => {
                let _ = manager.refresh_table(&table).await?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::ListColumns { table } => {
                let rows = manager.get_table_or_view(&table).await?.kind.columns();
                display.table_columns().to_record_batch(rows)?
            }
            CatalogCommand::FunctionExists { .. } => {
                return Err(CatalogError::NotSupported("function exists".to_string()));
            }
            CatalogCommand::GetFunction { .. } => {
                return Err(CatalogError::NotSupported("get function".to_string()));
            }
            CatalogCommand::ListFunctions { .. } => {
                return Err(CatalogError::NotSupported("list functions".to_string()));
            }
            // TODO: `ctx` will not be needed if `CatalogManager` manages functions internally.
            CatalogCommand::DropFunction {
                function,
                if_exists,
                is_temporary,
            } => {
                manager
                    .deregister_function(ctx, &function, if_exists, is_temporary)
                    .await?;
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::RegisterFunction { udf } => {
                manager.register_function(ctx, udf)?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::RegisterTableFunction { name, udtf } => {
                manager.register_table_function(ctx, name, udtf)?;
                display.empty().to_record_batch(vec![])?
            }
            CatalogCommand::DropTemporaryView {
                view,
                is_global,
                options,
            } => {
                if is_global {
                    manager.drop_global_temporary_view(&view, options).await?;
                } else {
                    manager.drop_temporary_view(&view, options).await?;
                }
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::DropView { view, options } => {
                manager.drop_maybe_temporary_view(&view, options).await?;
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::CreateTemporaryView {
                view,
                is_global,
                options,
            } => {
                if is_global {
                    manager.create_global_temporary_view(&view, options).await?;
                } else {
                    manager.create_temporary_view(&view, options).await?;
                }
                display.bools().to_record_batch(vec![true])?
            }
            CatalogCommand::CreateView { view, options } => {
                manager.create_view(&view, options).await?;
                display.bools().to_record_batch(vec![true])?
            }
        };
        Ok(batch)
    }
}
