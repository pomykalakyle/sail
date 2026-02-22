use datafusion::prelude::SessionContext;
use log::warn;
use sail_common::spec;
use sail_common_datafusion::extension::SessionExtensionAccessor;
use sail_common_datafusion::rename::schema::rename_schema;
use sail_plan::explain::{explain_string, ExplainOptions};
use sail_plan::resolver::plan::NamedPlan;
use sail_plan::resolver::PlanResolver;

use crate::error::{ProtoFieldExt, SparkError, SparkResult};
use crate::proto::data_type::parse_spark_data_type;
use crate::schema::{to_spark_schema, to_tree_string};
use crate::session::SparkSession;
use crate::spark::connect as sc;
use crate::spark::connect::analyze_plan_request::explain::ExplainMode;
use crate::spark::connect::analyze_plan_request::{
    DdlParse as DdlParseRequest, Explain as ExplainRequest,
    GetStorageLevel as GetStorageLevelRequest, InputFiles as InputFilesRequest,
    IsLocal as IsLocalRequest, IsStreaming as IsStreamingRequest, JsonToDdl as JsonToDdlRequest,
    Persist as PersistRequest, SameSemantics as SameSemanticsRequest, Schema as SchemaRequest,
    SemanticHash as SemanticHashRequest, SparkVersion as SparkVersionRequest,
    TreeString as TreeStringRequest, Unpersist as UnpersistRequest,
};
use crate::spark::connect::analyze_plan_response::{
    DdlParse as DdlParseResponse, Explain as ExplainResponse,
    GetStorageLevel as GetStorageLevelResponse, InputFiles as InputFilesResponse,
    IsLocal as IsLocalResponse, IsStreaming as IsStreamingResponse, JsonToDdl as JsonToDdlResponse,
    Persist as PersistResponse, SameSemantics as SameSemanticsResponse, Schema as SchemaResponse,
    SemanticHash as SemanticHashResponse, SparkVersion as SparkVersionResponse,
    TreeString as TreeStringResponse, Unpersist as UnpersistResponse,
};
use crate::spark::connect::StorageLevel;
use crate::SPARK_VERSION;

fn default_data_frame_storage_level() -> spec::StorageLevel {
    // Spark Connect defaults DataFrame.persist()/cache() to MEMORY_AND_DISK_DESER.
    spec::StorageLevel {
        use_disk: true,
        use_memory: true,
        use_off_heap: false,
        deserialized: true,
        replication: 1,
    }
}

fn none_data_frame_storage_level() -> StorageLevel {
    StorageLevel {
        use_disk: false,
        use_memory: false,
        use_off_heap: false,
        deserialized: false,
        replication: 1,
    }
}

fn to_proto_storage_level(level: spec::StorageLevel) -> SparkResult<StorageLevel> {
    let spec::StorageLevel {
        use_disk,
        use_memory,
        use_off_heap,
        deserialized,
        replication,
    } = level;
    Ok(StorageLevel {
        use_disk,
        use_memory,
        use_off_heap,
        deserialized,
        replication: i32::try_from(replication).required("replication")?,
    })
}

async fn analyze_schema(ctx: &SessionContext, plan: sc::Plan) -> SparkResult<sc::DataType> {
    let spark = ctx.extension::<SparkSession>()?;
    let resolver = PlanResolver::new(ctx, spark.plan_config()?);
    let NamedPlan { plan, fields } = resolver
        .resolve_named_plan(spec::Plan::Query(plan.try_into()?))
        .await?;
    let schema = if let Some(fields) = fields {
        rename_schema(plan.schema().inner(), fields.as_slice())?
    } else {
        plan.schema().inner().clone()
    };
    to_spark_schema(schema)
}

pub(crate) async fn handle_analyze_schema(
    ctx: &SessionContext,
    request: SchemaRequest,
) -> SparkResult<SchemaResponse> {
    let SchemaRequest { plan } = request;
    let plan = plan.required("plan")?;
    let schema = analyze_schema(ctx, plan).await?;
    Ok(SchemaResponse {
        schema: Some(schema),
    })
}

pub(crate) async fn handle_analyze_explain(
    ctx: &SessionContext,
    request: ExplainRequest,
) -> SparkResult<ExplainResponse> {
    let spark = ctx.extension::<SparkSession>()?;
    let ExplainRequest { plan, explain_mode } = request;
    let plan = plan.required("plan")?;
    let explain_mode = ExplainMode::try_from(explain_mode)?;
    let spec_mode = explain_mode.try_into()?;
    let options = ExplainOptions::from_mode(spec_mode);
    let explain = explain_string(
        ctx,
        spark.plan_config()?,
        spec::Plan::Query(plan.try_into()?),
        options,
    )
    .await?;
    Ok(ExplainResponse {
        explain_string: explain.output,
    })
}

pub(crate) async fn handle_analyze_tree_string(
    ctx: &SessionContext,
    request: TreeStringRequest,
) -> SparkResult<TreeStringResponse> {
    let TreeStringRequest { plan, level } = request;
    let plan = plan.required("plan")?;
    let schema = analyze_schema(ctx, plan).await?;
    Ok(TreeStringResponse {
        tree_string: to_tree_string(&schema, level),
    })
}

pub(crate) async fn handle_analyze_is_local(
    _ctx: &SessionContext,
    _request: IsLocalRequest,
) -> SparkResult<IsLocalResponse> {
    Err(SparkError::todo("handle analyze is local"))
}

pub(crate) async fn handle_analyze_is_streaming(
    _ctx: &SessionContext,
    _request: IsStreamingRequest,
) -> SparkResult<IsStreamingResponse> {
    // TODO: support streaming
    Ok(IsStreamingResponse {
        is_streaming: false,
    })
}

pub(crate) async fn handle_analyze_input_files(
    _ctx: &SessionContext,
    _request: InputFilesRequest,
) -> SparkResult<InputFilesResponse> {
    Err(SparkError::todo("handle analyze input files"))
}

pub(crate) async fn handle_analyze_spark_version(
    _ctx: &SessionContext,
    _request: SparkVersionRequest,
) -> SparkResult<SparkVersionResponse> {
    Ok(SparkVersionResponse {
        version: SPARK_VERSION.to_string(),
    })
}

pub(crate) async fn handle_analyze_ddl_parse(
    ctx: &SessionContext,
    request: DdlParseRequest,
) -> SparkResult<DdlParseResponse> {
    let data_type = parse_spark_data_type(request.ddl_string.as_str())?;
    let spark = ctx.extension::<SparkSession>()?;
    let resolver = PlanResolver::new(ctx, spark.plan_config()?);
    let data_type = resolver.resolve_data_type_for_plan(&data_type)?;
    Ok(DdlParseResponse {
        parsed: Some(data_type.try_into()?),
    })
}

pub(crate) async fn handle_analyze_same_semantics(
    _ctx: &SessionContext,
    _request: SameSemanticsRequest,
) -> SparkResult<SameSemanticsResponse> {
    Err(SparkError::todo("handle analyze same semantics"))
}

pub(crate) async fn handle_analyze_semantic_hash(
    _ctx: &SessionContext,
    _request: SemanticHashRequest,
) -> SparkResult<SemanticHashResponse> {
    Err(SparkError::todo("handle analyze semantic hash"))
}

pub(crate) async fn handle_analyze_persist(
    ctx: &SessionContext,
    request: PersistRequest,
) -> SparkResult<PersistResponse> {
    let spark = ctx.extension::<SparkSession>()?;
    let PersistRequest {
        relation,
        storage_level,
    } = request;
    let relation = relation.required("relation")?;
    let plan: spec::QueryPlan = relation.try_into()?;
    let storage_level = storage_level
        .map(|level| level.try_into())
        .transpose()?
        .unwrap_or_else(default_data_frame_storage_level);
    let _ = spark.persist_dataframe_cache(&plan, storage_level)?;
    Ok(PersistResponse {})
}

pub(crate) async fn handle_analyze_unpersist(
    ctx: &SessionContext,
    request: UnpersistRequest,
) -> SparkResult<UnpersistResponse> {
    let spark = ctx.extension::<SparkSession>()?;
    let UnpersistRequest {
        relation,
        blocking: _,
    } = request;
    let relation = relation.required("relation")?;
    let plan: spec::QueryPlan = relation.try_into()?;
    if let Some(entry) = spark.unpersist_dataframe_cache(&plan)? {
        // Best-effort cleanup. The cache map is authoritative.
        let _ = ctx.deregister_table(&entry.relation_id);
    }
    Ok(UnpersistResponse {})
}

pub(crate) async fn handle_analyze_get_storage_level(
    ctx: &SessionContext,
    request: GetStorageLevelRequest,
) -> SparkResult<GetStorageLevelResponse> {
    let spark = ctx.extension::<SparkSession>()?;
    let GetStorageLevelRequest { relation } = request;
    let relation = relation.required("relation")?;
    let plan: spec::QueryPlan = relation.try_into()?;
    let storage_level = spark
        .get_dataframe_cache_storage_level(&plan)?
        .map(to_proto_storage_level)
        .transpose()?
        .unwrap_or_else(none_data_frame_storage_level);
    Ok(GetStorageLevelResponse {
        storage_level: Some(storage_level),
    })
}

pub(crate) async fn handle_analyze_json_to_ddl(
    _ctx: &SessionContext,
    _request: JsonToDdlRequest,
) -> SparkResult<JsonToDdlResponse> {
    Err(SparkError::todo("handle analyze json to ddl"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::Arc;

    use datafusion::arrow::ipc::reader::StreamReader;
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use futures::StreamExt;
    use sail_common::config::AppConfig;
    use sail_common::runtime::{RuntimeHandle, RuntimeManager};

    use crate::error::SparkResult;
    use crate::executor::ExecutorMetadata;
    use crate::service::handle_execute_relation;
    use crate::session::SparkSessionKey;
    use crate::session_manager::create_spark_session_manager;
    use crate::spark::connect::analyze_plan_request::{
        GetStorageLevel as GetStorageLevelRequest, Persist as PersistRequest,
        Unpersist as UnpersistRequest,
    };
    use crate::spark::connect::relation::RelType;
    use crate::spark::connect::{Relation, RelationCommon, Sql, StorageLevel as ProtoStorageLevel};

    fn create_test_context() -> SparkResult<(RuntimeHandle, SessionContext)> {
        let config = Arc::new(AppConfig::load()?);
        let runtime = RuntimeManager::try_new(&config.runtime)?;
        let handle = runtime.handle();
        let manager = handle
            .primary()
            .block_on(async { create_spark_session_manager(config, handle.clone()) })?;
        let session_key = SparkSessionKey {
            user_id: "".to_string(),
            session_id: "test".to_string(),
        };
        let context = handle
            .primary()
            .block_on(manager.get_or_create_session_context(session_key))?;
        Ok((handle, context))
    }

    fn sql_relation(query: &str, plan_id: i64) -> Relation {
        #[allow(deprecated)]
        Relation {
            common: Some(RelationCommon {
                source_info: String::new(),
                plan_id: Some(plan_id),
                origin: None,
            }),
            #[allow(deprecated)]
            rel_type: Some(RelType::Sql(Sql {
                query: query.to_string(),
                args: HashMap::new(),
                pos_args: vec![],
                named_arguments: HashMap::new(),
                pos_arguments: vec![],
            })),
        }
    }

    fn assert_none_storage_level(level: &ProtoStorageLevel) {
        assert!(!level.use_disk);
        assert!(!level.use_memory);
        assert!(!level.use_off_heap);
        assert!(!level.deserialized);
        assert_eq!(level.replication, 1);
    }

    async fn execute_relation_and_collect_batches(
        ctx: &SessionContext,
        relation: Relation,
    ) -> SparkResult<Vec<RecordBatch>> {
        let metadata = ExecutorMetadata {
            operation_id: uuid::Uuid::new_v4().to_string(),
            tags: vec![],
            reattachable: false,
        };
        let mut stream = handle_execute_relation(ctx, relation, metadata).await?;
        let mut output = vec![];
        while let Some(item) = stream.next().await {
            let response = item.map_err(|e| crate::error::SparkError::internal(e.to_string()))?;
            if let Some(crate::spark::connect::execute_plan_response::ResponseType::ArrowBatch(
                batch,
            )) = response.response_type
            {
                let reader = StreamReader::try_new(Cursor::new(batch.data), None)?;
                for item in reader {
                    output.push(item?);
                }
            }
        }
        Ok(output)
    }

    fn count_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|batch| batch.num_rows()).sum()
    }

    #[test]
    fn test_persist_storage_level_round_trip() -> SparkResult<()> {
        let (handle, context) = create_test_context()?;
        let relation = sql_relation("SELECT 1 AS x", 1001);
        let storage_level = ProtoStorageLevel {
            use_disk: true,
            use_memory: true,
            use_off_heap: false,
            deserialized: true,
            replication: 1,
        };
        handle.primary().block_on(async {
            super::handle_analyze_persist(
                &context,
                PersistRequest {
                    relation: Some(relation.clone()),
                    storage_level: Some(storage_level.clone()),
                },
            )
            .await?;

            let response = super::handle_analyze_get_storage_level(
                &context,
                GetStorageLevelRequest {
                    relation: Some(relation.clone()),
                },
            )
            .await?;
            assert_eq!(response.storage_level, Some(storage_level));

            super::handle_analyze_unpersist(
                &context,
                UnpersistRequest {
                    relation: Some(relation.clone()),
                    blocking: Some(false),
                },
            )
            .await?;

            let response = super::handle_analyze_get_storage_level(
                &context,
                GetStorageLevelRequest {
                    relation: Some(relation),
                },
            )
            .await?;
            let level = response
                .storage_level
                .ok_or_else(|| crate::error::SparkError::internal("missing storage level"))?;
            assert_none_storage_level(&level);
            Ok::<(), crate::error::SparkError>(())
        })?;
        Ok(())
    }

    #[test]
    fn test_persist_actions_and_unpersist_actions() -> SparkResult<()> {
        let (handle, context) = create_test_context()?;
        let relation = sql_relation("SELECT 1 AS x", 2002);
        handle.primary().block_on(async {
            super::handle_analyze_persist(
                &context,
                PersistRequest {
                    relation: Some(relation.clone()),
                    storage_level: None,
                },
            )
            .await?;
            let level = super::handle_analyze_get_storage_level(
                &context,
                GetStorageLevelRequest {
                    relation: Some(relation.clone()),
                },
            )
            .await?
            .storage_level
            .ok_or_else(|| crate::error::SparkError::internal("missing storage level"))?;
            assert_eq!(
                level,
                ProtoStorageLevel {
                    use_disk: true,
                    use_memory: true,
                    use_off_heap: false,
                    deserialized: true,
                    replication: 1,
                }
            );

            let first = count_rows(
                execute_relation_and_collect_batches(&context, relation.clone())
                    .await?
                    .as_slice(),
            );
            let second = count_rows(
                execute_relation_and_collect_batches(&context, relation.clone())
                    .await?
                    .as_slice(),
            );
            assert_eq!(first, second);

            super::handle_analyze_unpersist(
                &context,
                UnpersistRequest {
                    relation: Some(relation.clone()),
                    blocking: Some(false),
                },
            )
            .await?;

            let third = count_rows(
                execute_relation_and_collect_batches(&context, relation)
                    .await?
                    .as_slice(),
            );
            assert_eq!(second, third);
            Ok::<(), crate::error::SparkError>(())
        })?;
        Ok(())
    }
}
