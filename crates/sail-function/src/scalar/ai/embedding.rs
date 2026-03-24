use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, FixedSizeListArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::cast::{as_large_string_array, as_string_array, as_string_view_array};
use datafusion_common::{exec_err, internal_err, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

use crate::error::generic_exec_err;
use crate::functions_utils::make_scalar_function;

const DEFAULT_MODEL: &str = "text-embedding-3-small";
const SMALL_MODEL: &str = "text-embedding-3-small";
const LARGE_MODEL: &str = "text-embedding-3-large";
const SMALL_DIMENSIONS: i32 = 512;
const LARGE_DIMENSIONS: i32 = 1024;
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TEXT_CHARS: usize = 2_048;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Embedding {
    signature: Signature,
}

impl Default for Embedding {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedding {
    pub const NAME: &'static str = "embedding";

    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Volatile),
        }
    }
}

impl ScalarUDFImpl for Embedding {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        Self::NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(fixed_size_vector_type(default_model_dimensions()))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let model = scalar_string_argument(args.scalar_arguments, 2)?;
        let dimensions = dimensions_for_model(model.as_deref())?;
        Ok(Arc::new(Field::new(
            Self::NAME,
            fixed_size_vector_type(dimensions),
            true,
        )))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if !(2..=3).contains(&arg_types.len()) {
            return exec_err!(
                "Spark `embedding` function requires 2 or 3 arguments, got {}",
                arg_types.len()
            );
        }

        let mut coerced = Vec::with_capacity(arg_types.len());
        for (index, arg_type) in arg_types.iter().enumerate() {
            let is_string_like = matches!(
                arg_type,
                DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
            );
            let is_nullable_model = index == 2 && matches!(arg_type, DataType::Null);
            let is_nullable_text_or_token = index < 2 && matches!(arg_type, DataType::Null);
            if is_string_like || is_nullable_model || is_nullable_text_or_token {
                coerced.push(if matches!(arg_type, DataType::LargeUtf8) {
                    DataType::LargeUtf8
                } else {
                    DataType::Utf8
                });
            } else {
                let argument_name = match index {
                    0 => "text",
                    1 => "api token",
                    2 => "model",
                    _ => "unknown",
                };
                return exec_err!(
                    "Spark `embedding` function: {argument_name} argument must be STRING, got {arg_type:?}"
                );
            }
        }
        Ok(coerced)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        if !(2..=3).contains(&args.len()) {
            return exec_err!(
                "Spark `embedding` function requires 2 or 3 arguments, got {}",
                args.len()
            );
        }
        make_scalar_function(embedding_inner, vec![])(&args)
    }
}

fn fixed_size_vector_type(dimensions: i32) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new_list_field(DataType::Float32, true)),
        dimensions,
    )
}

fn default_model_dimensions() -> i32 {
    SMALL_DIMENSIONS
}

fn model_dimensions(model: &str) -> Result<i32> {
    match model {
        SMALL_MODEL => Ok(SMALL_DIMENSIONS),
        LARGE_MODEL => Ok(LARGE_DIMENSIONS),
        other => exec_err!(
            "Spark `embedding` function: unsupported model {other}. Expected `{SMALL_MODEL}` or `{LARGE_MODEL}`"
        ),
    }
}

fn dimensions_for_model(model: Option<&str>) -> Result<i32> {
    match model {
        Some(model) => model_dimensions(model),
        None => Ok(default_model_dimensions()),
    }
}

fn embedding_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let text_values = collect_string_values(&args[0])?;
    let api_key = required_scalar_string_from_array(&args[1], "api token")?;

    let model = if args.len() == 3 {
        scalar_string_from_array_named(&args[2], "model")?
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    } else {
        DEFAULT_MODEL.to_string()
    };

    let dimensions = model_dimensions(&model)?;
    let config = EmbeddingConfig::new(
        Client::builder()
            .timeout(std::time::Duration::from_millis(DEFAULT_TIMEOUT_MS))
            .build()
            .map_err(|err| {
                generic_exec_err(
                    Embedding::NAME,
                    &format!("failed to build HTTP client: {err}"),
                )
            })?,
        DEFAULT_BASE_URL.to_string(),
        api_key,
        model,
        None,
    );

    let values = text_values
        .iter()
        .map(|text_opt| match text_opt {
            None => Ok(None),
            Some(text) => embed_text(&config, text, dimensions).map(Some),
        })
        .collect::<Result<Vec<_>>>()?;

    let array =
        FixedSizeListArray::from_iter_primitive::<datafusion::arrow::datatypes::Float32Type, _, _>(
            values, dimensions,
        );
    Ok(Arc::new(array))
}

fn scalar_string_argument(
    scalar_arguments: &[Option<&ScalarValue>],
    index: usize,
) -> Result<Option<String>> {
    scalar_arguments
        .get(index)
        .and_then(|arg| *arg)
        .map(scalar_value_to_string)
        .transpose()
}

fn collect_string_values(array: &ArrayRef) -> Result<Vec<Option<String>>> {
    match array.data_type() {
        DataType::Utf8 => Ok(as_string_array(array)?
            .iter()
            .map(|value| value.map(ToString::to_string))
            .collect()),
        DataType::Utf8View => Ok(as_string_view_array(array)?
            .iter()
            .map(|value| value.map(ToString::to_string))
            .collect()),
        DataType::LargeUtf8 => Ok(as_large_string_array(array)?
            .iter()
            .map(|value| value.map(ToString::to_string))
            .collect()),
        other => exec_err!("Spark `embedding` function: text input must be STRING, got {other:?}"),
    }
}

fn scalar_value_to_string(value: &ScalarValue) -> Result<String> {
    match value {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::Utf8View(Some(value))
        | ScalarValue::LargeUtf8(Some(value)) => Ok(value.clone()),
        other => internal_err!("Expected string scalar argument, got {other:?}"),
    }
}

fn scalar_string_from_array_named(array: &ArrayRef, argument_name: &str) -> Result<Option<String>> {
    match array.data_type() {
        DataType::Utf8 | DataType::Utf8View => {
            let values = collect_string_values(array)?;
            if values.len() != 1 {
                return exec_err!(
                    "Spark `embedding` function: {argument_name} argument must be a scalar string literal"
                );
            }
            Ok(values.into_iter().next().flatten())
        }
        DataType::LargeUtf8 => {
            let values = collect_string_values(array)?;
            if values.len() != 1 {
                return exec_err!(
                    "Spark `embedding` function: {argument_name} argument must be a scalar string literal"
                );
            }
            Ok(values.into_iter().next().flatten())
        }
        DataType::Null => Ok(None),
        other => {
            exec_err!(
                "Spark `embedding` function: {argument_name} argument must be STRING, got {other:?}"
            )
        }
    }
}

fn required_scalar_string_from_array(array: &ArrayRef, argument_name: &str) -> Result<String> {
    scalar_string_from_array_named(array, argument_name)?.ok_or_else(|| {
        generic_exec_err(
            Embedding::NAME,
            &format!("{argument_name} argument cannot be NULL"),
        )
    })
}

#[derive(Debug, Clone)]
struct EmbeddingConfig {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    dimensions: Option<i64>,
}

impl EmbeddingConfig {
    fn new(
        client: Client,
        base_url: String,
        api_key: String,
        model: String,
        dimensions: Option<i64>,
    ) -> Self {
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            dimensions,
        }
    }
}

fn embed_text(
    config: &EmbeddingConfig,
    text: &str,
    expected_dimensions: i32,
) -> Result<Vec<Option<f32>>> {
    let mut body = json!({
        "input": truncate_text(text),
        "model": config.model,
    });
    if let Some(dimensions) = config.dimensions {
        body["dimensions"] = json!(dimensions);
    }
    let url = format!("{}/embeddings", config.base_url);
    let response = config
        .client
        .post(url)
        .bearer_auth(&config.api_key)
        .json(&body)
        .send()
        .map_err(|err| {
            generic_exec_err(Embedding::NAME, &format!("embedding request failed: {err}"))
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| "<unreadable response body>".to_string());
        return Err(generic_exec_err(
            Embedding::NAME,
            &format!("embedding request failed with status {status}: {body}"),
        ));
    }

    let response: EmbeddingResponse = response.json().map_err(|err| {
        generic_exec_err(
            Embedding::NAME,
            &format!("invalid embedding response: {err}"),
        )
    })?;
    let embedding = response.data.first().ok_or_else(|| {
        generic_exec_err(Embedding::NAME, "embedding response did not contain data")
    })?;
    if embedding.embedding.len() != expected_dimensions as usize {
        return Err(generic_exec_err(
            Embedding::NAME,
            &format!(
                "embedding response dimension mismatch: expected {expected_dimensions}, got {}",
                embedding.embedding.len()
            ),
        ));
    }
    Ok(embedding.embedding.iter().copied().map(Some).collect())
}

fn truncate_text(text: &str) -> &str {
    if text.len() <= MAX_TEXT_CHARS {
        return text;
    }
    let mut end = MAX_TEXT_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use tokio::runtime::Runtime;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn test_embedding_request() {
        let runtime = Runtime::new().expect("runtime");
        let server = runtime.block_on(MockServer::start());

        runtime.block_on(async {
            Mock::given(method("POST"))
                .and(path("/embeddings"))
                .and(header("authorization", "Bearer secret"))
                .and(body_partial_json(json!({
                    "input": "hello world",
                    "model": SMALL_MODEL
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "data": [{
                        "embedding": vec![0.5_f32; SMALL_DIMENSIONS as usize]
                    }]
                })))
                .mount(&server)
                .await;
        });

        let config = EmbeddingConfig::new(
            Client::builder()
                .timeout(std::time::Duration::from_millis(1_000))
                .build()
                .expect("client"),
            server.uri(),
            "secret".to_string(),
            SMALL_MODEL.to_string(),
            None,
        );
        let embedding = embed_text(&config, "hello world", SMALL_DIMENSIONS).expect("embedding");
        assert_eq!(embedding.len(), SMALL_DIMENSIONS as usize);
        assert_eq!(embedding[0], Some(0.5));
    }

    #[test]
    fn test_truncate_text_preserves_utf8() {
        let value = "abc".repeat(700) + "🙂";
        let truncated = truncate_text(&value);
        assert!(truncated.len() <= MAX_TEXT_CHARS);
        assert!(truncated.is_char_boundary(truncated.len()));
    }
}
