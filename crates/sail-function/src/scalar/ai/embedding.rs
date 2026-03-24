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
const OPENAI_BASE_URL_ENV: &str = "SAIL_EMBEDDING_BASE_URL";
const OPENAI_API_KEY_ENV: &str = "SAIL_EMBEDDING_API_KEY";
const OPENAI_MODEL_ENV: &str = "SAIL_EMBEDDING_MODEL";
const OPENAI_DIMENSIONS_ENV: &str = "SAIL_EMBEDDING_DIMENSIONS";
const OPENAI_TIMEOUT_MS_ENV: &str = "SAIL_EMBEDDING_TIMEOUT_MS";
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
        let model = scalar_string_argument(args.scalar_arguments, 1)?
            .or_else(|| std::env::var(OPENAI_MODEL_ENV).ok());
        let dimensions = dimensions_for_model(model.as_deref())?;
        Ok(Arc::new(Field::new(
            Self::NAME,
            fixed_size_vector_type(dimensions),
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        if !(1..=2).contains(&args.len()) {
            return exec_err!(
                "Spark `embedding` function requires 1 or 2 arguments, got {}",
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

    let model = if args.len() == 2 {
        scalar_string_from_array(&args[1])?.unwrap_or_else(|| DEFAULT_MODEL.to_string())
    } else {
        std::env::var(OPENAI_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_string())
    };

    let dimensions = model_dimensions(&model)?;
    let config = EmbeddingConfig::from_env(&model)?;

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

fn scalar_string_from_array(array: &ArrayRef) -> Result<Option<String>> {
    match array.data_type() {
        DataType::Utf8 | DataType::Utf8View => {
            let values = collect_string_values(array)?;
            if values.len() != 1 {
                return exec_err!(
                    "Spark `embedding` function: model argument must be a scalar string literal"
                );
            }
            Ok(values.into_iter().next().flatten())
        }
        DataType::LargeUtf8 => {
            let values = collect_string_values(array)?;
            if values.len() != 1 {
                return exec_err!(
                    "Spark `embedding` function: model argument must be a scalar string literal"
                );
            }
            Ok(values.into_iter().next().flatten())
        }
        DataType::Null => Ok(None),
        other => {
            exec_err!("Spark `embedding` function: model argument must be STRING, got {other:?}")
        }
    }
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
    fn from_env(model: &str) -> Result<Self> {
        let base_url =
            std::env::var(OPENAI_BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let api_key = std::env::var(OPENAI_API_KEY_ENV).map_err(|_| {
            generic_exec_err(
                Embedding::NAME,
                &format!("missing API key in environment variable `{OPENAI_API_KEY_ENV}`"),
            )
        })?;
        let timeout_ms = std::env::var(OPENAI_TIMEOUT_MS_ENV)
            .ok()
            .map(|value| {
                value.parse::<u64>().map_err(|_| {
                    generic_exec_err(
                        Embedding::NAME,
                        &format!(
                            "invalid timeout value `{value}` in environment variable `{OPENAI_TIMEOUT_MS_ENV}`"
                        ),
                    )
                })
            })
            .transpose()?
            .unwrap_or(DEFAULT_TIMEOUT_MS);
        let dimensions = std::env::var(OPENAI_DIMENSIONS_ENV)
            .ok()
            .map(|value| {
                value.parse::<i64>().map_err(|_| {
                    generic_exec_err(
                        Embedding::NAME,
                        &format!(
                            "invalid dimensions value `{value}` in environment variable `{OPENAI_DIMENSIONS_ENV}`"
                        ),
                    )
                })
            })
            .transpose()?;
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .build()
            .map_err(|err| {
                generic_exec_err(
                    Embedding::NAME,
                    &format!("failed to build HTTP client: {err}"),
                )
            })?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: model.to_string(),
            dimensions,
        })
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

    struct EnvGuard {
        vars: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(values: &[(&'static str, &str)]) -> Self {
            let vars = values
                .iter()
                .map(|(key, value)| {
                    let prev = std::env::var(key).ok();
                    std::env::set_var(key, value);
                    (*key, prev)
                })
                .collect();
            Self { vars }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.vars.drain(..) {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }

    #[test]
    fn test_embedding_request() {
        let runtime = Runtime::new().expect("runtime");
        let server = runtime.block_on(MockServer::start());
        let _guard = EnvGuard::set(&[
            (OPENAI_BASE_URL_ENV, &server.uri()),
            (OPENAI_API_KEY_ENV, "secret"),
            (OPENAI_TIMEOUT_MS_ENV, "1000"),
        ]);

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

        let config = EmbeddingConfig::from_env(SMALL_MODEL).expect("config");
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
