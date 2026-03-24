use sail_function::scalar::ai::embedding::Embedding;
use sail_function::scalar::ai::vector_cosine_similarity::VectorCosineSimilarity;

use crate::function::common::ScalarFunction;

pub(super) fn list_built_in_ai_functions() -> Vec<(&'static str, ScalarFunction)> {
    use crate::function::common::ScalarFunctionBuilder as F;

    vec![
        ("embedding", F::udf(Embedding::new())),
        (
            "array_cosine_similarity",
            F::udf(VectorCosineSimilarity::new()),
        ),
        (
            "vector_cosine_similarity",
            F::udf(VectorCosineSimilarity::new()),
        ),
    ]
}
