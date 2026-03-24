use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::{exec_err, plan_err, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use crate::scalar::map::utils::{get_list_offsets, get_list_values};

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct VectorCosineSimilarity {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for VectorCosineSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorCosineSimilarity {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            aliases: vec!["array_cosine_similarity".to_string()],
        }
    }
}

impl ScalarUDFImpl for VectorCosineSimilarity {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "vector_cosine_similarity"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 2 {
            return exec_err!(
                "Spark `vector_cosine_similarity` function requires 2 arguments, got {}",
                arg_types.len()
            );
        }

        let coerced = arg_types
            .iter()
            .map(coerce_vector_type)
            .collect::<Result<Vec<_>>>()?;
        let left = &coerced[0];
        let right = &coerced[1];

        let left_element = list_element_type(left)?;
        let right_element = list_element_type(right)?;
        let element_type = common_numeric_type(left_element, right_element)?;
        let target = list_type_like(left, element_type)?;
        Ok(vec![target; 2])
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        let [left, right] = args.as_slice() else {
            return exec_err!(
                "Spark `vector_cosine_similarity` function requires 2 arguments, got {}",
                args.len()
            );
        };

        let left_array = columnar_to_array(left)?;
        let right_array = columnar_to_array(right)?;
        let result = vector_cosine_similarity_arrays(&left_array, &right_array)?;

        if matches!(left, ColumnarValue::Scalar(_)) && matches!(right, ColumnarValue::Scalar(_)) {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

fn columnar_to_array(value: &ColumnarValue) -> Result<ArrayRef> {
    match value {
        ColumnarValue::Array(array) => Ok(array.clone()),
        ColumnarValue::Scalar(scalar) => scalar.to_array(),
    }
}

fn vector_cosine_similarity_arrays(left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef> {
    let left_offsets = get_list_offsets(left)?;
    let right_offsets = get_list_offsets(right)?;
    if left.len() != right.len() {
        return exec_err!(
            "Spark `vector_cosine_similarity` function requires arrays with the same row count"
        );
    }

    let left_values = cast_list_values_to_f64(left)?;
    let right_values = cast_list_values_to_f64(right)?;
    let left_values = as_float64_array(&left_values)?;
    let right_values = as_float64_array(&right_values)?;

    let result = (0..left.len())
        .map(|row| {
            if left.is_null(row) || right.is_null(row) {
                return Ok(None);
            }

            let (left_start, left_end) = offset_range(&left_offsets, row);
            let (right_start, right_end) = offset_range(&right_offsets, row);
            let left_len = left_end - left_start;
            let right_len = right_end - right_start;

            if left_len != right_len {
                return exec_err!(
                    "Spark `vector_cosine_similarity` function requires vectors with the same dimension"
                );
            }
            if left_len == 0 {
                return exec_err!(
                    "Spark `vector_cosine_similarity` function does not support empty vectors"
                );
            }

            let mut dot = 0.0_f64;
            let mut left_norm_sq = 0.0_f64;
            let mut right_norm_sq = 0.0_f64;

            for idx in 0..left_len {
                let left_idx = left_start + idx;
                let right_idx = right_start + idx;
                if left_values.is_null(left_idx) || right_values.is_null(right_idx) {
                    return Ok(None);
                }
                let left_value = left_values.value(left_idx);
                let right_value = right_values.value(right_idx);
                dot += left_value * right_value;
                left_norm_sq += left_value * left_value;
                right_norm_sq += right_value * right_value;
            }

            if left_norm_sq == 0.0 || right_norm_sq == 0.0 {
                return Ok(None);
            }

            Ok(Some(dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt())))
        })
        .collect::<Result<Float64Array>>()?;

    Ok(Arc::new(result) as ArrayRef)
}

fn cast_list_values_to_f64(array: &ArrayRef) -> Result<ArrayRef> {
    let values = get_list_values(array)?.clone();
    let values = match values.data_type() {
        DataType::Float64 => values,
        DataType::Float32
        | DataType::Float16
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => datafusion::arrow::compute::cast(&values, &DataType::Float64)?,
        other => {
            return exec_err!(
                "Spark `vector_cosine_similarity` function requires numeric array elements, got {other}"
            );
        }
    };
    Ok(values)
}

fn coerce_vector_type(data_type: &DataType) -> Result<DataType> {
    match data_type {
        DataType::Null => Ok(DataType::List(Arc::new(
            datafusion::arrow::datatypes::Field::new_list_field(DataType::Float64, true),
        ))),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            Ok(data_type.clone())
        }
        other => exec_err!(
            "Spark `vector_cosine_similarity` function requires ARRAY arguments, got {other}"
        ),
    }
}

fn list_element_type(data_type: &DataType) -> Result<&DataType> {
    match data_type {
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            Ok(field.data_type())
        }
        other => plan_err!("Expected list type for vector similarity coercion, got {other}"),
    }
}

fn common_numeric_type(left: &DataType, right: &DataType) -> Result<DataType> {
    if left == &DataType::Null {
        return Ok(right.clone());
    }
    if right == &DataType::Null {
        return Ok(left.clone());
    }
    if !left.is_numeric() || !right.is_numeric() {
        return exec_err!(
            "Spark `vector_cosine_similarity` function requires numeric array elements, got {left} and {right}"
        );
    }
    if left.is_floating() || right.is_floating() {
        Ok(DataType::Float64)
    } else {
        Ok(DataType::Float64)
    }
}

fn list_type_like(template: &DataType, element_type: DataType) -> Result<DataType> {
    let field = Arc::new(datafusion::arrow::datatypes::Field::new_list_field(
        element_type,
        true,
    ));
    match template {
        DataType::List(_) => Ok(DataType::List(field)),
        DataType::LargeList(_) => Ok(DataType::LargeList(field)),
        DataType::FixedSizeList(_, size) => Ok(DataType::FixedSizeList(field, *size)),
        other => plan_err!("Expected list type for vector similarity coercion, got {other}"),
    }
}

fn offset_range(offsets: &[i32], row: usize) -> (usize, usize) {
    (offsets[row] as usize, offsets[row + 1] as usize)
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::ListArray;
    use datafusion::arrow::datatypes::Float64Type;
    use datafusion_common::cast::as_float64_array;

    use super::*;

    #[test]
    fn computes_similarity() -> Result<()> {
        let left = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(1.0), Some(0.0)]),
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
        ])) as ArrayRef;
        let right = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(1.0), Some(0.0)]),
            Some(vec![Some(4.0), Some(5.0), Some(6.0)]),
        ])) as ArrayRef;

        let result = vector_cosine_similarity_arrays(&left, &right)?;
        let result = as_float64_array(&result)?;

        assert_eq!(result.value(0), 1.0);
        assert!((result.value(1) - 0.9746318461970762).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn returns_null_for_zero_vectors_and_null_elements() -> Result<()> {
        let left = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(0.0), Some(0.0)]),
            Some(vec![Some(1.0), None]),
        ])) as ArrayRef;
        let right = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(1.0), Some(1.0)]),
            Some(vec![Some(1.0), Some(2.0)]),
        ])) as ArrayRef;

        let result = vector_cosine_similarity_arrays(&left, &right)?;
        let result = as_float64_array(&result)?;

        assert!(result.is_null(0));
        assert!(result.is_null(1));
        Ok(())
    }
}
