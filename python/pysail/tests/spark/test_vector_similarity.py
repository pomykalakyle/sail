import pytest
from pyspark.sql.types import Row

from pysail.testing.spark.utils.common import is_jvm_spark


@pytest.mark.skipif(is_jvm_spark(), reason="Sail-only built-in function")
def test_vector_cosine_similarity_sql(spark):
    assert spark.sql(
        """
        SELECT
            vector_cosine_similarity(array(1.0D, 0.0D), array(1.0D, 0.0D)) AS same,
            array_cosine_similarity(array(1.0D, 0.0D), array(0.0D, 1.0D)) AS orthogonal
        """
    ).collect() == [Row(same=1.0, orthogonal=0.0)]


@pytest.mark.skipif(is_jvm_spark(), reason="Sail-only built-in function")
def test_vector_cosine_similarity_null_behavior(spark):
    assert spark.sql(
        """
        SELECT
            vector_cosine_similarity(array(0.0D, 0.0D), array(1.0D, 1.0D)) AS zero_vector,
            vector_cosine_similarity(array(1.0D, CAST(NULL AS DOUBLE)), array(1.0D, 2.0D)) AS null_item,
            vector_cosine_similarity(CAST(NULL AS ARRAY<DOUBLE>), array(1.0D, 2.0D)) AS null_vector
        """
    ).collect() == [Row(zero_vector=None, null_item=None, null_vector=None)]
