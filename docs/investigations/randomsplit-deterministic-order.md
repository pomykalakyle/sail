# Investigation: randomSplit Doctest Failure

## Problem Statement

The PySpark `randomSplit` doctest fails when running against Sail:

```
>>> splits = df.randomSplit([1.0, 2.0], 24)
>>> splits[0].count()
Expected:
    2
Got:
    1
```

**Test command:**
```bash
hatch run test-spark.spark-4.1.1:env scripts/spark-tests/run-tests.sh \
    --doctest-modules --pyargs pyspark.sql.dataframe -k "randomSplit"
```

**Actual results:**
- Sail produces: `split[0].count() = 1`, `split[1].count() = 3`
- Spark expects: `split[0].count() = 2`, `split[1].count() = 2`

---

## Investigation Timeline

### Step 1: Initial Hypothesis - Wrong RNG Algorithm

The first hypothesis was that Sail's random number generator (ChaCha8Rng) produced different values than Spark's XORShiftRandom.

**Evidence gathered:**
- Sail used `ChaCha8Rng` from the `rand_chacha` crate
- Spark uses a custom `XORShiftRandom` class with MurmurHash3 seed hashing

**Location:** `crates/sail-function/src/scalar/math/random.rs`
```rust
// Original code
let mut rng = ChaCha8Rng::seed_from_u64(seed);
let values = std::iter::repeat_with(|| rng.random_range(0.0..1.0)).take(number_rows);
```

### Step 2: Created Spark-Compatible RNG

Created a new crate `sail-spark-random` with a Rust port of Spark's XORShiftRandom:

**Location:** `crates/sail-spark-random/src/xorshift.rs`

The implementation includes:
- MurmurHash3 seed hashing (matching Scala's `scala.util.hashing.MurmurHash3.bytesHash`)
- XORShift algorithm with shifts 21, 35, 4
- Java's `Random.nextDouble()` compatible output

**Reference:** [Spark's XORShiftRandom.scala](https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/util/random/XORShiftRandom.scala)

### Step 3: Verified RNG Produces Identical Values

Created a test to verify the Rust implementation matches Spark exactly:

```rust
// Expected values obtained from spark-shell
const SPARK_XORSHIFT_SEED_24: [f64; 5] = [
    0.3943255396952755,
    0.48619924381941027,
    0.2923951640552428,
    0.33335316633280176,
    0.3981939745854918,
];

#[test]
fn test_spark_xorshift_seed_24() {
    let mut rng = SparkXorShiftRandom::new(24);
    for expected in SPARK_XORSHIFT_SEED_24 {
        let actual = rng.next_double();
        assert!((actual - expected).abs() < 1e-15);
    }
}
```

**Result:** All tests passed - the RNG implementation is correct.

### Step 4: Test Still Failed After RNG Fix

After integrating `SparkXorShiftRandom` into Sail's `Random` UDF, the test still failed:

```
Split 0 count: 1
Split 1 count: 3
```

This ruled out the RNG algorithm as the root cause.

### Step 5: Investigated Spark Connect Server

Examined how Spark's server handles the Sample operation:

**Location:** `spark/sql/connect/server/src/main/scala/org/apache/spark/sql/connect/planner/SparkConnectPlanner.scala`

```scala
private def transformSample(rel: proto.Sample): LogicalPlan = {
  val plan = if (rel.getDeterministicOrder) {
    val input = Dataset.ofRows(session, transformRelation(rel.getInput))

    // It is possible that the underlying dataframe doesn't guarantee the ordering
    // of rows in its constituent partitions each time a split is materialized
    // which could result in overlapping splits. To prevent this, we explicitly
    // sort each input partition to make the ordering deterministic.
    val sortOrder = input.logicalPlan.output
      .filter(attr => RowOrdering.isOrderable(attr.dataType))
      .map(SortOrder(_, Ascending))
    if (sortOrder.nonEmpty) {
      Sort(sortOrder, global = false, input.logicalPlan)
    } else {
      input.cache()
      input.logicalPlan
    }
  } else {
    transformRelation(rel.getInput)
  }

  Sample(rel.getLowerBound, rel.getUpperBound, rel.getWithReplacement,
    if (rel.hasSeed) rel.getSeed else Utils.random.nextLong, plan)
}
```

### Step 6: Found the Root Cause

**Key discovery:** When `deterministic_order=true` (which `randomSplit` sets), Spark **sorts the input data** by all orderable columns in ascending order before sampling.

Sail's implementation ignores this flag entirely.

---

## Root Cause Analysis

### The Problem

The `randomSplit` method in PySpark sets `deterministic_order=True`:

**Location:** `.venvs/test-spark.spark-4.1.1/lib/python3.11/site-packages/pyspark/sql/connect/dataframe.py:1098-1106`
```python
samplePlan = DataFrame(
    plan.Sample(
        child=self._plan,
        lower_bound=lowerBound,
        upper_bound=upperBound,
        with_replacement=False,
        seed=int(seed),
        deterministic_order=True,  # <-- This is set!
    ),
    session=self._session,
)
```

### Spark's Behavior

When `deterministic_order=true`, Spark sorts the input by all columns ascending before applying random sampling. This ensures:
1. Consistent row ordering across multiple split operations
2. No overlapping rows between splits

### Sail's Behavior

**Location:** `crates/sail-plan/src/resolver/query/sample.rs:23-30`
```rust
let spec::Sample {
    input,
    lower_bound,
    upper_bound,
    with_replacement,
    seed,
    ..  // <-- deterministic_order is IGNORED!
} = sample;
```

Sail completely ignores the `deterministic_order` field.

### Impact on Test Data

For the test DataFrame:
```python
df = spark.createDataFrame([
    Row(age=10, height=80, name="Alice"),
    Row(age=5, height=None, name="Bob"),
    Row(age=None, height=None, name="Tom"),
    Row(age=None, height=None, name=None),
])
```

**Sail's row order (original):** Alice, Bob, Tom, None

**Spark's row order (sorted by age ASC, height ASC, name ASC, nulls first):** 
Different ordering due to null handling and multi-column sort.

Since the same random values are assigned to different rows, the split results differ.

---

## How to Reproduce

### 1. Start the Sail server

```bash
cd /workspaces/sail
hatch run scripts/spark-tests/run-server.sh
```

### 2. Run the test script

```python
from pyspark.sql import SparkSession, Row

spark = SparkSession.builder \
    .remote("sc://localhost:50051") \
    .appName("RandomSplit Test") \
    .getOrCreate()

df = spark.createDataFrame([
    Row(age=10, height=80, name="Alice"),
    Row(age=5, height=None, name="Bob"),
    Row(age=None, height=None, name="Tom"),
    Row(age=None, height=None, name=None),
])

splits = df.randomSplit([1.0, 2.0], 24)
print(f"Split 0 count: {splits[0].count()}")  # Sail: 1, Spark: 2
print(f"Split 1 count: {splits[1].count()}")  # Sail: 3, Spark: 2

spark.stop()
```

### 3. Run the doctest directly

```bash
hatch run test-spark.spark-4.1.1:env scripts/spark-tests/run-tests.sh \
    --doctest-modules --pyargs pyspark.sql.dataframe -k "randomSplit" -v
```

---

## Proposed Fix

Sail needs to implement the `deterministic_order` sorting behavior in `sample.rs`:

1. Extract the `deterministic_order` field from the Sample spec
2. When `deterministic_order=true`:
   - Get all orderable columns from the input schema
   - Sort the input by those columns in ascending order (with appropriate null handling)
3. Then apply the existing sampling logic

### Reference Implementation

From Spark's `SparkConnectPlanner.scala`:
```scala
val sortOrder = input.logicalPlan.output
  .filter(attr => RowOrdering.isOrderable(attr.dataType))
  .map(SortOrder(_, Ascending))
if (sortOrder.nonEmpty) {
  Sort(sortOrder, global = false, input.logicalPlan)
}
```

---

## Related Files

### Sail
- `crates/sail-plan/src/resolver/query/sample.rs` - Sample operation resolver (needs fix)
- `crates/sail-spark-random/src/xorshift.rs` - Spark-compatible RNG (implemented)
- `crates/sail-function/src/scalar/math/random.rs` - Random UDF (updated to use SparkXorShiftRandom)
- `crates/sail-common/src/spec/plan.rs:694-701` - Sample spec definition (has `deterministic_order` field)

### Spark
- `sql/connect/server/src/main/scala/org/apache/spark/sql/connect/planner/SparkConnectPlanner.scala:409-437` - transformSample with sorting logic
- `core/src/main/scala/org/apache/spark/util/random/XORShiftRandom.scala` - RNG implementation
- `core/src/main/scala/org/apache/spark/rdd/RDD.scala:603-609` - randomSampleWithRange
- `core/src/main/scala/org/apache/spark/util/random/RandomSampler.scala:98-133` - BernoulliCellSampler

---

## Conclusion

The `randomSplit` test failure is **not** caused by the RNG algorithm (which is now Spark-compatible), but by Sail's failure to implement the `deterministic_order` sorting behavior. When this flag is true, Spark sorts the input data before sampling to ensure deterministic and non-overlapping splits. Sail ignores this flag, resulting in different rows receiving different random values and thus different split results.
