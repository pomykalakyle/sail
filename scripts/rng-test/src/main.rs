use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_xoshiro::Xoroshiro128Plus;
use std::io::Cursor;

/// Port of Spark's XORShiftRandom
/// https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/util/random/XORShiftRandom.scala
struct SparkXorShiftRandom {
    seed: i64,
}

impl SparkXorShiftRandom {
    fn new(init: i64) -> Self {
        Self {
            seed: Self::hash_seed(init),
        }
    }

    /// Port of XORShiftRandom.hashSeed using MurmurHash3
    /// Scala's MurmurHash3.bytesHash with MurmurHash3.arraySeed (0x3c074a61)
    fn hash_seed(seed: i64) -> i64 {
        // Convert seed to big-endian bytes (like Java's ByteBuffer.putLong)
        let bytes = seed.to_be_bytes();

        // MurmurHash3.arraySeed = 0x3c074a61
        let array_seed: u32 = 0x3c074a61;

        // Hash twice like Spark does
        let low_bits = Self::murmur3_bytes_hash(&bytes, array_seed);
        let high_bits = Self::murmur3_bytes_hash(&bytes, low_bits);

        // Combine: (highBits.toLong << 32) | (lowBits.toLong & 0xFFFFFFFFL)
        ((high_bits as i64) << 32) | ((low_bits as i64) & 0xFFFFFFFF)
    }

    /// Port of Scala's MurmurHash3.bytesHash
    /// This matches scala.util.hashing.MurmurHash3.bytesHash
    fn murmur3_bytes_hash(data: &[u8], seed: u32) -> u32 {
        let mut h1 = seed;
        let len = data.len();

        // Process 4-byte chunks
        let n_blocks = len / 4;
        for i in 0..n_blocks {
            let i4 = i * 4;
            // Little-endian read for the block
            let k1 = u32::from_le_bytes([data[i4], data[i4 + 1], data[i4 + 2], data[i4 + 3]]);
            h1 = Self::mix(h1, k1);
        }

        // Process remaining bytes
        let tail_start = n_blocks * 4;
        let mut k1: u32 = 0;
        let tail_len = len - tail_start;

        if tail_len >= 3 {
            k1 ^= (data[tail_start + 2] as u32) << 16;
        }
        if tail_len >= 2 {
            k1 ^= (data[tail_start + 1] as u32) << 8;
        }
        if tail_len >= 1 {
            k1 ^= data[tail_start] as u32;
            k1 = k1.wrapping_mul(0xcc9e2d51);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(0x1b873593);
            h1 ^= k1;
        }

        // Finalization
        h1 ^= len as u32;
        Self::fmix32(h1) as u32
    }

    fn mix(h1: u32, k1: u32) -> u32 {
        let mut k = k1;
        k = k.wrapping_mul(0xcc9e2d51);
        k = k.rotate_left(15);
        k = k.wrapping_mul(0x1b873593);

        let mut h = h1 ^ k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
        h
    }

    fn fmix32(mut h: u32) -> u32 {
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h
    }

    /// Port of next(bits) - the core XORShift algorithm
    fn next(&mut self, bits: i32) -> i32 {
        let mut next_seed = self.seed ^ (self.seed << 21);
        // >>> in Java/Scala is unsigned right shift
        next_seed ^= ((next_seed as u64) >> 35) as i64;
        next_seed ^= next_seed << 4;
        self.seed = next_seed;

        (next_seed & ((1i64 << bits) - 1)) as i32
    }

    /// Port of Java's Random.nextDouble()
    /// return (((long)(next(26)) << 27) + next(27)) / (double)(1L << 53);
    fn next_double(&mut self) -> f64 {
        let high = (self.next(26) as i64) << 27;
        let low = self.next(27) as i64;
        ((high + low) as f64) / ((1i64 << 53) as f64)
    }
}

/// Expected values from Spark's XORShiftRandom (run via spark-shell)
const SPARK_XORSHIFT_SEED_1: [f64; 5] = [
    0.6363787615254752,
    0.5993846534021868,
    0.134842710012538,
    0.07684163905460906,
    0.8539211111755448,
];

const SPARK_XORSHIFT_SEED_24: [f64; 5] = [
    0.3943255396952755,
    0.48619924381941027,
    0.2923951640552428,
    0.33335316633280176,
    0.3981939745854918,
];

fn main() {
    println!("\n=== Testing Spark XORShiftRandom Implementation ===\n");

    let mut all_match = true;

    for (seed, expected) in [
        (1i64, SPARK_XORSHIFT_SEED_1),
        (24i64, SPARK_XORSHIFT_SEED_24),
    ] {
        let mut spark_rng = SparkXorShiftRandom::new(seed);

        println!("Seed = {}:", seed);
        println!(
            "  {:>3} {:>22} {:>22} {:>8}",
            "i", "Our Implementation", "Spark Expected", "Match?"
        );
        println!("  {}", "-".repeat(60));

        for (i, &expected_val) in expected.iter().enumerate() {
            let our_val = spark_rng.next_double();
            let matches = (our_val - expected_val).abs() < 1e-15;
            if !matches {
                all_match = false;
            }

            println!(
                "  {:>3} {:>22.16} {:>22.16} {:>8}",
                i,
                our_val,
                expected_val,
                if matches { "YES" } else { "NO" }
            );
        }
        println!();
    }

    if all_match {
        println!("SUCCESS! Our implementation matches Spark's XORShiftRandom!");
    } else {
        println!("FAILED: Some values don't match.");
    }
}
