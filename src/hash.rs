//! Hashing utilities.

use std::hash::{Hash, Hasher};
use xxhash_rust::xxh3::Xxh3;

const SEED: u64 = 0x7f95_ef85_be33_c337u64;

/// The default hasher used to shard records across workers
pub fn default_hasher() -> Xxh3 {
    Xxh3::with_seed(SEED)
}

/// Default hashing function used to shard records across workers.
pub fn default_hash<T: Hash>(x: &T) -> u64 {
    let mut hasher = default_hasher();
    x.hash(&mut hasher);
    hasher.finish()
}
