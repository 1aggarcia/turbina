//! Shared factory functions for object construction patterns used in several
//! parts of the codebase.

use std::collections::HashMap;

/// Generically convert a field map keyed by &str to one keyed by String
pub fn field_map<T, const N: usize>(
    mapping: [(&str, T); N],
) -> HashMap<String, T> {
    HashMap::<String, T>::from(mapping.map(|(k, v)| (k.to_string(), v)))
}
