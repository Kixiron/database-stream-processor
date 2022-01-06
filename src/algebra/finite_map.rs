/*
MIT License
SPDX-License-Identifier: MIT

Copyright (c) 2021 VMware, Inc
*/

//! This module implements groups that are finite maps
//! from values to a group.

use super::GroupValue;
use super::*;
use std::collections::hash_map::Entry;
use std::collections::{hash_map, HashMap};
use std::fmt::{Display, Formatter, Result};
use std::hash::Hash;

////////////////////////////////////////////////////////
/// Finite map trait.
///
/// A finite map maps arbitrary values (comparable for equality)
/// to values in a group.  It has finite support: it is non-zero
/// only for a finite number of values.
///
/// `DataType` - Type of values stored in finite map.
/// `ResultType` - Type of results.
pub trait FiniteMap<DataType, ResultType>:
    GroupValue + IntoIterator<Item = (DataType, ResultType)> + FromIterator<(DataType, ResultType)>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    /// Find the value associated to the specified key
    fn lookup(&self, key: &DataType) -> ResultType;
    /// Return the set of values that are mapped to non-zero values.
    // FIXME: the return type is wrong, the result should be a more abstract iterator.
    fn support(&self) -> hash_map::Keys<'_, DataType, ResultType>;
    /// The size of the support: number of elements for which the map does not return zero.
    fn support_size(&self) -> usize;
    /// Increase the value associated to `key` by the specified `value`
    fn increment(&mut self, key: &DataType, value: &ResultType);
    /// Create a map containing a singleton value.
    fn singleton(key: DataType, value: ResultType) -> Self;
    /// Apply map to every 'key' in the support of this map and generate a new map.
    //  TODO: it would be nice for this to return a trait instead of a type.
    fn map<F, ConvertedDataType>(&self, mapper: F) -> FiniteHashMap<ConvertedDataType, ResultType>
    where
        F: Fn(DataType) -> ConvertedDataType,
        ConvertedDataType: Clone + Hash + Eq + 'static;
}

#[derive(Debug, Clone)]
pub struct FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
{
    // Unfortunately I cannot just implement these traits for
    // HashMap since they conflict with some existing traits.
    // We maintain the invariant that the keys (and only these keys)
    // that have non-zero values are in this map.
    pub(super) value: HashMap<DataType, ResultType>,
}

impl<DataType, ResultType> FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    /// Allocate an empty FiniteHashMap
    pub fn new() -> Self {
        FiniteHashMap::default()
    }
    /// Allocate an empty FiniteHashMap that is expected to hold 'size' values.
    pub fn with_capacity(size: usize) -> Self {
        FiniteHashMap::<DataType, ResultType> {
            value: HashMap::with_capacity(size),
        }
    }
}

impl<DataType, ResultType> IntoIterator for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    type Item = (DataType, ResultType);
    type IntoIter = std::collections::hash_map::IntoIter<DataType, ResultType>;

    fn into_iter(self) -> Self::IntoIter {
        self.value.into_iter()
    }
}

impl<DataType, ResultType> FromIterator<(DataType, ResultType)>
    for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (DataType, ResultType)>,
    {
        let mut result = FiniteHashMap::new();
        for (k, v) in iter {
            result.increment(&k, &v);
        }
        result
    }
}

impl<DataType, ResultType> FiniteMap<DataType, ResultType> for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn singleton(key: DataType, value: ResultType) -> Self {
        let mut result = Self::with_capacity(1);
        result.value.insert(key, value);
        result
    }

    fn lookup(&self, key: &DataType) -> ResultType {
        let val = self.value.get(key);
        match val {
            Some(w) => w.clone(),
            None => ResultType::zero(),
        }
    }

    fn support<'a>(&self) -> hash_map::Keys<'_, DataType, ResultType> {
        self.value.keys()
    }

    fn support_size(&self) -> usize {
        self.value.len()
    }

    fn increment(&mut self, key: &DataType, value: &ResultType) {
        if value.is_zero() {
            return;
        }
        // TODO: the HashMap API does not support avoiding this clone.
        // This has been a known issue since 2015: https://github.com/rust-lang/rust/issues/56167
        // We should use a different implementation or API if one becomes available.
        let e = self.value.entry(key.clone());
        match e {
            Entry::Vacant(ve) => {
                ve.insert(value.clone());
            }
            Entry::Occupied(mut oe) => {
                let w = oe.get().add_by_ref(value);
                if w.is_zero() {
                    oe.remove_entry();
                } else {
                    oe.insert(w);
                };
            }
        };
    }

    fn map<F, ConvertedDataType>(&self, mapper: F) -> FiniteHashMap<ConvertedDataType, ResultType>
    where
        F: Fn(DataType) -> ConvertedDataType,
        ConvertedDataType: Clone + Hash + Eq + 'static,
    {
        self.clone()
            .into_iter()
            .map(|(k, v)| (mapper(k), v))
            .collect()
    }
}

impl<DataType, ResultType> Default for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
{
    fn default() -> Self {
        FiniteHashMap::<DataType, ResultType> {
            value: HashMap::default(),
        }
    }
}

impl<DataType, ResultType> AddByRef for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn add_by_ref(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, v) in &other.value {
            // TODO: unfortunately there is no way to avoid this k.clone() currently.
            // See also note on 'insert' below.
            let entry = result.value.entry(k.clone());
            match entry {
                Entry::Vacant(e) => {
                    e.insert(v.clone());
                }
                Entry::Occupied(mut e) => {
                    let w = e.get().add_by_ref(v);
                    if w.is_zero() {
                        e.remove_entry();
                    } else {
                        e.insert(w);
                    };
                }
            }
        }
        result
    }
}

impl<DataType, ResultType> AddAssignByRef for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn add_assign_by_ref(&mut self, other: &Self) {
        for (k, v) in &other.value {
            // TODO: unfortunately there is no way to avoid this clone.
            let entry = self.value.entry(k.clone());
            match entry {
                Entry::Vacant(e) => {
                    e.insert(v.clone());
                }
                Entry::Occupied(mut e) => {
                    let w = e.get().add_by_ref(v);
                    if w.is_zero() {
                        e.remove_entry();
                    } else {
                        e.insert(w);
                    };
                }
            }
        }
    }
}

impl<DataType, ResultType> HasZero for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn zero() -> Self {
        FiniteHashMap::default()
    }

    fn is_zero(&self) -> bool {
        self.value.is_empty()
    }
}

impl<DataType, ResultType> NegByRef for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn neg_by_ref(&self) -> Self {
        let mut result = self.clone();
        for val in result.value.values_mut() {
            *val = val.clone().neg_by_ref();
        }
        result
    }
}

impl<DataType, ResultType> PartialEq for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl<DataType, ResultType> Eq for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static,
    ResultType: GroupValue,
{
}

/// This class knows how to display a FiniteMap to a string, but only
/// if the map keys support comparison
impl<DataType, ResultType> Display for FiniteHashMap<DataType, ResultType>
where
    DataType: Clone + Hash + Eq + 'static + Display + Ord,
    ResultType: GroupValue + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut vec: Vec<DataType> = self.support().cloned().collect();
        vec.sort_by(DataType::cmp);
        write!(f, "{{")?;

        let mut first = true;
        for k in vec {
            if !first {
                write!(f, ",")?;
            } else {
                first = false;
            }
            let val = self.lookup(&k);
            write!(f, "{}", k)?;
            write!(f, "=>")?;
            write!(f, "{}", val)?;
        }
        write!(f, "}}")
    }
}

#[test]
fn hashmap_tests() {
    let mut z = FiniteHashMap::<i64, i64>::with_capacity(5);
    assert_eq!(0, z.support_size());
    assert_eq!("{}", z.to_string());
    assert_eq!(0, z.lookup(&0)); // not present -> 0
    assert_eq!(z, FiniteHashMap::<i64, i64>::zero());
    assert!(z.is_zero());
    let z2 = FiniteHashMap::<i64, i64>::new();
    assert_eq!(z, z2);

    let z3 = FiniteHashMap::singleton(3, 4);
    assert_eq!("{3=>4}", z3.to_string());

    z.increment(&0, &1);
    assert_eq!(1, z.support_size());
    assert_eq!("{0=>1}", z.to_string());
    assert_eq!(1, z.lookup(&0));
    assert_eq!(0, z.lookup(&1));
    assert_ne!(z, FiniteHashMap::<i64, i64>::zero());
    assert_eq!(false, z.is_zero());

    z.increment(&2, &0);
    assert_eq!(1, z.support_size());
    assert_eq!("{0=>1}", z.to_string());

    z.increment(&1, &-1);
    assert_eq!(2, z.support_size());
    assert_eq!("{0=>1,1=>-1}", z.to_string());

    z.increment(&-1, &1);
    assert_eq!(3, z.support_size());
    assert_eq!("{-1=>1,0=>1,1=>-1}", z.to_string());

    let d = z.neg_by_ref();
    assert_eq!(3, d.support_size());
    assert_eq!("{-1=>-1,0=>-1,1=>1}", d.to_string());
    assert_ne!(d, z);

    let i = d.clone().into_iter().collect::<FiniteHashMap<i64, i64>>();
    assert_eq!(i, d);

    z.increment(&1, &1);
    assert_eq!(2, z.support_size());
    assert_eq!("{-1=>1,0=>1}", z.to_string());

    let mut z2 = z.add_by_ref(&z);
    assert_eq!(2, z2.support_size());
    assert_eq!("{-1=>2,0=>2}", z2.to_string());

    z2.add_assign_by_ref(&z);
    assert_eq!(2, z2.support_size());
    assert_eq!("{-1=>3,0=>3}", z2.to_string());

    let z3 = z2.map(|_x| 0);
    assert_eq!(1, z3.support_size());
    assert_eq!("{0=>6}", z3.to_string());
}
