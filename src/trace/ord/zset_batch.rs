use crate::{
    algebra::{AddAssignByRef, AddByRef, MonoidValue, NegByRef},
    time::AntichainRef,
    trace::{
        layers::{
            column_layer::{
                ColumnLayer, ColumnLayerBuilder, ColumnLayerConsumer, ColumnLayerCursor,
                ColumnLayerValues,
            },
            Builder as TrieBuilder, Cursor as TrieCursor, MergeBuilder, Trie, TupleBuilder,
        },
        ord::{merge_batcher::MergeBatcher, zset_index::ZSetIndex},
        Batch, BatchReader, Builder, Consumer, Cursor, Merger, ValueConsumer,
    },
    DBData, DBWeight, NumEntries,
};
use size_of::SizeOf;
use std::{
    cmp::max,
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Neg},
    rc::Rc,
};

/// An immutable collection of `(key, weight)` pairs without timing information.
#[derive(Debug, Clone, SizeOf)]
pub struct OrdZSet<K, R> {
    #[doc(hidden)]
    pub layer: ColumnLayer<K, R>,
    index: ZSetIndex<K>,
}

impl<K, R> OrdZSet<K, R> {
    pub const fn from_layer(layer: ColumnLayer<K, R>) -> Self {
        Self {
            layer,
            index: ZSetIndex::uninit(),
        }
    }

    pub fn len(&self) -> usize {
        self.layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layer.is_empty()
    }

    pub fn retain<F>(&mut self, retain: F)
    where
        F: FnMut(&K, &R) -> bool,
    {
        self.index.invalidate();
        self.layer.retain(retain);
    }

    pub fn truncate(&mut self, length: usize) {
        self.index.truncate(length);
        self.layer.truncate(length);
    }

    pub fn truncate_clone(&self, length: usize) -> Self
    where
        K: Clone,
        R: Clone,
    {
        Self::from_layer(self.layer.truncate_clone(length))
    }

    pub fn shrink_to_fit(&mut self) {
        self.layer.shrink_to_fit();
    }
}

impl<K, R> Eq for OrdZSet<K, R>
where
    K: Eq,
    R: Eq,
{
}

impl<K, R> PartialEq for OrdZSet<K, R>
where
    K: PartialEq,
    R: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.layer == other.layer
    }
}

impl<K, R> Display for OrdZSet<K, R>
where
    K: DBData,
    R: DBWeight,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "layer:\n{}",
            textwrap::indent(&self.layer.to_string(), "    ")
        )
    }
}

impl<K, R> From<ColumnLayer<K, R>> for OrdZSet<K, R> {
    fn from(layer: ColumnLayer<K, R>) -> Self {
        Self::from_layer(layer)
    }
}

impl<K, R> From<ColumnLayer<K, R>> for Rc<OrdZSet<K, R>> {
    fn from(layer: ColumnLayer<K, R>) -> Self {
        Rc::new(layer.into())
    }
}

impl<K, R> NumEntries for OrdZSet<K, R>
where
    K: DBData,
    R: DBWeight,
{
    const CONST_NUM_ENTRIES: Option<usize> = <ColumnLayer<K, R>>::CONST_NUM_ENTRIES;

    fn num_entries_shallow(&self) -> usize {
        self.layer.num_entries_shallow()
    }

    fn num_entries_deep(&self) -> usize {
        self.layer.num_entries_deep()
    }
}

impl<K, R> Default for OrdZSet<K, R> {
    fn default() -> Self {
        Self {
            layer: ColumnLayer::empty(),
            index: ZSetIndex::empty(),
        }
    }
}

impl<K, R> NegByRef for OrdZSet<K, R>
where
    K: DBData,
    R: MonoidValue + NegByRef,
{
    fn neg_by_ref(&self) -> Self {
        Self::from_layer(self.layer.neg_by_ref())
    }
}

impl<K, R> Neg for OrdZSet<K, R>
where
    K: DBData,
    R: MonoidValue + Neg<Output = R>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            layer: self.layer.neg(),
            // `ColumnLayer::neg()` doesn't invalidate `keys` in any way so we can pass on `index`
            // unaffected
            index: self.index,
        }
    }
}

// TODO: by-value merge
impl<K, R> Add<Self> for OrdZSet<K, R>
where
    K: DBData,
    R: MonoidValue,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_layer(self.layer.add(rhs.layer))
    }
}

impl<K, R> AddAssign<Self> for OrdZSet<K, R>
where
    K: DBData,
    R: MonoidValue,
{
    fn add_assign(&mut self, rhs: Self) {
        self.index.invalidate();
        self.layer.add_assign(rhs.layer);
    }
}

impl<K, R> AddAssignByRef for OrdZSet<K, R>
where
    K: DBData,
    R: MonoidValue,
{
    fn add_assign_by_ref(&mut self, rhs: &Self) {
        self.index.invalidate();
        self.layer.add_assign_by_ref(&rhs.layer);
    }
}

impl<K, R> AddByRef for OrdZSet<K, R>
where
    K: DBData,
    R: MonoidValue,
{
    fn add_by_ref(&self, rhs: &Self) -> Self {
        Self::from_layer(self.layer.add_by_ref(&rhs.layer))
    }
}

impl<K, R> BatchReader for OrdZSet<K, R>
where
    K: DBData,
    R: DBWeight,
{
    type Key = K;
    type Val = ();
    type Time = ();
    type R = R;
    type Cursor<'s> = OrdZSetCursor<'s, K, R>;
    type Consumer = OrdZSetConsumer<K, R>;

    #[inline]
    fn cursor(&self) -> Self::Cursor<'_> {
        OrdZSetCursor {
            valid: true,
            index: &self.index,
            cursor: self.layer.cursor(),
        }
    }

    #[inline]
    fn consumer(self) -> Self::Consumer {
        OrdZSetConsumer {
            consumer: ColumnLayerConsumer::from(self.layer),
        }
    }

    #[inline]
    fn key_count(&self) -> usize {
        Trie::keys(&self.layer)
    }

    #[inline]
    fn len(&self) -> usize {
        self.layer.tuples()
    }

    #[inline]
    fn lower(&self) -> AntichainRef<'_, ()> {
        AntichainRef::new(&[()])
    }

    #[inline]
    fn upper(&self) -> AntichainRef<'_, ()> {
        AntichainRef::empty()
    }
}

impl<K, R> Batch for OrdZSet<K, R>
where
    K: DBData,
    R: DBWeight,
{
    type Item = K;
    type Batcher = MergeBatcher<K, (), R, Self>;
    type Builder = OrdZSetBuilder<K, R>;
    type Merger = OrdZSetMerger<K, R>;

    fn item_from(key: K, _val: ()) -> Self::Item {
        key
    }

    fn from_keys(time: Self::Time, keys: Vec<(Self::Key, Self::R)>) -> Self {
        Self::from_tuples(time, keys)
    }

    fn begin_merge(&self, other: &Self) -> Self::Merger {
        OrdZSetMerger::new_merger(self, other)
    }

    fn recede_to(&mut self, _frontier: &()) {}

    fn empty(_time: Self::Time) -> Self {
        Self::default()
    }
}

/// State for an in-progress merge.
#[derive(SizeOf)]
pub struct OrdZSetMerger<K, R>
where
    K: DBData,
    R: DBWeight,
{
    // result that we are currently assembling.
    result: <ColumnLayer<K, R> as Trie>::MergeBuilder,
}

impl<K, R> Merger<K, (), (), R, OrdZSet<K, R>> for OrdZSetMerger<K, R>
where
    Self: SizeOf,
    K: DBData,
    R: DBWeight,
{
    fn new_merger(batch1: &OrdZSet<K, R>, batch2: &OrdZSet<K, R>) -> Self {
        Self {
            result: <<ColumnLayer<K, R> as Trie>::MergeBuilder as MergeBuilder>::with_capacity(
                &batch1.layer,
                &batch2.layer,
            ),
        }
    }

    fn done(self) -> OrdZSet<K, R> {
        OrdZSet::from_layer(self.result.done())
    }

    fn work(&mut self, source1: &OrdZSet<K, R>, source2: &OrdZSet<K, R>, fuel: &mut isize) {
        *fuel -= self
            .result
            .push_merge(source1.layer.cursor(), source2.layer.cursor()) as isize;
        *fuel = max(*fuel, 1);
    }
}

/// A cursor for navigating a single layer.
#[derive(Debug, SizeOf)]
pub struct OrdZSetCursor<'s, K, R>
where
    K: DBData,
    R: DBWeight,
{
    valid: bool,
    index: &'s ZSetIndex<K>,
    cursor: ColumnLayerCursor<'s, K, R>,
}

impl<'s, K, R> Cursor<'s, K, (), (), R> for OrdZSetCursor<'s, K, R>
where
    K: DBData,
    R: DBWeight,
{
    fn key(&self) -> &K {
        self.cursor.current_key()
    }

    fn val(&self) -> &() {
        &()
    }

    fn fold_times<F, U>(&mut self, init: U, mut fold: F) -> U
    where
        F: FnMut(U, &(), &R) -> U,
    {
        if self.cursor.valid() {
            fold(init, &(), self.cursor.current_diff())
        } else {
            init
        }
    }

    fn fold_times_through<F, U>(&mut self, _upper: &(), init: U, fold: F) -> U
    where
        F: FnMut(U, &(), &R) -> U,
    {
        self.fold_times(init, fold)
    }

    fn weight(&mut self) -> R {
        debug_assert!(&self.cursor.valid());
        self.cursor.current_diff().clone()
    }

    fn key_valid(&self) -> bool {
        self.cursor.valid()
    }

    fn val_valid(&self) -> bool {
        self.valid
    }

    fn step_key(&mut self) {
        self.cursor.step();
        self.valid = true;
    }

    fn seek_key(&mut self, key: &K) {
        // /*
        self.index.crack(self.cursor.storage().keys(), |index| {
            let current = self.cursor.position();
            // Interestingly it seems that *not* limiting the binary search to values past
            // the current cursor's position yields significantly better performance (in my
            // tests turing 84 seconds for the ldbc datagen-8_4-fb benchmark to 73 seconds
            // for a ~13% improvement)

            match index.binary_search_by(|&rhs| unsafe { key.cmp(rhs.as_ref()) }) {
                // If one of the index values is the same as the key, we can advance directly to
                // that value
                Ok(idx) => self
                    .cursor
                    .advance_to(max(idx * ZSetIndex::<K>::BUCKET_SIZE, current)),

                // Otherwise the value should lie between the `idx` and `idx + 1` buckets (the
                // first element in the layer is skipped)
                Err(idx) => {
                    self.cursor
                        .advance_to(max(idx * ZSetIndex::<K>::BUCKET_SIZE, current));
                    self.cursor.seek_key(key);
                }
            }

            self.valid = true;
        });
        // */

        // self.cursor.seek_key(key);
        // self.valid = true;
    }

    fn last_key(&mut self) -> Option<&K> {
        self.cursor.last_key().map(|(k, _)| k)
    }

    fn step_val(&mut self) {
        self.valid = false;
    }

    fn seek_val(&mut self, _val: &()) {}

    fn seek_val_with<P>(&mut self, predicate: P)
    where
        P: Fn(&()) -> bool + Clone,
    {
        if !predicate(&()) {
            self.valid = false;
        }
    }

    fn rewind_keys(&mut self) {
        self.cursor.rewind();
        self.valid = true;
    }

    fn rewind_vals(&mut self) {
        self.valid = true;
    }
}

/// A builder for creating layers from unsorted update tuples.
#[derive(SizeOf)]
pub struct OrdZSetBuilder<K, R>
where
    K: Ord,
    R: DBWeight,
{
    builder: ColumnLayerBuilder<K, R>,
}

impl<K, R> Builder<K, (), R, OrdZSet<K, R>> for OrdZSetBuilder<K, R>
where
    Self: SizeOf,
    K: DBData,
    R: DBWeight,
{
    #[inline]
    fn new_builder(_time: ()) -> Self {
        Self {
            builder: ColumnLayerBuilder::new(),
        }
    }

    #[inline]
    fn with_capacity(_time: (), capacity: usize) -> Self {
        Self {
            builder: <ColumnLayerBuilder<K, R> as TupleBuilder>::with_capacity(capacity),
        }
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.builder.reserve(additional);
    }

    #[inline]
    fn push(&mut self, (key, diff): (K, R)) {
        self.builder.push_tuple((key, diff));
    }

    #[inline(never)]
    fn done(self) -> OrdZSet<K, R> {
        OrdZSet::from_layer(self.builder.done())
    }
}

#[derive(Debug, SizeOf)]
pub struct OrdZSetConsumer<K, R> {
    consumer: ColumnLayerConsumer<K, R>,
}

impl<K, R> Consumer<K, (), R, ()> for OrdZSetConsumer<K, R> {
    type ValueConsumer<'a> = OrdZSetValueConsumer<'a, K, R>
    where
        Self: 'a;

    fn key_valid(&self) -> bool {
        self.consumer.key_valid()
    }

    fn peek_key(&self) -> &K {
        self.consumer.peek_key()
    }

    fn next_key(&mut self) -> (K, Self::ValueConsumer<'_>) {
        let (key, values) = self.consumer.next_key();
        (key, OrdZSetValueConsumer { values })
    }

    fn seek_key(&mut self, key: &K)
    where
        K: Ord,
    {
        // TODO: Utilize indices for seeking
        self.consumer.seek_key(key);
    }
}

#[derive(Debug)]
pub struct OrdZSetValueConsumer<'a, K, R> {
    values: ColumnLayerValues<'a, K, R>,
}

impl<'a, K, R> ValueConsumer<'a, (), R, ()> for OrdZSetValueConsumer<'a, K, R> {
    fn value_valid(&self) -> bool {
        self.values.value_valid()
    }

    fn next_value(&mut self) -> ((), R, ()) {
        self.values.next_value()
    }

    fn remaining_values(&self) -> usize {
        self.values.remaining_values()
    }
}
