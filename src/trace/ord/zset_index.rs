use size_of::SizeOf;
use std::{cell::RefCell, ptr::NonNull};

#[derive(Debug, Clone, SizeOf)]
pub struct ZSetIndex<K> {
    pub(super) index: RefCell<Option<Vec<NonNull<K>>>>,
}

impl<K> ZSetIndex<K> {
    pub const BUCKET_SIZE: usize = 1 << 13;

    pub const fn uninit() -> Self {
        Self {
            index: RefCell::new(None),
        }
    }

    pub const fn empty() -> Self {
        Self {
            index: RefCell::new(Some(Vec::new())),
        }
    }

    pub fn invalidate(&mut self) {
        *self.index.get_mut() = None;
    }

    pub fn crack<F, T>(&self, keys: &[K], with: F) -> T
    where
        F: FnOnce(&[NonNull<K>]) -> T,
    {
        let mut index = self.index.borrow_mut();
        let cracked = index.get_or_insert_with(
            #[cold]
            || {
                // FIXME: Should this be `(layer.len() / Self::BUCKET_SIZE) - 1`?
                let mut index = Vec::with_capacity(keys.len() / Self::BUCKET_SIZE);
                for key in keys.iter().step_by(Self::BUCKET_SIZE).skip(1) {
                    index.push(NonNull::from(key));
                }

                index
            },
        );
        with(cracked)
    }

    pub fn truncate(&mut self, length: usize) {
        if let Some(index) = self.index.get_mut().as_mut() {
            // FIXME: Is the subtraction correct here?
            index.truncate((length / Self::BUCKET_SIZE).saturating_sub(1));
        }
    }
}

unsafe impl<K: Send> Send for ZSetIndex<K> {}
