// Copyright (c) 2016 Jerome Froelich

#![no_std]
#![cfg_attr(feature = "nightly", feature(alloc, optin_builtin_traits))]

#[cfg(feature = "nightly")]
extern crate alloc;
#[cfg(feature = "hashbrown")]
extern crate hashbrown;
#[cfg(test)]
extern crate scoped_threadpool;
#[cfg(not(feature = "nightly"))]
extern crate std as alloc;
#[cfg(test)]
#[macro_use]
extern crate std;

use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;
use core::ptr;
use core::usize;

#[cfg(feature = "hashbrown")]
use hashbrown::HashMap;

use alloc::borrow::Borrow;
use alloc::boxed::Box;
#[cfg(not(feature = "hashbrown"))]
use alloc::collections::HashMap;
use alloc::vec::Vec;

// Two pointers for each of the linked list pointers (2*8) = 16
// key value = 8 bytes (u64)
// pointer to the value (usize) = 8 bytes
// TODO: figure out the actual overhead of the entry hashbrown table - 32 bytes for now
// TODO: verify the actual overhead of Vec<u8> appears to be 32 bytes to an untrained eye
static NUM_BYTES_OVERHEAD_PER_ENTRY: usize = 96;

// TODO: Right now completely arbitrary based on the expected size of head and tail nodes
static INITIALLY_USED_MEMORY: usize = 64;

static EXISTING_NODE_FOUND: u8 = 1;
static EXISTING_NODE_REMOVED_FROM_TAIL: u8 = 2;
static CLEANED_UP_ON_MEM_SIZE: u8 = 4;
static CLEANED_UP_ON_CAPACITY: u8 = 8;
static NEW_ENTRY_ADDED: u8 = 16;


// Struct used to hold a reference to a key
#[doc(hidden)]
pub struct KeyRef {
    k: *const u64,
}

impl Hash for KeyRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { (*self.k).hash(state) }
    }
}

impl PartialEq for KeyRef {
    fn eq(&self, other: &KeyRef) -> bool {
        unsafe { (*self.k).eq(&*other.k) }
    }
}

impl Eq for KeyRef {}

#[cfg(feature = "nightly")]
#[doc(hidden)]
pub auto trait NotKeyRef {}

#[cfg(feature = "nightly")]
impl<K> ! NotKeyRef for KeyRef<K> {}

#[cfg(feature = "nightly")]
impl<K, D> Borrow<D> for KeyRef<K>
    where
        K: Borrow<D>,
        D: NotKeyRef + ?Sized,
{
    fn borrow(&self) -> &D {
        unsafe { (&*self.k) }.borrow()
    }
}

#[cfg(not(feature = "nightly"))]
impl Borrow<u64> for KeyRef {
    fn borrow(&self) -> &u64 {
        unsafe { (&*self.k) }
    }
}

// Struct used to hold a key value pair. Also contains references to previous and next entries
// so we can maintain the entries in a linked list ordered by their use.
struct LruEntry {
    key: u64,
    val: Vec<u8>,
    expires: u64,
    prev: *mut LruEntry,
    next: *mut LruEntry,
}

impl LruEntry {
    fn new(key: u64, val: Vec<u8>, expires: u64) -> Self {
        LruEntry {
            key,
            val,
            expires,
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
        }
    }
}

#[cfg(feature = "hashbrown")]
pub type DefaultHasher = hashbrown::hash_map::DefaultHashBuilder;
#[cfg(not(feature = "hashbrown"))]
pub type DefaultHasher = alloc::collections::hash_map::RandomState;

/// An LRU Cache
pub struct LruCache<S = DefaultHasher> {
    map: HashMap<KeyRef, Box<LruEntry>, S>,
    cap: usize,

    // head and tail are sigil nodes to faciliate inserting entries
    head: *mut LruEntry,
    tail: *mut LruEntry,

    pub used_memory: usize,
    pub lru_memory: usize,
}

impl LruCache {
    /// Creates a new LRU Cache that holds leaves `leave_free_ram` free RAM on the system.
    pub fn new(free_ram: usize, leave_free_ram: usize, average_element_size: usize) -> LruCache {
        let capacity = get_capacity(free_ram, leave_free_ram, average_element_size);

        LruCache::construct(free_ram, leave_free_ram, capacity, HashMap::with_capacity(capacity))
    }
}

fn get_capacity(free_memory: usize, leave_free_ram: usize, average_element_size: usize) -> usize {
//    let lru_memory = (sys.get_free_memory() * 1024) - leave_free_ram as u64;
    let lru_memory = free_memory - leave_free_ram;

    // Add 1 element buffer to capacity - prevent world-stopping re-hashing
    (lru_memory / (NUM_BYTES_OVERHEAD_PER_ENTRY + average_element_size) + 1) as usize
}

impl<S: BuildHasher> LruCache<S> {
    /// Creates a new LRU Cache that holds at most `cap` items and
    /// uses the providedash builder to hash keys.
    pub fn with_hasher(free_ram: usize, leave_free_ram: usize, average_element_size: usize, hash_builder: S) -> LruCache<S> {
        let capacity = get_capacity(free_ram, leave_free_ram, average_element_size);

        LruCache::construct(free_ram, leave_free_ram, capacity, HashMap::with_capacity_and_hasher(capacity, hash_builder))
    }

    /// Creates a new LRU Cache with the given capacity.
    fn construct(free_ram: usize, leave_free_ram: usize, cap: usize, map: HashMap<KeyRef, Box<LruEntry>, S>) -> LruCache<S> {
        let lru_memory = free_ram - leave_free_ram;

        if lru_memory <= 0 {
            panic!("Not enough memory to run an LRU cache");
        }

        // NB: The compiler warns that cache does not need to be marked as mutable if we
        // declare it as such since we only mutate it inside the unsafe block.
        let cache = LruCache {
            map,
            cap,
            head: unsafe { Box::into_raw(Box::new(mem::MaybeUninit::uninit().assume_init())) },
            tail: unsafe { Box::into_raw(Box::new(mem::MaybeUninit::uninit().assume_init())) },
            used_memory: INITIALLY_USED_MEMORY,
            lru_memory: lru_memory as usize,
        };

        unsafe {
            (*cache.head).next = cache.tail;
            (*cache.tail).prev = cache.head;
        }

        cache
    }

    /// Puts a key-value pair into cache. If the key already exists in the cache, then it updates
    /// the key's value and returns the old value. Otherwise, `None` is returned.
    pub fn put(&mut self, k: u64, v: Vec<u8>) -> u8 {
        self.put_until(k, v, 0)
    }

    pub fn put_until(&mut self, k: u64, mut v: Vec<u8>, expires: u64) -> u8 {
        let mut existing_size = 0;
        let mut existing_node_key: u64 = 0;
        let mut debug_code = NEW_ENTRY_ADDED;

        // NOTE: query is still needed in case of parallel insert of the same value
        let node_ptr = self.map.get_mut(&KeyRef { k: &k }).map(|node| {
            existing_size = node.val.len() + NUM_BYTES_OVERHEAD_PER_ENTRY;
            existing_node_key = node.key;
            debug_code = EXISTING_NODE_FOUND;
            let node_ptr: *mut LruEntry = &mut **node;
            node_ptr
        });

        let size = v.len() + NUM_BYTES_OVERHEAD_PER_ENTRY;

        match node_ptr {
            Some(node_ptr) => {
                // if the key is already in the cache just update its value and move it to the
                // front of the list

                let mut removed_existing_entry = false;

                // New version of the item with the same key is larger than previous
                // - can happen when threads are re-cached (with more content).
                while self.used_memory + (size - existing_size) > self.lru_memory {
                    match self.remove_last() {
                        Some(removed_entry) => {
                            if removed_entry.key == existing_node_key {
                                // Just removed the element that is being replaced
                                // hence existing size
                                removed_existing_entry = true;
                                existing_size = 0;
                                debug_code = EXISTING_NODE_REMOVED_FROM_TAIL;
                            }
                        }
                        None => {
                            // No more elements to remove (shouldn't ever happen)
                        }
                    }
                }

                if removed_existing_entry {
                    // Existing matching entry has been removed, so need to create
                    // a new Entry
                    let mut node = Box::new(LruEntry::new(k, v, expires));

                    let node_ptr: *mut LruEntry = &mut *node;
                    self.attach(node_ptr);

                    let keyref = unsafe { &(*node_ptr).key };
                    self.map.insert(KeyRef { k: keyref }, node);
                } else {
                    unsafe {
                        self.used_memory = self.used_memory -
                            (*node_ptr).val.len() - NUM_BYTES_OVERHEAD_PER_ENTRY;
                        (*node_ptr).expires = expires;
                        mem::swap(&mut v, &mut (*node_ptr).val)
                    }
                    self.detach(node_ptr);
                    self.attach(node_ptr);
                }

                self.used_memory = self.used_memory + size;

                debug_code
            }
            None => {
                while self.used_memory + size > self.lru_memory {
                    self.remove_last();
                    debug_code = CLEANED_UP_ON_MEM_SIZE;
                }

                let mut node = if self.len() == self.cap() {
                    // if the cache is full, remove the last entry so we can use it for the new key
                    let old_key = KeyRef {
                        k: unsafe { &(*(*self.tail).prev).key },
                    };
                    let mut old_node = self.map.remove(&old_key).unwrap();
                    self.used_memory = self.used_memory -
                        old_node.val.len() - NUM_BYTES_OVERHEAD_PER_ENTRY;

                    old_node.key = k;
                    old_node.val = v;
                    old_node.expires = expires;

                    let node_ptr: *mut LruEntry = &mut *old_node;
                    self.detach(node_ptr);

                    debug_code = CLEANED_UP_ON_CAPACITY;

                    old_node
                } else {
                    // if the cache is not full allocate a new LruEntry

                    Box::new(LruEntry::new(k, v, expires))
                };

                let node_ptr: *mut LruEntry = &mut *node;
                self.attach(node_ptr);

                let keyref = unsafe { &(*node_ptr).key };
                self.map.insert(KeyRef { k: keyref }, node);

                self.used_memory = self.used_memory + size;

                debug_code
            }
        }
    }

    /// Returns a reference to the value of the key in the cache or `None` if it is not
    /// present in the cache. Moves the key to the head of the LRU list if it exists.
    pub fn get<'a, Q>(&'a mut self, k: &Q) -> Option<&'a Vec<u8>>
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        self.get_since(k, 0)
    }

    pub fn get_since<'a, Q>(&'a mut self, k: &Q, seconds_since_epoc: u64) -> Option<&'a Vec<u8>>
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        if let Some(node) = self.map.get_mut(k) {
            if seconds_since_epoc != 0
                && seconds_since_epoc > node.expires {
                return None;
            }

            let node_ptr: *mut LruEntry = &mut **node;

            self.detach(node_ptr);
            self.attach(node_ptr);

            Some(unsafe { &(*node_ptr).val })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value of the key in the cache or `None` if it
    /// is not present in the cache. Moves the key to the head of the LRU list if it exists.
    pub fn get_mut<'a, Q>(&'a mut self, k: &Q) -> Option<&'a mut Vec<u8>>
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        if let Some(node) = self.map.get_mut(k) {
            let node_ptr: *mut LruEntry = &mut **node;

            self.detach(node_ptr);
            self.attach(node_ptr);

            Some(unsafe { &mut (*node_ptr).val })
        } else {
            None
        }
    }

    /// Returns a reference to the value corresponding to the key in the cache or `None` if it is
    /// not present in the cache. Unlike `get`, `peek` does not update the LRU list so the key's
    /// position will be unchanged.
    pub fn peek<'a, Q>(&'a self, k: &Q) -> Option<&'a Vec<u8>>
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        match self.map.get(k) {
            None => None,
            Some(node) => Some(&node.val),
        }
    }

    /// Returns a mutable reference to the value corresponding to the key in the cache or `None`
    /// if it is not present in the cache. Unlike `get_mut`, `peek_mut` does not update the LRU
    /// list so the key's position will be unchanged.
    pub fn peek_mut<'a, Q>(&'a mut self, k: &Q) -> Option<&'a mut Vec<u8>>
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        match self.map.get_mut(k) {
            None => None,
            Some(node) => Some(&mut node.val),
        }
    }

    /// Returns the value corresponding to the least recently used item or `None` if the
    /// cache is empty. Like `peek`, `peek_lru` does not update the LRU list so the item's
    /// position will be unchanged.
    pub fn peek_lru<'a>(&'a self) -> Option<(&'a u64, &'a Vec<u8>)> {
        if self.len() == 0 {
            return None;
        }

        let (key, val);
        unsafe {
            let node = (*self.tail).prev;
            key = &(*node).key;
            val = &(*node).val;
        }

        Some((key, val))
    }

    /// Returns a bool indicating whether the given key is in the cache. Does not update the
    /// LRU list.
    pub fn contains<Q>(&self, k: &Q) -> bool
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(k)
    }

    /// Removes and returns the value corresponding to the key from the cache or
    /// `None` if it does not exist.
    pub fn pop<Q>(&mut self, k: &Q) -> Option<Vec<u8>>
        where
            KeyRef: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        match self.map.remove(&k) {
            None => None,
            Some(mut old_node) => {
                let node_ptr: *mut LruEntry = &mut *old_node;
                self.detach(node_ptr);
                Some(old_node.val)
            }
        }
    }

    /// Removes and returns the key and value corresponding to the least recently
    /// used item or `None` if the cache is empty.
    pub fn pop_lru(&mut self) -> Option<(u64, Vec<u8>)> {
        let node = self.remove_last()?;
        // N.B.: Can't destructure directly because of https://github.com/rust-lang/rust/issues/28536
        let node = *node;
        let LruEntry { key, val, .. } = node;
        Some((key, val))
    }

    /// Returns the number of key-value pairs that are currently in the the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns a bool indicating whether the cache is empty or not.
    pub fn is_empty(&self) -> bool {
        self.map.len() == 0
    }

    /// Returns the maximum number of key-value pairs the cache can hold.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Resizes the cache. If the new capacity is smaller than the size of the current
    /// cache any entries past the new capacity are discarded.
    pub fn resize(&mut self, cap: usize) {
        // return early if capacity doesn't change
        if cap == self.cap {
            return;
        }

        while self.map.len() > cap {
            self.remove_last();
        }
        self.map.shrink_to_fit();

        self.cap = cap;
    }

    /// Clears the contents of the cache.
    pub fn clear(&mut self) {
        loop {
            match self.remove_last() {
                Some(_) => (),
                None => break,
            }
        }
    }

    /// An iterator visiting all entries in order. The iterator element type is `(&'a K, &'a V)`.
    pub fn iter<'a>(&'a self) -> Iter<'a> {
        Iter {
            len: self.len(),
            ptr: unsafe { (*self.head).next },
            end: unsafe { (*self.tail).prev },
            phantom: PhantomData,
        }
    }

    /// An iterator visiting all entries in order, giving a mutable reference on V.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a> {
        IterMut {
            len: self.len(),
            ptr: unsafe { (*self.head).next },
            end: unsafe { (*self.tail).prev },
            phantom: PhantomData,
        }
    }

    pub fn shorten_by(&mut self, num_to_shorten_by: usize) {
        let mut num_nodes_to_remove = num_to_shorten_by;
        if num_nodes_to_remove > self.len() {
            num_nodes_to_remove = self.len();
        }

        let original_len = self.len();

        for _ in 0..num_nodes_to_remove {
            self.remove_last();
        }

        self.cap = self.cap - num_to_shorten_by;

//        1000 total removed 100
//        100 - 100 / 1000 = 0.9
//        10000 total RAM
//        10000 * 0.9 = 9000
        let percent_remaining = 1.0 - (num_to_shorten_by as f64) / (original_len as f64);
        self.lru_memory = ((self.lru_memory - INITIALLY_USED_MEMORY) as f64 * percent_remaining) as usize
            + INITIALLY_USED_MEMORY;
    }

    fn remove_last(&mut self) -> Option<Box<LruEntry>> {
        let prev;
        unsafe { prev = (*self.tail).prev }
        if prev != self.head {
            let old_key = KeyRef {
                k: unsafe { &(*(*self.tail).prev).key },
            };
            let mut old_node = self.map.remove(&old_key).unwrap();

            let old_node_bytes = old_node.val.len()
                + NUM_BYTES_OVERHEAD_PER_ENTRY;

            self.used_memory = self.used_memory - old_node_bytes;

            let node_ptr: *mut LruEntry = &mut *old_node;
            self.detach(node_ptr);
            Some(old_node)
        } else {
            None
        }
    }

    fn detach(&mut self, node: *mut LruEntry) {
        unsafe {
            (*(*node).prev).next = (*node).next;
            (*(*node).next).prev = (*node).prev;
        }
    }

    fn attach(&mut self, node: *mut LruEntry) {
        unsafe {
            (*node).next = (*self.head).next;
            (*node).prev = self.head;
            (*self.head).next = node;
            (*(*node).next).prev = node;
        }
    }
}

impl<S> Drop for LruCache<S> {
    fn drop(&mut self) {
        // Prevent compiler from trying to drop the un-initialized fields key and val in head
        // and tail
        unsafe {
            let head = *Box::from_raw(self.head);
            let tail = *Box::from_raw(self.tail);

            let LruEntry {
                key: head_key,
                val: head_val,
                ..
            } = head;
            let LruEntry {
                key: tail_key,
                val: tail_val,
                ..
            } = tail;

            mem::forget(head_key);
            mem::forget(head_val);
            mem::forget(tail_key);
            mem::forget(tail_val);
        }
    }
}

impl<'a, S: BuildHasher> IntoIterator for &'a LruCache<S> {
    type Item = (&'a u64, &'a Vec<u8>);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl<'a, S: BuildHasher> IntoIterator for &'a mut LruCache<S> {
    type Item = (&'a u64, &'a mut Vec<u8>);
    type IntoIter = IterMut<'a>;

    fn into_iter(self) -> IterMut<'a> {
        self.iter_mut()
    }
}

// The compiler does not automatically derive Send and Sync for LruCache because it contains
// raw pointers. The raw pointers are safely encapsulated by LruCache though so we can
// implement Send and Sync for it below.
unsafe impl Send for LruCache {}

unsafe impl Sync for LruCache {}

impl fmt::Debug for LruCache {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LruCache")
            .field("len", &self.len())
            .field("cap", &self.cap())
            .finish()
    }
}

/// An iterator over the entries of a `LruCache`.
///
/// This `struct` is created by the [`iter`] method on [`LruCache`][`LruCache`]. See its
/// documentation for more.
///
/// [`iter`]: struct.LruCache.html#method.iter
/// [`LruCache`]: struct.LruCache.html
pub struct Iter<'a> {
    len: usize,

    ptr: *const LruEntry,
    end: *const LruEntry,

    phantom: PhantomData<&'a u64>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a u64, &'a Vec<u8>);

    fn next(&mut self) -> Option<(&'a u64, &'a Vec<u8>)> {
        if self.len == 0 {
            return None;
        }

        let key = unsafe { &(*self.ptr).key };
        let val = unsafe { &(*self.ptr).val };

        self.len -= 1;
        self.ptr = unsafe { (*self.ptr).next };

        Some((key, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    fn count(self) -> usize {
        self.len
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<(&'a u64, &'a Vec<u8>)> {
        if self.len == 0 {
            return None;
        }

        let key = unsafe { &(*self.end).key };
        let val = unsafe { &(*self.end).val };

        self.len -= 1;
        self.end = unsafe { (*self.end).prev };

        Some((key, val))
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}

impl<'a> Clone for Iter<'a> {
    fn clone(&self) -> Iter<'a> {
        Iter {
            len: self.len,
            ptr: self.ptr,
            end: self.end,
            phantom: PhantomData,
        }
    }
}

// The compiler does not automatically derive Send and Sync for Iter because it contains
// raw pointers.
unsafe impl<'a> Send for Iter<'a> {}

unsafe impl<'a> Sync for Iter<'a> {}

/// An iterator over mutables entries of a `LruCache`.
///
/// This `struct` is created by the [`iter_mut`] method on [`LruCache`][`LruCache`]. See its
/// documentation for more.
///
/// [`iter_mut`]: struct.LruCache.html#method.iter_mut
/// [`LruCache`]: struct.LruCache.html
pub struct IterMut<'a> {
    len: usize,

    ptr: *mut LruEntry,
    end: *mut LruEntry,

    phantom: PhantomData<&'a u64>,
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (&'a u64, &'a mut Vec<u8>);

    fn next(&mut self) -> Option<(&'a u64, &'a mut Vec<u8>)> {
        if self.len == 0 {
            return None;
        }

        let key = unsafe { &(*self.ptr).key };
        let val = unsafe { &mut (*self.ptr).val };

        self.len -= 1;
        self.ptr = unsafe { (*self.ptr).next };

        Some((key, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    fn count(self) -> usize {
        self.len
    }
}

impl<'a> DoubleEndedIterator for IterMut<'a> {
    fn next_back(&mut self) -> Option<(&'a u64, &'a mut Vec<u8>)> {
        if self.len == 0 {
            return None;
        }

        let key = unsafe { &(*self.end).key };
        let val = unsafe { &mut (*self.end).val };

        self.len -= 1;
        self.end = unsafe { (*self.end).prev };

        Some((key, val))
    }
}

impl<'a> ExactSizeIterator for IterMut<'a> {}

impl<'a> FusedIterator for IterMut<'a> {}

// The compiler does not automatically derive Send and Sync for Iter because it contains
// raw pointers.
unsafe impl<'a> Send for IterMut<'a> {}

unsafe impl<'a> Sync for IterMut<'a> {}


#[cfg(test)]
mod tests {
    extern crate sysinfo;

    use alloc::vec::Vec;
    use INITIALLY_USED_MEMORY;

    use super::LruCache;

    use self::sysinfo::{System, SystemExt};

    struct TestData {
        average_element_size: usize,
        free_ram: usize,
        leave_free_ram: usize,
        one: Vec<u8>,
        two: Vec<u8>,
        two_v2: Vec<u8>,
        three: Vec<u8>,
        sb1: Vec<u8>,
        sb2: Vec<u8>,
        sb3: Vec<u8>,
        sb4: Vec<u8>,
        sb5: Vec<u8>,
    }

    fn get_test_data(capacity: u32, average_element_size: usize) -> TestData {
        let sys = System::new();
        let free_ram_kb = sys.get_free_memory();
        let free_ram = (free_ram_kb * 1024) as usize;
        let cache_ram_size = (super::NUM_BYTES_OVERHEAD_PER_ENTRY + average_element_size) * capacity as usize
            + INITIALLY_USED_MEMORY;
        let leave_free_ram = (free_ram - cache_ram_size) as usize;

        TestData {
            average_element_size,
            free_ram,
            leave_free_ram,
            one: vec![1u8],
            two: vec![2u8],
            two_v2: vec![2u8, 2u8],
            three: vec![3u8],
            sb1: vec![1u8, 1u8, 1u8, 1u8],
            sb2: vec![2u8, 2u8, 2u8, 2u8],
            sb3: vec![3u8, 3u8, 3u8, 3u8],
            sb4: vec![4u8, 4u8, 4u8, 4u8],
            sb5: vec![5u8, 5u8, 5u8, 5u8],
        }
    }

    fn get_test_cache(test_data: &TestData) -> LruCache {
        LruCache::new(test_data.free_ram, test_data.leave_free_ram, test_data.average_element_size)
    }

    #[test]
    fn test_capacity_computation() {
        let test_data = get_test_data(2, 1);
        let cache = get_test_cache(&test_data);

        assert_eq!(cache.cap(), 3);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_put_new_entry_added() {
        let test_data = get_test_data(2, 1);
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put(1, test_data.one.clone()), super::NEW_ENTRY_ADDED);

//        println!("Used Memory:   {}", cache.used_memory);
//        println!("LRU Memory:    {}", cache.lru_memory);
//        let size = test_data.two.len() + super::NUM_BYTES_OVERHEAD_PER_ENTRY;
//        println!("size:          {}", size);
        assert_eq!(cache.put(2, test_data.two.clone()), super::NEW_ENTRY_ADDED);

        assert_eq!(cache.get(&1).unwrap(), &test_data.one);
        assert_eq!(cache.get(&2).unwrap(), &test_data.two);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_put_existing_node_found() {
        let test_data = get_test_data(2, 1);
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put(2, test_data.two.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(2, test_data.two.clone()), super::EXISTING_NODE_FOUND);
        assert_eq!(cache.get(&2).unwrap(), &test_data.two);
        assert_eq!(cache.len(), 1);

//        println!("Used Memory:   {}", cache.used_memory);
//        println!("LRU Memory:    {}", cache.lru_memory);
//        let size = test_data.one.len() + super::NUM_BYTES_OVERHEAD_PER_ENTRY;
//        println!("size:          {}", size);
        assert_eq!(cache.put(1, test_data.one.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_put_existing_node_removed_from_tail() {
        let test_data = get_test_data(2, 1);
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put(2, test_data.two.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(1, test_data.one.clone()), super::NEW_ENTRY_ADDED);
//        println!("Used Memory:   {}", cache.used_memory);
//        println!("LRU Memory:    {}", cache.lru_memory);
//        let size = test_data.two_v2.len() + super::NUM_BYTES_OVERHEAD_PER_ENTRY;
//        let existing_size = test_data.two.len() + super::NUM_BYTES_OVERHEAD_PER_ENTRY;
//        println!("size:          {}", size);
//        println!("existing_size: {}", existing_size);
//        println!("used_memory + (size - existing_size): {}", cache.used_memory + (size - existing_size));
//        println!("used_memory - removed_entry.val.len() - NUM_BYTES_OVERHEAD_PER_ENTRY: {}", cache.used_memory - 1 - super::NUM_BYTES_OVERHEAD_PER_ENTRY);
        assert_eq!(cache.put(2, test_data.two_v2.clone()), super::EXISTING_NODE_REMOVED_FROM_TAIL);
        assert_eq!(cache.get(&2).unwrap(), &test_data.two_v2);
        assert!(cache.get(&1).is_none());
//        println!("Used Memory:   {}", cache.used_memory);
//        println!("LRU Memory:    {}", cache.lru_memory);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_put_cleaned_up_on_mem_size() {
        let test_data = get_test_data(2, 1);
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put(1, test_data.one.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(2, test_data.two.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(3, test_data.three.clone()), super::CLEANED_UP_ON_MEM_SIZE);

        assert_eq!(cache.get(&3).unwrap(), &test_data.three);
        assert_eq!(cache.get(&2).unwrap(), &test_data.two);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_put_cleaned_up_on_capacity() {
        let mut test_data = get_test_data(3, 1);
        test_data.average_element_size = 1 + super::NUM_BYTES_OVERHEAD_PER_ENTRY * 3 / 2;
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put(1, test_data.one.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(2, test_data.two.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(3, test_data.three.clone()), super::CLEANED_UP_ON_CAPACITY);

        assert_eq!(cache.get(&3).unwrap(), &test_data.three);
        assert_eq!(cache.get(&2).unwrap(), &test_data.two);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_shorten_by() {
        let test_data = get_test_data(5, 4);
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put(1, test_data.sb1.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(2, test_data.sb2.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(3, test_data.sb3.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(4, test_data.sb4.clone()), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.put(5, test_data.sb5.clone()), super::NEW_ENTRY_ADDED);

//        println!("lru_memory: {}", cache.lru_memory);
        cache.shorten_by(2);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.cap(), 4);
        assert_eq!(cache.lru_memory, INITIALLY_USED_MEMORY
            + (super::NUM_BYTES_OVERHEAD_PER_ENTRY + 4) * 3);
    }

    #[test]
    fn test_put_until_get_since() {
        let test_data = get_test_data(2, 1);
        let mut cache = get_test_cache(&test_data);

        assert_eq!(cache.put_until(1, test_data.one.clone(), 2), super::NEW_ENTRY_ADDED);
        assert_eq!(cache.get_since(&1, 2).unwrap(), &test_data.one);
        assert!(cache.get_since(&1, 3).is_none());

        assert_eq!(cache.put_until(1, test_data.one.clone(), 3), super::EXISTING_NODE_FOUND);
        assert!(cache.get_since(&1, 4).is_none());
        assert_eq!(cache.get_since(&1, 3).unwrap(), &test_data.one);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.cap(), 3);
    }
}
