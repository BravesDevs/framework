use std::collections::HashMap;

/// Caching memory allocator. Reuses freed buffers by size to avoid
/// repeated heap allocations in the hot training loop. On CUDA builds
/// this would wrap cudaMalloc/cudaFree; on CPU it wraps Vec<f32>.
pub struct CachingAllocator {
    free_lists: HashMap<usize, Vec<Vec<f32>>>,
    total_allocated: usize,
}

impl CachingAllocator {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
        }
    }

    pub fn alloc(&mut self, size: usize) -> Vec<f32> {
        if let Some(list) = self.free_lists.get_mut(&size) {
            if let Some(mut buf) = list.pop() {
                buf.iter_mut().for_each(|x| *x = 0.0);
                return buf;
            }
        }
        self.total_allocated += size * 4;
        vec![0.0f32; size]
    }

    pub fn dealloc(&mut self, buf: Vec<f32>) {
        let size = buf.len();
        self.free_lists.entry(size).or_default().push(buf);
    }

    pub fn allocated_bytes(&self) -> usize {
        self.total_allocated
    }
}
