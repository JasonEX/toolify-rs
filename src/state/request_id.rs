use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct RequestIdGenerator {
    seed: u128,
    counter: AtomicU64,
}

impl RequestIdGenerator {
    #[must_use]
    pub(crate) fn new() -> Self {
        let seed_hi = u128::from(fastrand::u64(..));
        let seed_lo = u128::from(fastrand::u64(..));
        Self {
            seed: (seed_hi << 64) | seed_lo,
            counter: AtomicU64::new(1),
        }
    }

    pub(crate) fn next_seq(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }

    #[must_use]
    pub(crate) fn request_uuid(&self, request_seq: u64) -> uuid::Uuid {
        uuid::Uuid::from_u128(self.seed ^ u128::from(request_seq))
    }
}
