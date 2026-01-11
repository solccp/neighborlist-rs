use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum size of the stack-allocated scratchpad. Fixed at compile time.
pub const MAX_STACK_SIZE: usize = 1024;

// Default heuristic values for this machine.
// Can be overridden at compile time via environment variables.
const DEFAULT_BRUTE_FORCE_THRESHOLD: usize = 1000;
const DEFAULT_PARALLEL_THRESHOLD: usize = 300;
const DEFAULT_STACK_THRESHOLD: usize = 1000;

static BRUTE_FORCE_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_BRUTE_FORCE_THRESHOLD);
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);
static STACK_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_STACK_THRESHOLD);

pub fn get_brute_force_threshold() -> usize {
    BRUTE_FORCE_THRESHOLD.load(Ordering::Relaxed)
}

pub fn set_brute_force_threshold(val: usize) {
    BRUTE_FORCE_THRESHOLD.store(val, Ordering::Relaxed);
}

pub fn get_parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

pub fn set_parallel_threshold(val: usize) {
    PARALLEL_THRESHOLD.store(val, Ordering::Relaxed);
}

pub fn get_stack_threshold() -> usize {
    STACK_THRESHOLD.load(Ordering::Relaxed)
}

pub fn set_stack_threshold(val: usize) {
    // Only use stack if requested threshold is within MAX_STACK_SIZE
    STACK_THRESHOLD.store(val.min(MAX_STACK_SIZE), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_accessors() {
        let original_bf = get_brute_force_threshold();
        let original_par = get_parallel_threshold();
        let original_stack = get_stack_threshold();

        set_brute_force_threshold(123);
        assert_eq!(get_brute_force_threshold(), 123);

        set_parallel_threshold(456);
        assert_eq!(get_parallel_threshold(), 456);

        set_stack_threshold(789);
        assert_eq!(get_stack_threshold(), 789);

        // Restore defaults
        set_brute_force_threshold(original_bf);
        set_parallel_threshold(original_par);
        set_stack_threshold(original_stack);
    }
}