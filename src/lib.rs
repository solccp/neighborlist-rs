pub mod batch;
pub mod cell;
pub mod config;
pub mod search;
pub mod single;

#[cfg(feature = "python")]
mod python_api;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
