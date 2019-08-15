use std;
use tf;

pub mod buffer;
pub mod error;
pub mod graph;
pub mod input;
pub mod operation;
pub mod output;
pub mod session;
pub mod tensor;
pub mod tensor_type;

pub(crate) use error::Status;
pub use error::{Error, TFError};

use buffer::{Buffer, StrBuffer};
pub use graph::Graph;
pub use input::Input;
pub use operation::Operation;
pub use output::Output;
pub use session::Session;
pub use tensor::Tensor;
pub use tensor_type::TensorType;

type Result<T> = std::result::Result<T, Error>;

/// Returns tensorflow version
///
/// ```rust
/// # use rustflow::tf_version;
/// assert!(!tf_version().is_empty());
pub fn tf_version() -> &'static str {
    unsafe { std::ffi::CStr::from_ptr(tf::TF_Version()).to_str() }.unwrap_or("")
}
