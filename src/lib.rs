use tf;
use std;


pub mod error;
pub mod buffer;
pub mod graph;
pub mod tensor;
pub mod tensor_type;
pub mod output;
pub mod input;
pub mod operation;
pub mod session;

pub use error::{TFError, Error};
pub(crate) use error::Status;

use buffer::{Buffer, StrBuffer};
pub use graph::Graph;
pub use tensor::Tensor;
pub use tensor_type::TensorType;
pub use output::Output;
pub use input::Input;
pub use operation::Operation;

type Result<T> = std::result::Result<T, Error>;

pub fn tf_version() -> &'static str {
    unsafe {
        std::ffi::CStr::from_ptr(tf::TF_Version())
            .to_str()
    }.unwrap_or("")
}

