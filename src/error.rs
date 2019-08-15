use failure::Fail;
use std::fmt;
use tf;

/// Wrapper for tensorflow `TF_Status`, ensuring its proper construction
/// and deletion.
///
/// `TF_Status` is constrained to be not null and pointing to valid `TF_Status`
/// object for any unsafe calls
///
/// The internal field is avaiable in API, but it should be never send to
/// `TF_DeleteStatus`, because of this constraint. If it happens, whole
/// `Status` should be forgot (via `mem::forget).
#[derive(Debug)]
pub(crate) struct Status(*mut tf::TF_Status);

impl Status {
    pub fn new() -> Self {
        let status = unsafe {
            // This function is guaranteed to return valid TF_NewStatus,
            // unless there are some memory issues, but there is nothing
            // possible to do with that (C++ `new TF_Status` is under the hood)
            tf::TF_NewStatus()
        };

        Self(status)
    }

    pub fn get(&mut self) -> *mut tf::TF_Status {
        self.0
    }

    fn code(&self) -> tf::TF_Code {
        unsafe { tf::TF_GetCode(self.0) }
    }

    fn message(&self) -> String {
        unsafe { std::ffi::CStr::from_ptr(tf::TF_Message(self.0)) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn to_result(&self) -> Result<(), TFError> {
        if let Some(err) = TFError::from_status(self) {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl Drop for Status {
    fn drop(&mut self) {
        unsafe { tf::TF_DeleteStatus(self.0) }
    }
}

macro_rules! def_code {
    ($($rval:ident : $cval:ident),*) => {
        #[derive(Debug, PartialEq, Eq)]
        pub enum TFCode {
            $($rval),*
        }

        impl TFCode {
            pub(crate) fn from_status(status: &Status) -> Option<Self> {
                match status.code() {
                    tf::TF_OK => None,
                    $(tf::$cval => Some(TFCode::$rval),)*
                }
            }
        }
    };
}

def_code! {
    Cancelled: TF_CANCELLED,
    Unknown: TF_UNKNOWN,
    InvalidArgument: TF_INVALID_ARGUMENT,
    DeadlineExceeded: TF_DEADLINE_EXCEEDED,
    NotFound: TF_NOT_FOUND,
    AlreadyExists: TF_ALREADY_EXISTS,
    PermissionDenied: TF_PERMISSION_DENIED,
    ResourceExhausted: TF_RESOURCE_EXHAUSTED,
    FailedPrecondition: TF_FAILED_PRECONDITION,
    Aborted: TF_ABORTED,
    OutOfRange: TF_OUT_OF_RANGE,
    Unimplemented: TF_UNIMPLEMENTED,
    Internal: TF_INTERNAL,
    Unavailable: TF_UNAVAILABLE,
    DataLoss: TF_DATA_LOSS,
    Unauthenticated: TF_UNAUTHENTICATED
}

/// Wrapped tensorflow error
#[derive(Debug, Fail, PartialEq, Eq)]
pub struct TFError {
    code: TFCode,
    message: String,
}

impl TFError {
    pub(crate) fn from_status(status: &Status) -> Option<Self> {
        Some(TFError {
            code: TFCode::from_status(status)?,
            message: status.message(),
        })
    }
}

impl fmt::Display for TFError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Any Error produced by library
#[derive(Debug, Fail, PartialEq, Eq)]
pub enum Error {
    #[fail(display = "Tensorflow error: {}", _0)]
    TFError(TFError),

    #[fail(
        display = "Tensor shape {:?} not valid for tensor data of len {}",
        shape, data_len
    )]
    InvalidShape { data_len: usize, shape: Vec<i64> },

    #[fail(display = "TF object creation failed")]
    ObjectCreationFailure,
}

impl From<TFError> for Error {
    fn from(err: TFError) -> Self {
        Error::TFError(err)
    }
}
