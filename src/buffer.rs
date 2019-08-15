use std::{self, marker::PhantomData};
use tf;

/// Trait for any object which can be used as TF Buffer.
pub(crate) trait Buffer {
    fn buffer(&self) -> &tf::TF_Buffer;
}

/// Buffer with borrowed string data
pub struct StrBuffer<'a> {
    buffer: *mut tf::TF_Buffer,
    phantom: PhantomData<&'a str>,
}

impl<'a> StrBuffer<'a> {
    /// Creates new buffer borrowing given str
    ///
    /// ```rust
    /// # use rustflow::buffer::StrBuffer;
    /// let buffer = StrBuffer::new("data");
    /// ```
    pub fn new(data: &'a str) -> Self {
        let buffer = unsafe { tf::TF_NewBuffer() };

        Self::update_buffer(buffer, data);

        StrBuffer {
            buffer,
            phantom: PhantomData,
        }
    }

    /// Creates new buffer copying given str
    /// into tensorflow memory
    ///
    /// ```rust
    /// # use rustflow::buffer::StrBuffer;
    /// let buffer = StrBuffer::new_internal("data");
    /// ```
    pub fn new_internal(data: &str) -> StrBuffer<'static> {
        let buffer = unsafe {
            tf::TF_NewBufferFromString(data.as_ptr() as *const std::ffi::c_void, data.len())
        };

        StrBuffer {
            buffer,
            phantom: PhantomData,
        }
    }

    fn update_buffer(buffer: *mut tf::TF_Buffer, data: &str) {
        let bytes = data.as_bytes();
        unsafe {
            (*buffer).data = bytes.as_ptr() as *const std::ffi::c_void;
            (*buffer).length = bytes.len();
        }
    }
}

impl<'a> Buffer for StrBuffer<'a> {
    fn buffer(&self) -> &tf::TF_Buffer {
        unsafe { &*self.buffer }
    }
}

impl<'a> Drop for StrBuffer<'a> {
    fn drop(&mut self) {
        unsafe { tf::TF_DeleteBuffer(self.buffer) }
    }
}
