use crate::{Error, Result, TensorType};
use std::{self, mem, ops, slice};
use tf;

/// Internally allocated Tensor
pub struct Tensor<T: 'static> {
    // Unsafe code assumes, this is always valild TF_Tensor object
    pub(crate) tensor: *mut tf::TF_Tensor,

    // Usize would be more natural choice here, but Tensorflow
    // uses i64 internally
    shape: Vec<i64>,

    data: &'static mut [T],
}

impl<T> Tensor<T> {
    /// Returns shape of tensor
    ///
    ///
    ///```rust
    /// # use rustflow::Tensor;
    /// assert_eq!(&[2, 2], Tensor::from_slice(&[2, 2], &[1, 2, 3, 4]).unwrap().shape());
    ///```
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
}

impl<T: TensorType> Tensor<T> {
    /// Function is unsafe, because returned tensor is uninitialized. `Tensor::data` field has to
    /// be filled with proper values before future use
    unsafe fn new_uninitialized(shape: &[i64]) -> Result<Self> {
        let len: i64 = shape.iter().product();
        let tensor = tf::TF_AllocateTensor(
            T::TF_TYPE,
            shape.as_ptr(),
            shape.len() as i32,
            mem::size_of::<T>() * len as usize,
        );

        if tensor.is_null() {
            return Err(Error::ObjectCreationFailure);
        }

        let data = slice::from_raw_parts_mut(tf::TF_TensorData(tensor) as _, len as usize);

        Ok(Tensor {
            tensor,
            shape: shape.to_vec(),
            data,
        })
    }
}

impl<T: TensorType + Copy> Tensor<T> {
    /// Tensor size must match given data length, otherwise `Error::InvalidShape` is returned.
    /// If tensor creation would fail for any other reason (some internal Tensorflow error),
    /// `Error::TensorCreationFailure` is returned.
    ///
    ///
    /// ```rust
    /// # use rustflow::Tensor;
    /// Tensor::from_slice(&[2, 2], &[1, 2, 3, 4]).unwrap();
    /// Tensor::from_slice(&[2, 2], &[1, 2, 3]).map(|_| ()).unwrap_err();
    /// ```
    pub fn from_slice(shape: &[i64], data: &[T]) -> Result<Self> {
        let len: i64 = shape.iter().product();
        if data.len() != len as usize {
            return Err(Error::InvalidShape {
                data_len: data.len(),
                shape: shape.to_vec(),
            });
        }

        let tensor = unsafe { Self::new_uninitialized(shape) }?;
        tensor.data.copy_from_slice(data);

        Ok(tensor)
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        unsafe { tf::TF_DeleteTensor(self.tensor) }
    }
}

impl<T> ops::Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.data
    }
}

impl<T> ops::DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.data
    }
}
