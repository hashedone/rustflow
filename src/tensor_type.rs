use tf;

pub unsafe trait TensorType {
    const TF_TYPE: tf::TF_DataType;
}

macro_rules! tensor_type {
    ($type:ty : $tf_type:expr) => {
        unsafe impl TensorType for $type {
            const TF_TYPE: tf::TF_DataType = $tf_type;
        }
    };
}

tensor_type!(f32: tf::TF_FLOAT);
tensor_type!(f64: tf::TF_DOUBLE);

tensor_type!(i8:  tf::TF_INT8);
tensor_type!(i16: tf::TF_INT16);
tensor_type!(i32: tf::TF_INT32);
tensor_type!(i64: tf::TF_INT64);

tensor_type!(u8:  tf::TF_UINT8);
tensor_type!(u16: tf::TF_UINT16);
tensor_type!(u32: tf::TF_UINT32);
tensor_type!(u64: tf::TF_UINT64);
