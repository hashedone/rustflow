use tf;
use std::marker::PhantomData;


/// Thin wrapper over tensorflow input object. TF_Input keeps TF_Operation
/// object internally, so artificial lifetime is added.
pub struct Input<'a> {
    pub(crate) input: tf::TF_Input,
    _phantom: PhantomData<&'a tf::TF_Operation>,
}

impl<'a> Input<'a> {
    /// Function is unsafe, because callee has to ensure, that:
    /// 1) operation outlives created Input
    /// 2) operation has at least `index+1` inputs
    pub(crate) unsafe fn new(operation: *mut tf::TF_Operation, index: i32) -> Self {
        let input = tf::TF_Input {
            oper: operation,
            index,
        };

        Input {
            input,
            _phantom: PhantomData,
        }
    }
}
