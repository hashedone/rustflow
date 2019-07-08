use tf;
use std::marker::PhantomData;


/// Thin wrapper over tensorflow output object. TF_Output keeps TF_Operation
/// object internally, so artificial lifetime is added.
pub struct Output<'a> {
    pub(crate) output: tf::TF_Output,
    _phantom: PhantomData<&'a tf::TF_Operation>,
}

impl<'a> Output<'a> {
    /// Function is unsafe, because callee has to ensure, that:
    /// 1) operation outlives created Output
    /// 2) operation has at least `index+1` outputs
    pub(crate) unsafe fn new(operation: *mut tf::TF_Operation, index: i32) -> Self {
        let output = tf::TF_Output {
            oper: operation,
            index,
        };

        Output {
            output,
            _phantom: PhantomData,
        }
    }
}
