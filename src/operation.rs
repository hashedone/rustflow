use crate::{Input, Output};
use std::marker::PhantomData;
use tf;

/// Thin wrapper over tensroflow operation pointer. TF_Operation
/// objects are not managed by its own, instead they are managed by
/// their partentss, so thats its why addtitional artificial lifetime
/// is added
pub struct Operation<'a> {
    pub(crate) operation: *mut tf::TF_Operation,
    _phantom: PhantomData<&'a tf::TF_Operation>,
}

impl<'a> Operation<'a> {
    /// This is unsafe, because its calee who has to ensure, that
    /// operation is valid TF_Operation object which outlives 'a
    pub(crate) unsafe fn new(operation: *mut tf::TF_Operation) -> Self {
        Operation {
            operation,
            _phantom: PhantomData,
        }
    }

    /// ```rust
    /// # use rustflow::Graph;
    /// # let proto = include_str!("../tests/data/addition.pb");
    /// # let graph = Graph::from_protobuff(proto).unwrap();
    /// let op = graph.operation_by_name("x").unwrap();
    /// assert_eq!("x", op.name());
    /// ```
    pub fn name(&self) -> &str {
        unsafe { std::ffi::CStr::from_ptr(tf::TF_OperationName(self.operation)).to_str() }
            .unwrap_or("")
    }

    /// ```rust
    /// # use rustflow::Graph;
    /// # let proto = include_str!("../tests/data/addition.pb");
    /// # let graph = Graph::from_protobuff(proto).unwrap();
    /// let op = graph.operation_by_name("x").unwrap();
    /// assert_eq!("Placeholder", op.op_type());
    /// ```
    pub fn op_type(&self) -> &str {
        unsafe { std::ffi::CStr::from_ptr(tf::TF_OperationOpType(self.operation)).to_str() }
            .unwrap_or("")
    }

    /// ```rust
    /// # use rustflow::Graph;
    /// # let proto = include_str!("../tests/data/addition.pb");
    /// # let graph = Graph::from_protobuff(proto).unwrap();
    /// let op = graph.operation_by_name("x").unwrap();
    /// op.op_type();
    /// ```
    pub fn device(&self) -> &str {
        unsafe { std::ffi::CStr::from_ptr(tf::TF_OperationDevice(self.operation)).to_str() }
            .unwrap_or("")
    }

    /// ```rust
    /// # use rustflow::Graph;
    /// # let proto = include_str!("../tests/data/addition.pb");
    /// # let graph = Graph::from_protobuff(proto).unwrap();
    /// let op = graph.operation_by_name("x").unwrap();
    /// assert_eq!(1, op.outputs().count());
    /// ```
    pub fn outputs(&self) -> impl Iterator<Item = Output<'a>> {
        let cnt = unsafe { tf::TF_OperationNumOutputs(self.operation) };
        let op = self.operation;
        (0..cnt)
            .map(move |idx| unsafe { Output::new(op, idx) })
    }

    /// ```rust
    /// # use rustflow::Graph;
    /// # let proto = include_str!("../tests/data/addition.pb");
    /// # let graph = Graph::from_protobuff(proto).unwrap();
    /// let op = graph.operation_by_name("x").unwrap();
    /// assert_eq!(0, op.inputs().count());
    /// ```
    pub fn inputs(&self) -> impl Iterator<Item = Input<'a>> {
        let cnt = unsafe { tf::TF_OperationNumInputs(self.operation) };
        let op = self.operation;
        (0..cnt)
            .map(move |idx| unsafe { Input::new(op, idx) })
    }
}
