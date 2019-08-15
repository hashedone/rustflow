use tf;
use std::{ffi, iter, ops};
use crate::{Buffer, StrBuffer, Result, Status, Operation};


/// Thin wrapper over tensorflow graph
pub struct Graph(*mut tf::TF_Graph);

impl Graph {
    /// Loads data from tensorflow serialized protobuf string, for example
    /// model builded via Python API
    ///
    ///
    /// ```rust
    /// # use rustflow::Graph;
    /// let proto = include_str!("../tests/data/addition.pb");
    /// Graph::from_protobuff(proto).unwrap();
    ///
    /// let proto = "invalid";
    /// Graph::from_protobuff(proto).map(|_| ()).unwrap_err();
    /// ```
    pub fn from_protobuff(data: &str) -> Result<OwnedGraph> {
        let buffer = StrBuffer::new(data);
        let graph = unsafe { tf::TF_NewGraph() };
        let graph = Self(graph);
        let mut status = Status::new();

        unsafe {
            let import_options = tf::TF_NewImportGraphDefOptions();

            tf::TF_GraphImportGraphDef(
                graph.0,
                buffer.buffer(),
                import_options,
                status.get()
            );
            tf::TF_DeleteImportGraphDefOptions(import_options);

            status.to_result()?;
            Ok(OwnedGraph(graph))
        }
    }

    /// Returns operation with given name in graph
    ///
    /// ```rust
    /// # use rustflow::Graph;
    /// # let proto = include_str!("../tests/data/addition.pb");
    /// # let graph = Graph::from_protobuff(proto).unwrap();
    /// graph.operation_by_name("x").unwrap();
    /// # assert!(graph.operation_by_name("a").is_none());
    /// ```
    pub fn operation_by_name<'a>(&'a self, name: &str) -> Option<Operation<'a>> {
        let name = ffi::CString::new(name).ok()?;
        let operation = unsafe {
            let op = tf::TF_GraphOperationByName(self.0, name.as_ptr());
            if op.is_null() {
                return None;
            }
            Operation::new(op)
        };
        Some(operation)
    }

    /// Returns iterator over all graph operations
    ///
    /// ```rust
    /// # use rustflow::Graph;
    /// let proto = include_str!("../tests/data/addition.pb");
    /// let graph = Graph::from_protobuff(proto).unwrap();
    /// let ops = graph.operations();
    /// # assert_eq!(4, ops.count());
    /// ```
    pub fn operations<'a>(&'a self) -> impl Iterator<Item=Operation<'a>> {
        let mut pos = 0usize;
        iter::from_fn(move || {
            let operation = unsafe {
                let op = tf::TF_GraphNextOperation(self.0, &mut pos as *mut _);
                if op.is_null() {
                    return None;
                }
                Operation::new(op)
            };
            Some(operation)
        })
    }

    /// Returns internal graph pointer. This is unsafe, because
    /// if callee have to make sure he would not corrupt the
    /// object under pointer.
    pub(crate) unsafe fn get_ptr(&self) -> *mut tf::TF_Graph {
        self.0
    }
}

/// Graph object which will be released when dropped
pub struct OwnedGraph(Graph);

impl ops::Deref for OwnedGraph {
    type Target = Graph;

    fn deref(&self) -> &Graph { &self.0 }
}

impl Drop for OwnedGraph {
    fn drop(&mut self) {
        unsafe { tf::TF_DeleteGraph((self.0).0) }
    }
}

