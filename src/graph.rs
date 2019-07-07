use tf;

use crate::{Buffer, StrBuffer, Result, Status};


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
    pub fn from_protobuff(data: &str) -> Result<Graph> {
        let buffer = StrBuffer::new(data);
        let graph = unsafe { tf::TF_NewGraph() };
        let graph = Self(graph);
        let mut status = Status::new();

        unsafe {
            let import_options = tf::TF_NewImportGraphDefOptions();

            tf::TF_GraphImportGraphDef(graph.0, buffer.buffer(), import_options, status.get());
            tf::TF_DeleteImportGraphDefOptions(import_options);

            status.to_result()?;
            Ok(graph)
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { tf::TF_DeleteGraph(self.0) }
    }
}

