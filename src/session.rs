use crate::{Error, Graph, Result, Status};
use tf;

/// Thin wrapper over tensorflow session
pub struct Session(*mut tf::TF_Session);

/// Represents session which is already closed
pub struct ClosedSession(*mut tf::TF_Session);

impl Session {
    /// Function for closing session. It should be called
    /// only to get information about error while closing session,
    /// otherwise closing and deleting session is done automatically
    /// while session dropping
    pub fn close(self) -> Result<ClosedSession> {
        let mut status = Status::new();
        unsafe {
            tf::TF_CloseSession(self.0, status.get());
        }

        let result = status.to_result();
        if let Err(err) = result {
            unsafe {
                tf::TF_DeleteSession(self.0, status.get());
            }

            Err(err.into())
        } else {
            Ok(ClosedSession(self.0))
        }
    }
}

impl ClosedSession {
    /// Function for deleting session. It should be called
    /// only to get information about error while closing session,
    /// otherwise deleteing session is done automatically
    /// while session dropping
    pub fn delete(self) -> Result<()> {
        let mut status = Status::new();
        unsafe {
            tf::TF_DeleteSession(self.0, status.get());
        }

        Ok(status.to_result()?)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let mut status = Status::new();
        unsafe {
            tf::TF_CloseSession(self.0, status.get());
            tf::TF_DeleteSession(self.0, status.get());
        }
    }
}

impl Drop for ClosedSession {
    fn drop(&mut self) {
        let mut status = Status::new();
        unsafe {
            tf::TF_DeleteSession(self.0, status.get());
        }
    }
}

/// Builder for Session object. Artificial lifetime is here to
/// ensure, that SessionBuilder will not outlive graph pointer,
/// which is not owned by SessionBuilder.
pub struct SessionBuilder<'a> {
    options: *mut tf::TF_SessionOptions,
    graph: &'a Graph,
}

/// Thin wrapper over tensorflow session option for
/// building actual session object
impl<'a> SessionBuilder<'a> {
    /// Creates new session builder with associated with graph
    ///
    /// ```rust
    /// # use rustflow::Graph;
    /// # use rustflow::session::SessionBuilder;
    /// let proto = include_str!("../tests/data/addition.pb");
    /// let graph = Graph::from_protobuff(proto).unwrap();
    /// let builder = SessionBuilder::with_graph(&graph).unwrap();
    /// ```
    pub fn with_graph(graph: &'a Graph) -> Result<Self> {
        let options = unsafe { tf::TF_NewSessionOptions() };

        if options.is_null() {
            return Err(Error::ObjectCreationFailure);
        }

        Ok(SessionBuilder { options, graph })
    }

    /// Builds final session object
    ///
    /// ```rust
    /// # use rustflow::Graph;
    /// # use rustflow::session::SessionBuilder;
    /// let proto = include_str!("../tests/data/addition.pb");
    /// let graph = Graph::from_protobuff(proto).unwrap();
    /// let session = SessionBuilder::with_graph(&graph)
    ///     .unwrap()
    ///     .build()
    ///     .unwrap();
    pub fn build(self) -> Result<Session> {
        let mut status = Status::new();
        let session =
            unsafe { tf::TF_NewSession(self.graph.get_ptr(), self.options, status.get()) };

        status.to_result()?;
        Ok(Session(session))
    }
}

impl<'a> Drop for SessionBuilder<'a> {
    fn drop(&mut self) {
        unsafe { tf::TF_DeleteSessionOptions(self.options) }
    }
}
