use tf;
use crate::{Result, Error, Status};
use std::marker::PhantomData;

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
    graph: *mut tf::TF_Graph,
    phantom: PhantomData<&'a tf::TF_Graph>,
}

/// Thin wrapper over tensorflow session option for
/// building actual session object
impl<'a> SessionBuilder<'a> {
    /// Function is unsafe, because its callee responsibility to ensure,
    /// that graph is valid not-null TF_Graph object
    pub(crate) unsafe fn with_graph(graph: *mut tf::TF_Graph)
        -> Result<Self>
    {
        let options = unsafe { tf::TF_NewSessionOptions() };

        if options.is_null() {
            return Err(Error::ObjectCreationFailure);
        }

        Ok(SessionBuilder {
            options,
            graph,
            phantom: PhantomData,
        })
    }

    /// Builds final session object
    pub fn build(self) -> Result<Session> {
        let mut status = Status::new();
        let session = unsafe {
            tf::TF_NewSession(
                self.graph,
                self.options,
                status.get()
            )
        };

        status.to_result()?;
        Ok(Session(session))
    }
}

impl<'a> Drop for SessionBuilder<'a> {
    fn drop(&mut self) {
        unsafe {
            tf::TF_DeleteSessionOptions(self.options)
        }
    }
}
