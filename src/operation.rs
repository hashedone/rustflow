use tf;


pub struct Operation(*mut tf::TF_Operation);

impl Operation {
    pub(crate) unsafe fn new(operation: *mut tf::TF_Operation) -> Self {
        Self(operation)
    }
}
