use rustflow;
use std::fs;


fn main() {
    let data = fs::read_to_string("../tests/data/addition.pb").unwrap();
    let _session = rustflow::Graph::from_protobuff(&data).unwrap()
        .session_builder().unwrap()
        .build().unwrap();
}
