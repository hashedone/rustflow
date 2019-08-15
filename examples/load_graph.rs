use rustflow;
use std::fs;


fn main() {
    let data = fs::read_to_string("tests/data/addition.pb").unwrap();
    let _graph = rustflow::Graph::from_protobuff(&data).unwrap();
}
