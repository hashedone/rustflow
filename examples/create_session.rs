use rustflow;
use std::fs;


fn main() {
    let data = fs::read_to_string("tests/data/addition.pb").unwrap();
    let graph = rustflow::Graph::from_protobuff(&data).unwrap();
    let _session =
        rustflow::session::SessionBuilder::with_graph(&graph)
            .unwrap()
            .build()
            .unwrap();
}
