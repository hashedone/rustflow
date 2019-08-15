use rustflow::Graph;

#[test]
fn loads_valid_protobuff_graph() {
    let proto = include_str!("data/addition.pb");
    Graph::from_protobuff(proto).unwrap();
}

#[test]
fn error_loadnig_invalid_protobuff_graph() {
    let proto = "invalid";
    Graph::from_protobuff(proto).map(|_| ()).unwrap_err();
}
