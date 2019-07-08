use rustflow::Graph;

fn main() {
    let proto = include_str!("../tests/data/addition.pb");
    let graph = Graph::from_protobuff(proto).unwrap();

    for op in graph.operations() {
        println!("{}: {} ({} inputs, {} outputs)", op.name(), op.op_type(), op.inputs().count(), op.outputs().count());
    }
}
