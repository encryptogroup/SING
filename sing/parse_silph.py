from .circuit import ops, binops, share_types, encode_node_features, encode_y

import line_profiler


class Circuit:
    def __init__(self):
        self.next_index = 0
        self.name_to_index_map = {}
        self.nodes = []
        self.edge_sources = []
        self.edge_targets = []

    @line_profiler.profile
    def add_from_file(self, f):
        for line in f:
            fields = line.rstrip().split(" ")
            num_inputs = int(fields[0])
            num_outputs = int(fields[1])

            input_wires = fields[2 : 2 + num_inputs]
            output_wires = fields[2 + num_inputs : 2 + num_inputs + num_outputs]

            op = fields[-1]

            if op in binops:
                assert len(input_wires) == 2
                assert len(output_wires) == 1
            elif op == "CONS":
                assert len(output_wires) == 1
            elif op == "MUX":
                assert len(input_wires) % 2 == 1
                assert len(output_wires) == len(input_wires) // 2
            elif op == "NOT":
                assert len(input_wires) == 1
                assert len(output_wires) == 1
            elif op in ["SHL", "LSHR"]:
                # second input wire is constant shift value
                assert len(input_wires) == 2
                assert len(output_wires) == 1
            elif op == "SELECT":
                assert len(input_wires) > 1
                assert len(output_wires) == 1
            elif op == "STORE":
                assert len(input_wires) > 2
                assert len(output_wires) == len(input_wires) - 2
            elif op == "IN":
                # inputs could be unused in circuit
                if len(output_wires) == 0:
                    continue  # gates are not added to nodes (and thus edges)

                assert len(output_wires) == 1
            elif op == "OUT":
                assert len(input_wires) == 1
                assert len(output_wires) == 0
                continue  # OUT gates are not added to nodes (and thus edges)
            else:
                raise Exception("Invalid operation")

            if op in ["SHL", "LSHR"]:
                # second input wire is constant shift value
                input_wires = input_wires[0:1]
            elif op in ["CONS", "IN"]:
                # special because no (real) input wires
                input_wires = []

            # connect input wires to this node
            wires = [self.name_to_index_map[s] for s in input_wires]
            wires = list(dict.fromkeys(wires))  # remove duplicates, retaining order
            self.edge_sources += wires
            self.edge_targets += [self.next_index] * len(wires)

            for o in output_wires:
                self.name_to_index_map[o] = self.next_index

            node_features = encode_node_features[op]
            self.nodes += [node_features]
            self.next_index += 1

            assert len(self.nodes) == self.next_index

    def add_bytecode_and_const(
        self,
        bytecode_file_path,
        constant_input_file_path,
    ):
        with open(constant_input_file_path, "r") as f:
            self.add_from_file(f)

        with open(bytecode_file_path, "r") as f:
            self.add_from_file(f)


def share_assignment_from_file(f, circuit):
    share_assignment = [None] * len(circuit.nodes)
    unused = []

    added = 0
    for line in f:
        fields = line.rstrip().split(" ")
        assert len(fields) == 2

        name = fields[0]
        assignment = fields[1]

        # some shares may have been optimized away
        if name in circuit.name_to_index_map:
            index = circuit.name_to_index_map[name]
            encoded = encode_y(assignment)
            if share_assignment[index] is None:
                share_assignment[index] = encoded
                added += 1
            else:
                assert share_assignment[index] == encoded
        else:
            unused.append(name)

    assert added == len(circuit.nodes)

    return share_assignment, unused


def parse_silph(
    bytecode_file_path,
    constant_input_file_path,
    share_assignment_file_path,
):
    circuit = Circuit()
    circuit.add_bytecode_and_const(
        bytecode_file_path,
        constant_input_file_path,
    )

    with open(share_assignment_file_path) as f:
        share_assignment, unused = share_assignment_from_file(f, circuit)
        assert len(share_assignment) == len(circuit.nodes)

    return circuit, share_assignment, unused
