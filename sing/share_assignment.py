from .circuit import share_types


def read_share_assignment_wires(share_assignment_file):
    result = []
    with open(share_assignment_file, "r") as f:
        for line in f:
            fields = line.rstrip().split(" ")
            result.append(fields[0])

    return result


def save_share_assignment_text(tensor, share_assignment_wires, circuit, path):
    with open(path, "w") as f:
        save_share_assignment_text_f(tensor, share_assignment_wires, circuit, f)


def save_share_assignment_text_f(tensor, share_assignment_wires, circuit, f):
    for wire in share_assignment_wires:
        if wire in circuit.name_to_index_map:
            i = circuit.name_to_index_map[wire]
            x = tensor[i]
            text = share_types[x]
        else:
            # unused
            text = "y"

        f.write(f"{wire} {text}\n")
