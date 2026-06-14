ops = [
    "ADD",
    "SUB",
    "MUL",
    "EQ",
    "GT",
    "LT",
    "GE",
    "LE",
    "REM",
    "AND",
    "OR",
    "XOR",
    "CONS",
    "MUX",
    "NOT",
    "DIV",
    "SHL",
    "LSHR",
    "IN",
    "OUT",
    "SELECT",
    "STORE",
]
share_types = ["a", "b", "y"]
binops = [
    "ADD",
    "SUB",
    "MUL",
    "EQ",
    "GT",
    "LT",
    "GE",
    "LE",
    "REM",
    "DIV",
    "AND",
    "OR",
    "XOR",
]


def encode_node_features_slow(op):
    result = [0] * len(ops)
    index = ops.index(op)
    result[index] = 1
    return result


encode_node_features = {op: encode_node_features_slow(op) for op in ops}


def encode_y(share_type):
    index = share_types.index(share_type)
    return index
