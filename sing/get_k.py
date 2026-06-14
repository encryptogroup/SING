import torch
import networkx as nx

from .circuit import ops

node_feature_size = len(ops)

# ops which count toward depth_bool according to Silph
depth_bool_ops = [
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
    "XOR",  # sic
    "MUX",
    "NOT",  # sic
    "DIV",
    "SHL",
    "LSHR",
    "SELECT",
    "STORE",
]

# ops which count toward depth_arith according to Silph
depth_arith_ops = [
    "MUL",
]

# ops which count toward num_bool according to Silph
num_bool_ops = [
    "EQ",
    "REM",
    "AND",
    "OR",
    "MUX",
    "NOT",
    "DIV",
    "SELECT",  # TODO scale by length
    "STORE",  # TODO scale by length
]

# ops which count toward num_mul according to Silph
num_mul_ops = [
    "MUL",
]


def get_depth(x, edge_index, significant_ops):
    # 1 if op is a significant op, 0 otherwise
    is_significant_op = torch.zeros((node_feature_size))
    significant_op_idx = [ops.index(op) for op in significant_ops]
    is_significant_op[significant_op_idx] = 1

    node_significant = is_significant_op[torch.argmax(x, dim=1)]

    # edge weight == 1 if coming from a significant node
    edge_weight = node_significant[edge_index[0]]

    edges = torch.cat((edge_index, torch.unsqueeze(edge_weight, 0)))

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges.t())

    return nx.dag_longest_path_length(graph)


def get_num(x, significant_ops):
    # 1 if op is a significant op, 0 otherwise
    is_significant_op = torch.zeros((node_feature_size))
    significant_op_idx = [ops.index(op) for op in significant_ops]
    is_significant_op[significant_op_idx] = 1

    node_significant = is_significant_op[torch.argmax(x, dim=1)]

    return torch.sum(node_significant)


def get_k(x, edge_index):
    depth_arith = get_depth(x, edge_index, depth_arith_ops)
    depth_bool = get_depth(x, edge_index, depth_bool_ops)

    num_mul = get_num(x, num_mul_ops)
    num_bool = get_num(x, num_bool_ops)

    return {
        "a": min(1, depth_arith / num_mul),
        "b": min(1, depth_bool / num_bool),
    }
