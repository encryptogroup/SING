import json

import torch

from .circuit import ops, share_types

node_feature_size = len(ops)
num_classes = len(share_types)

op_to_cost_op_map = {
    "ADD": "add",
    "SUB": "sub",
    "MUL": "mul",
    "EQ": "eq",
    "GT": "gt",
    "LT": "lt",
    "GE": "ge",
    "LE": "le",
    "REM": "rem",
    "AND": "and",
    "OR": "or",
    "XOR": "xor",
    # "CONS": None,
    "MUX": "mux",
    "NOT": "ne",
    "DIV": "div",
    "SHL": "shl",
    "LSHR": "shr",
    # "IN": None,
    # "OUT": None,
    "SELECT": "select",
    "STORE": "store",
}


def get_cost_opt(share_type, cost_info):
    if not share_type in cost_info:
        return None

    return cost_info[share_type]["32"]


def get_cost(op, costs):
    return costs[op]["32"]


def get_depth(share_type, cost_info):
    return cost_info["depth"][share_type]


class CostModel:
    def __init__(self, opa_cost_file_path=None, smart=True):
        self.conversions = {}
        self.ops_cost_no_depth = {}
        self.ops_cost_depth = {}
        self.ops = {}
        self.invalid = None
        self.op_cost = None
        self.conv_cost = None
        self.smart = smart

        if opa_cost_file_path is not None:
            with open(opa_cost_file_path) as f:
                self.from_opa_cost_file(f)

    def to(self, device):
        self.invalid = self.invalid.to(device)
        self.op_cost = self.op_cost.to(device)
        self.conv_cost = self.conv_cost.to(device)

        return self

    def update_tensors(self):
        # 0 if combination of op and share_type is valid, 1 otherwise
        #
        # default: all combinations invalid
        invalid = torch.ones((node_feature_size, num_classes), dtype=torch.float)
        op_cost = torch.zeros((node_feature_size, num_classes), dtype=torch.float)

        conv_cost = torch.zeros((num_classes, num_classes), dtype=torch.float)

        # IN and CONS
        invalid[[ops.index("IN"), ops.index("CONS")], :] = 0
        if self.smart:
            op_cost[[ops.index("IN"), ops.index("CONS")]] = torch.tensor(
                [0.1, 0.12, 0.11]
            )
        else:
            op_cost[[ops.index("IN"), ops.index("CONS")], :] = 0

        # OUT
        invalid[ops.index("OUT"), :] = 0
        op_cost[ops.index("OUT"), :] = 0

        for op, cost_op in op_to_cost_op_map.items():
            cost_info = self.ops[cost_op]
            for share_type, cost in cost_info.items():
                op_idx = ops.index(op)
                share_type_idx = share_types.index(share_type)

                invalid[op_idx, share_type_idx] = 0
                op_cost[op_idx, share_type_idx] = cost

        for conv, cost in self.conversions.items():
            from_share_type, to_share_type = conv
            from_idx = share_types.index(from_share_type)
            to_idx = share_types.index(to_share_type)
            conv_cost[from_idx, to_idx] = cost

        self.invalid = invalid
        self.op_cost = op_cost
        self.conv_cost = conv_cost

    def from_opa_cost_file(self, f):
        costs = json.load(f)

        for from_share_type in share_types:
            for to_share_type in share_types:
                if from_share_type == to_share_type:
                    continue

                cost = get_cost(f"{from_share_type}2{to_share_type}", costs)
                self.conversions[(from_share_type, to_share_type)] = cost

        for op, cost_info in costs.items():
            if "2" in op:
                continue
            if "depth" in op:
                continue

            for share_type in share_types:
                cost = get_cost_opt(share_type, cost_info)
                if cost is not None:
                    cost_depth = 0
                    if share_type != "y":
                        d = get_depth(share_type, cost_info)
                        cost_depth += d * get_depth(share_type, costs)

                    if not op in self.ops_cost_no_depth:
                        self.ops_cost_no_depth[op] = {}
                    self.ops_cost_no_depth[op][share_type] = cost

                    if not op in self.ops_cost_depth:
                        self.ops_cost_depth[op] = {}
                    self.ops_cost_depth[op][share_type] = cost_depth

                    if not op in self.ops:
                        self.ops[op] = {}
                    self.ops[op][share_type] = cost + cost_depth

        self.update_tensors()

    def load_k(self, k):
        for op, cost_info in self.ops.items():
            for share_type, cost in cost_info.items():
                cost = self.ops_cost_no_depth[op][share_type]

                cost_depth = 0
                if share_type != "y":
                    cost_depth += (
                        k.get(share_type, 1) * self.ops_cost_depth[op][share_type]
                    )

                self.ops[op][share_type] = cost + cost_depth

        self.update_tensors()
