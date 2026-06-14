import os
import os.path as osp

import argparse
import random

import torch
import torch.nn.functional as F

from tqdm import trange

version = "v2"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
    help="Output directory",
)
parser.add_argument(
    "--budget",
    type=int,
    default=100,
    help="Maximum number of nodes to generate",
)
parser.add_argument(
    "--max-depth",
    type=int,
    default=2,
    help="Max for-loop depth",
)
parser.add_argument(
    "--num-generate",
    type=int,
    default=100,
    help="Number of circuits to generate",
)
args = parser.parse_args()


def new_variable(variables, param=False):
    name = f"v{len(variables)}"
    assert name not in variables

    prob = 0.9 if param else 0.02
    if random.random() < prob:
        array_len = random.randint(3, 32)
        var = (name, array_len)
    else:
        var = (name, 0)

    variables.append(var)
    return var


def var_decl(variable):
    name, array_len = variable

    if array_len == 0:
        return name
    else:
        return f"{name}[{array_len}]"


def var_use(variable, variables):
    name, array_len = variable

    if array_len == 0:
        return name
    else:
        non_array_variables = list(filter(lambda x: x[1] == 0, variables))

        if len(non_array_variables) > 0 and random.random() < 0.5:
            index, _ = random.choice(non_array_variables)
        else:
            index = random.randrange(0, array_len)

        return f"{name}[{index}]"


def generate_for(budget, current_depth, variables, constants, multiplier):
    num_iterations = random.randint(3, 10)
    i = f"i_{current_depth}"

    result = []
    result.append(f"for (int {i} = 0; {i} < {num_iterations}; {i}++) {{")

    result += generate_compound(
        budget,
        current_depth + 1,
        variables.copy(),
        constants + [(i, 0)],
        multiplier * num_iterations,
    )

    result.append("}")

    return result


comparisons = [
    ("==", 1),
    ("!=", 0.4),
    (">", 1),
    ("<", 1),
    (">=", 0.4),
    ("<=", 0.4),
]
comparisons_logits = torch.tensor([y for x, y in comparisons])
comparisons_probs = F.softmax(comparisons_logits, dim=0)


def generate_if(budget, current_depth, variables, constants, multiplier):
    assert budget[0] >= multiplier
    budget[0] -= multiplier

    comparison, _ = comparisons[torch.multinomial(comparisons_probs, 1).item()]
    lhs = random.choice(variables)
    rhs = random.choice(variables)

    lhs = var_use(lhs, variables)
    rhs = var_use(rhs, variables)
    condition = f"{lhs} {comparison} {rhs}"

    result = []
    result.append(f"if ({condition}) {{")

    result += generate_compound(
        budget, current_depth + 1, variables.copy(), constants, multiplier
    )

    result.append("}")

    return result


def get_dest(variables):
    if random.random() > 0.5:
        dest = random.choice(variables)

        return False, dest
    else:
        dest = new_variable(variables)

        return True, dest


def generate_assignment(budget, current_depth, variables, constants, multiplier):
    assert budget[0] >= multiplier
    budget[0] -= multiplier

    val = random.choice(variables)
    new, dest = get_dest(variables)
    result = [f"int {var_decl(dest)};"] if new else []

    dest = var_use(dest, variables)
    val = var_use(val, variables)

    result.append(f"{dest} = {val};")
    return result


binops = [
    ("+", 5),
    ("-", 4),
    ("*", 3),
    ("/", 0.2),
    ("%", 0.2),
    ("&", 0.2),
    ("|", 0.2),
    ("^", 0.2),
    ("<<", 0.1),
    (">>", 0.1),
]
binops_logits = torch.tensor([y for x, y in binops])
binops_probs = F.softmax(binops_logits, dim=0)


def generate_binop(budget, current_depth, variables, constants, multiplier):
    assert budget[0] >= multiplier
    budget[0] -= multiplier

    binop, _ = binops[torch.multinomial(binops_probs, 1).item()]

    lhs = random.choice(variables)

    if binop in ["<<", ">>"]:
        rhs = random.choice(constants)
    elif binop in ["&", "|", "^"]:
        rhs = random.choice(variables)
    else:
        rhs = random.choice(variables + constants)

    new, dest = get_dest(variables)
    result = [f"int {var_decl(dest)};"] if new else []

    dest = var_use(dest, variables)
    lhs = var_use(lhs, variables)
    rhs = var_use(rhs, variables)

    result.append(f"{dest} = {lhs} {binop} {rhs};")
    return result


rules = [
    (generate_for, 0.001),
    (generate_if, 0.00005),
    (generate_assignment, 3.0),
    (generate_binop, 3.0),
]
rules_logits = torch.tensor([y for x, y in rules])
rules_probs = F.softmax(rules_logits, dim=0)

rules_no_depth = [(x, y) for x, y in rules if x not in [generate_for, generate_if]]
rules_logits_no_depth = torch.tensor([y for x, y in rules_no_depth])
rules_probs_no_depth = F.softmax(rules_logits_no_depth, dim=0)


def generate_statement(budget, current_depth, variables, constants, multiplier):
    assert current_depth <= args.max_depth

    if budget[0] < multiplier:
        return []

    if current_depth == args.max_depth:
        rule = torch.multinomial(rules_probs_no_depth, 1).item()
        rule_fn, _ = rules_no_depth[rule]
    else:
        rule = torch.multinomial(rules_probs, 1).item()
        rule_fn, _ = rules[rule]

    return rule_fn(budget, current_depth, variables, constants, multiplier)


def generate_compound(budget, current_depth, variables, constants, multiplier):
    result = []

    while random.random() < 0.7:
        result += generate_statement(
            budget, current_depth, variables, constants, multiplier
        )

    return result


os.makedirs(args.output_dir, exist_ok=True)
i = len(os.listdir(args.output_dir))
output_dir = osp.join(args.output_dir, f"{i}-{version}-{args.budget}-{args.max_depth}")
os.makedirs(output_dir)

for j in trange(args.num_generate):
    variables = []

    input_a = new_variable(variables, param=True)
    input_b = new_variable(variables, param=True)

    budget = [random.randint(100, args.budget)]

    constants = [(x, 0) for x in [2, 3, 5]]

    fn_body = []
    while budget[0] > 0:
        fn_body += generate_compound(budget, 1, variables, constants, 1)

    num_outputs = random.randint(1, len(variables) // 10)
    outputs = random.choices(variables, k=num_outputs)

    header = []
    header.append(f"struct Output {{int result[{num_outputs}];}};")
    header.append(
        f"struct Output main(__attribute__((private(0))) int {var_decl(input_a)}, __attribute__((private(0))) int {var_decl(input_b)}) {{"
    )

    footer = []
    footer.append("struct Output output;")
    footer += [
        f"output.result[{i}] = {var_use(v, variables)};" for i, v in enumerate(outputs)
    ]
    footer.append(f"return output;\n}}")

    fn = header + fn_body + footer

    with open(osp.join(output_dir, f"grammar_{version}_{i}_{j}.c"), "w") as f:
        f.write("\n".join(fn))
