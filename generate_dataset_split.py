import os
import os.path as osp

import argparse
import json
import glob

from tqdm import tqdm

import torch
import torch.nn.functional as F

import pandas as pd

from sing import parse_silph, share_types, read_hashes

num_classes = len(share_types)

splits = ["train", "val", "test"]

split_ratios = torch.tensor([0.6, 0.2, 0.2])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-dir",
    type=str,
    default="dataset/raw",
    help="Path to dataset directory",
)
parser.add_argument(
    "--output-file",
    type=str,
    default="sing/dataset.json",
    help="Path to output directory",
)
parser.add_argument(
    "--max-size",
    type=int,
    default=10 * 1000 * 1000,
    help="Maximum size in bytes",
)
parser.add_argument(
    "--include-file",
    type=str,
    help="File with hashes to include",
)
parser.add_argument(
    "--exclude-file",
    type=str,
    default=[],
    action="append",
    help="Files with hashes to exclude",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.3,
    help="Weight of score_size",
)
args = parser.parse_args()

if len(args.exclude_file) == 0:
    args.exclude_file += ["exclude.txt"]
    file_list = ", ".join(args.exclude_file)
    print(f"excluding files {file_list}")

if args.include_file:
    hashes = read_hashes(args.include_file)
else:
    hashes = os.listdir(args.dataset_dir)

exclude_hashes = []
for file_path in args.exclude_file:
    exclude_hashes += read_hashes(file_path)

data = []

for shahash in tqdm(hashes):
    if not osp.isdir(osp.join(args.dataset_dir, shahash)):
        continue

    if shahash in exclude_hashes:
        continue

    bytecode_file = glob.glob(
        osp.join(args.dataset_dir, shahash, "*_c_main_bytecode.txt")
    )
    constant_input_file = glob.glob(
        osp.join(args.dataset_dir, shahash, "*_c_const.txt")
    )
    share_assignment_file = glob.glob(
        osp.join(args.dataset_dir, shahash, "*_c_share_map.txt")
    )

    assert len(bytecode_file) == 1
    assert len(constant_input_file) == 1
    assert len(share_assignment_file) == 1

    bytecode_size = osp.getsize(bytecode_file[0])
    constant_input_size = osp.getsize(constant_input_file[0])

    if bytecode_size + constant_input_size > args.max_size:
        continue

    circuit, share_assignment, _ = parse_silph(
        bytecode_file[0],
        constant_input_file[0],
        share_assignment_file[0],
    )

    y = torch.tensor(share_assignment, dtype=torch.long)
    y_one_hot = F.one_hot(y, num_classes)
    share_types_sum = torch.sum(y_one_hot, dim=0)

    data.append((shahash, share_types_sum))

# sort descending in circuit size (num nodes)
data.sort(key=lambda x: torch.sum(x[1]), reverse=True)

distribution = torch.zeros(num_classes)
for x in data:
    _, share_types_sum = x
    distribution += share_types_sum
distribution /= torch.sum(distribution)

split_shahashes = [[] for _ in splits]
split_share_types_sum = torch.zeros((len(splits), len(share_types)))

for x in data:
    shahash, share_types_sum = x

    possible_new_split_share_types_sum = split_share_types_sum + share_types_sum

    # for each split, how large (num nodes)
    possible_new_split_sum = torch.sum(possible_new_split_share_types_sum, dim=1)
    possible_new_split_sum_norm = possible_new_split_sum / torch.sum(
        possible_new_split_sum
    )

    # for each split, how are share types distributed
    possible_new_split_share_types_dist = (
        possible_new_split_share_types_sum / possible_new_split_sum.unsqueeze(1)
    )

    # for each split, score of how much size differs to wanted size
    score_size = split_ratios - possible_new_split_sum_norm

    # for each split, score of how much share type distribution differs from wanted
    score_share_types_dist = distribution - possible_new_split_share_types_dist
    score_share_types_dist = torch.min(score_share_types_dist, dim=1)[0]

    alpha = args.alpha
    score = alpha * score_size + (1 - alpha) * score_share_types_dist

    split = torch.argmax(score)
    split_shahashes[split].append(shahash)
    split_share_types_sum[split] += share_types_sum

split_share_types_ratio = split_share_types_sum / torch.sum(split_share_types_sum, dim=1).unsqueeze(1)

print("Statistics")
print(f"{len(data)} circuits, {torch.sum(split_share_types_sum).long()} nodes")
print()
print("overall distribution of share types")
print(pd.DataFrame(distribution.unsqueeze(0), columns=share_types))
print()
print("split sizes")
print(pd.DataFrame(torch.cat((
    torch.tensor([len(split) for split in split_shahashes]).unsqueeze(0),
    torch.sum(split_share_types_sum, dim=1).unsqueeze(0),
)).long(), columns=splits, index=["circuits", "nodes"]))
print()
print("distribution of share types in splits (nodes)")
print(pd.DataFrame(split_share_types_sum.long(), columns=share_types, index=splits))
print()
print("distribution of share types in splits (ratio of split)")
print(pd.DataFrame(split_share_types_ratio, columns=share_types, index=splits))

output = {
    split_name: shahashes for split_name, shahashes in zip(splits, split_shahashes)
}
output["args"] = vars(args)

with open(args.output_file, "w") as f:
    json.dump(output, f)
