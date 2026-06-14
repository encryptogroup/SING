import os
import os.path as osp

import glob
import hashlib
import json
import sqlite3

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from tqdm import tqdm

from .circuit import share_types
from .parse_silph import parse_silph
from .eval_cost import eval_cost
from .dataset import load_metadata, check_assignment_filter


num_classes = len(share_types)
splits = ["train", "val", "test"]


class CostPredictionSilphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        cost_model,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
        only_valid=False,
        assignment_filter=None,
    ):
        self.cost_model = cost_model
        self.only_valid = only_valid
        self.assignment_filter = assignment_filter
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        h = hashlib.sha3_256()

        for shahash in self.raw_file_names:
            h.update(shahash.encode("utf-8"))

            alternative_assignments = osp.join(self.raw_dir, shahash, "assignments")
            if not osp.isdir(alternative_assignments):
                continue

            for name in os.listdir(alternative_assignments):
                h.update(name.encode("utf-8"))

        self.dataset_fingerprint = h.hexdigest()

        if split == "train":
            self.load(self.processed_paths[0])
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        dataset_meta_path = osp.join(osp.dirname(__file__), "dataset.json")

        if not osp.exists(dataset_meta_path):
            raise RuntimeError(
                f"Did not find {dataset_meta_path}. Run `generate_dataset_split.py` in the repository root to generate this file from the dataset."
            )

        with open(dataset_meta_path, "r") as f:
            self.dataset_meta = json.load(f)

        hashes = sum([self.dataset_meta[split] for split in splits], [])
        return hashes

    @property
    def processed_file_names(self):
        return [f"{split}.pt" for split in splits]

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please move all files to {self.raw_dir}"
        )

    def process(self):
        data_list = []

        db_con = sqlite3.connect(osp.join(self.raw_dir, "metadata.sqlite"))
        metadata_keys = db_con.execute("SELECT DISTINCT key FROM metadata").fetchall()
        metadata_keys = [key[0] for key in metadata_keys]

        for shahash in tqdm(sorted(self.raw_file_names)):
            bytecode_file = glob.glob(
                osp.join(self.raw_dir, shahash, "*_c_main_bytecode.txt")
            )
            constant_input_file = glob.glob(
                osp.join(self.raw_dir, shahash, "*_c_const.txt")
            )
            share_assignment_file = glob.glob(
                osp.join(self.raw_dir, shahash, "*_c_share_map.txt")
            )

            assert len(bytecode_file) == 1
            assert len(constant_input_file) == 1
            assert len(share_assignment_file) == 1

            circuit, _, _ = parse_silph(
                bytecode_file[0],
                constant_input_file[0],
                share_assignment_file[0],
            )

            x = torch.tensor(circuit.nodes, dtype=torch.float)

            edge_index = torch.tensor(
                [circuit.edge_sources, circuit.edge_targets], dtype=torch.long
            )

            assignments = osp.join(self.raw_dir, shahash, "assignments")
            if not osp.isdir(assignments):
                continue

            for name in os.listdir(assignments):
                # only capture hashes
                if len(name) != 64:
                    continue

                metadata = load_metadata(shahash, name, db_con)
                if not check_assignment_filter(metadata, self.assignment_filter):
                    continue

                y = torch.load(osp.join(assignments, name))
                y_one_hot = F.one_hot(y, num_classes).float()

                invalid, cost = eval_cost(x, edge_index, y, self.cost_model)

                if self.only_valid:
                    if invalid > 0:
                        continue

                if invalid > 0:
                    y = torch.tensor([[1, 0]])
                else:
                    y = torch.tensor([[0, cost]])

                metadata_or_none = {key: metadata.get(key, "") for key in metadata_keys}

                data = Data(
                    x=torch.cat((x, y_one_hot), dim=1),
                    edge_index=edge_index,
                    y=y,
                    shahash=shahash,
                    assignment_hash=name,
                    **metadata_or_none,
                )
                data_list.append(data)


        for split in splits:
            split_datas = [x for x in data_list if x.shahash in self.dataset_meta[split]]
            self.save(split_datas, osp.join(self.processed_dir, f"{split}.pt"))

        db_con.close()
