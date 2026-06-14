import os
import os.path as osp

import glob
import hashlib
import json

import torch
from torch_geometric.data import Data, InMemoryDataset

from tqdm import tqdm

from .parse_silph import parse_silph


splits = ["train", "val", "test"]


class SilphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        h = hashlib.sha3_256()
        for shahash in self.raw_file_names:
            h.update(shahash.encode("utf-8"))
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

            circuit, share_assignment, _ = parse_silph(
                bytecode_file[0],
                constant_input_file[0],
                share_assignment_file[0],
            )

            data = Data(
                x=torch.tensor(circuit.nodes, dtype=torch.float),
                edge_index=torch.tensor(
                    [circuit.edge_sources, circuit.edge_targets], dtype=torch.long
                ),
                y=torch.tensor(share_assignment, dtype=torch.long),
                shahash=shahash,
            )
            data_list.append(data)


        for split in splits:
            split_datas = [x for x in data_list if x.shahash in self.dataset_meta[split]]
            self.save(split_datas, osp.join(self.processed_dir, f"{split}.pt"))
