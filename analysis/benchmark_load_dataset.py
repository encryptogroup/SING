import time

import torch

from sing import (
    SilphDataset,
)

times = torch.zeros(3, dtype=torch.float)

for i, _ in enumerate(times):
    start = time.perf_counter()
    train_dataset = SilphDataset("dataset", split="train", force_reload=True)
    end = time.perf_counter()
    times[i] = end - start

print(times)
