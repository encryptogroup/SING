import os
import os.path as osp
from pathlib import Path

import requests
from tqdm import tqdm
import mistletoe

model = os.environ["MODEL"]
prompt_file = os.environ["PROMPT_FILE"]
output_dir = os.environ["OUTPUT_DIR"]

with open(prompt_file, "r") as f:
    prompt = f.read()

os.makedirs(output_dir, exist_ok=True)
i = len(os.listdir(output_dir))
prompt_name = Path(osp.basename(prompt_file)).stem
output_dir = osp.join(output_dir, f"{i}-{model}-{prompt_name}")
os.makedirs(output_dir)

invocations = 30


for i in tqdm(range(invocations)):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
    )

    with open(osp.join(output_dir, f"{i}.json"), "w") as f:
        f.write(r.text)

    json = r.json()
    parsed = mistletoe.Document(json["response"])
    for j, child in enumerate(parsed.children):
        if isinstance(child, mistletoe.block_token.CodeFence):
            if child.language == "c":
                with open(
                    osp.join(output_dir, f"llm_{model}_{prompt_name}_{i}_{j}.c"), "w"
                ) as f:
                    f.write(child.content)
