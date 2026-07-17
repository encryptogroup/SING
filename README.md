# SING: Improving the efficiency of Secure Multi-Party Computation Protocol Assignment using Neural Networks

This repository contains the full code and a small excerpt of our
dataset for reproducing our training and evaluation results.

## Initial Setup

```bash
python -m venv venv
source venv/bin/activate

# may differ depending on your platform
#
# see https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

pip install -r requirements.txt


# unpack dataset
tar xf dataset-excerpt.tar.gz
tar xf dataset-excerpt-c.tar.gz
```

<details>

<summary>
Optional: Compiling ABY
</summary>

This step is not necessary for training or evaluation of our
models. ABY is required for benchmarking MPC performance.

```bash
cd ABY-vendor
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DABY_BUILD_EXE=On
make
```

</details>

<details>

<summary>
Optional: Compiling Silph
</summary>

This step is not necessary for training, evaluation, or
benchmarking. Compiling Silph is a precondition for comparing SING and
Silph performance. We have applied minor fixes to the build system of
the [original Silph
repository](https://github.com/circify/circ/tree/mpc_aws). The build
system as a whole is identical to the original release. We thus refer
to the documentation and paper for more detailed instructions.

Silph is written in Rust and thus requires a stable Rust toolchain
which is commonly installed with [rustup](https://rustup.rs/).

Furthermore, an installation of `coinor-cbc` is required for ILP
solving. This library can be found in some distribution repositories
or [built from source](https://coin-or.github.io/coinbrew/). Other
libraries that Silph depends on (e.g., KaHyPar) will be built from
source automatically.

```bash
cd silph
python driver.py --features aby bench c lp r1cs smt
python driver.py --install
python driver.py --build --mode release
```

Note that this build process may take multiple hours as tests will be
run as part of the build process.

</details>

## Dataset

Due to the large size of the dataset, we only provide a small excerpt
in this repository.

- `dataset-excerpt-c` contains the original C source files for the
  circuits.
- `dataset-excerpt` contains a processed excerpt of the dataset
  consisting of compiled circuits, alternative share assignments, and
  benchmarked metrics.

Training scripts will process the dataset further into the PyTorch
Geometric format. This happens automatically. We refer to the
following sections on training for more details.

Note that the training and evaluation performance of this dataset
excerpt will vary from the results in our paper.

<details>

<summary>
Optional: Bootstrapping the dataset from C source files
</summary>

The dataset processing starts with a directory containing C source
files (`dataset-excerpt-c`).

### Step 1: Compile all circuits

Compilation requires a compiled version of Silph in
`$CARGO_MANIFEST_DIR` (cf. Initial Setup).

```bash
scripts/generate_dataset.sh
```

### Step 2: Clean up the dataset

```bash
pushd dataset-compiled
scripts/find_duplicate_circuits.sh failed.log
popd
```

### Step 3: Generating alternative share assignments

This is a necessary step for training our cost prediction model.

```bash
python generate_alternative_share_assignments.py
```

### Step 4: Benchmark MPC runtimes

This step is necessary for training our cost prediction model on
real-world benchmarks (SING 3).

Benchmarking requires ABY (cf. Initial Setup).

```bash
python benchmark_mpc.py --mode dataset --network-setting LAN --cost-name runtime-neon-lan --metric runtime
```

</details>

### Calculate dataset splits

Training and evaluation requires splitting the dataset into training,
validation, and test sets. This is done as follows:

```bash
python generate_dataset_split.py
```

`generate_dataset_split.py` includes many configuration options that
influence the resulting split (e.g., setting a maximum size threshold
for circuits included in the dataset). Run

```bash
python generate_dataset_split.py --help
```

for a detailed overview of all command-line options.

<details>

<summary>
Optional: Generating circuits with LLMs
</summary>

We use [Ollama](https://ollama.com/) as a local LLM inference
engine. Instructions on installing Ollama can be found on the
[official website](https://ollama.com/download).

Once Ollama is set up, various open-source LLMs can be downloaded,
e.g.,

```bash
ollama pull gemma3:4b
```

Using a script, all locally downloaded models can be queried with all
available prompts respectively.

```bash
cd llm-generate
bash generate_missing_combinations.sh
```

</details>

<details>

<summary>
Optional: Generating random circuits
</summary>

```bash
cd grammar-generate
python generate.py
```

Several options of the generation process (e.g., operation budget) can
be configured via command-line options. Run

```bash
python generate.py --help
```

for a full list of options.

</details>

## Cost Prediction Model

Our cost prediction model $C^\text{SING}$ predicts a numeric cost
given a circuit $c$ and a share assignment $s$.

We provide model checkpoints used in our evaluation in the
`pretrained` directory.

### Train

Depending on whether the model predicts Silph costs or benchmarked
runtimes, the training process uses a different directory to store the
dataset in PyTorch Geometric format.

- Silph costs: `dataset-cost-prediction`
- Benchmarked runtimes: `dataset-cost-prediction-measured`

```bash
# train on Silph costs
mkdir -p dataset-cost-prediction
ln -s dataset-excerpt dataset-cost-prediction/raw

python train_cost_prediction.py --lr 0.001 --cost-name silph


# train on benchmarked costs (e.g., runtime-neon-lan)
mkdir -p dataset-cost-prediction-measured
ln -s dataset-excerpt dataset-cost-prediction-measured/raw

python train_cost_prediction.py --lr 0.001 --cost-name runtime-neon-lan
```

Trained models will be saved in the `checkpoints` directory.

### Evaluation

Use `eval_cost_prediction.py` to calculate metrics on how the share
assignment of SING differs from that of Silph (e.g., MSE, R2-score).

```bash
python eval_cost_prediction.py
```

Using the `--checkpoint <PATH>` flag, the evaluation can be performed
on a specific model checkpoint.

This script supports multiple command-line options to load a specific
model checkpoint, filter the dataset, or configure visulization. Run

```bash
python eval_cost_prediction.py --help
```

for a detailed overview of all command-line options.

## Share Assignment Model

Our share assignment model $S^\text{SING}$ outputs a share assignment
$s$ for a circuit $c$.

We provide model checkpoints used in our evaluation in the
`pretrained` directory.

### Train

The dataset in PyTorch Geometric format will be stored in `dataset`.

```bash
mkdir -p dataset

ln -s dataset-excerpt dataset/raw

# supervised (SING 1)
python train.py --lr 0.01 --alpha 0.5

# semi-supervised (SING 2, SING 3)
python train.py --lr 0.01 --alpha 0.1 --predicted-cost
```

Trained models will be saved in the `checkpoints` directory.

### Evaluation

Use `eval.py` to calculate metrics on how the share assignment of SING
differs from that of Silph (e.g., accuracy, confusion matrix).

```bash
python eval.py
```

Use `benchmark_share_assignment.py` to compare the SING and Silph
runtimes of generating share assignments for circuits. This step
requires a compiled version of Silph (cf. Initial Setup).

```bash
python benchmark_share_assignment.py --circuits-file paper_benchmark_c.txt
```

Use `bechmark_mpc.py` to benchmark runtimes and communication amounts
of SING and Silph share assignments. The result will be written to a
CSV file. This step requires a compiled version of ABY (cf. Initial
Setup).

```bash
python benchmark_mpc.py --mode benchmark --hashes paper_benchmark_hashes.txt
```

For setting up network simulations, `benchmark_mpc.py` needs to be run
as a user with `sudo` ability, i.e., the user needs to be in the
`wheel` group.

Using the `--checkpoint <PATH>` flag, the evaluation can be performed
on a specific model checkpoint.

## Ablation

To verify that models consider both the circuit structure and
operation types when predicting costs or assigning shares, we have
included options to remove the respective information during the
training process. Both `train.py` and `train_cost_prediction.py`
support following ablations:

- `--delete-edges` removes all edges during training
- `--zero-features` zeroes all node features, effectively removing
  information about the node operation type for each node

After training with ablation, run `eval.py` or
`eval_cost_prediction.py` respectively to measure the impact of the
ablation on model performance.

## Reproducing Plots and Tables

All plots and tables in our paper can be reproduced from measured data
using the `benchmark-results/plot.py` script.

```bash
cd benchmark-results

# generate plots with plotly
python plot.py --format pdf

# generate LaTeX tables
python plot.py --format latex
```

`plot.py` expects benchmark results to be in specific locations in the
`benchmark-results` subdirectory:

- `share-assignment/smart_glp.csv`: Share assignment benchmarks output
  from `benchmark_share_assignment.py`
- `mpc/silph_LAN.csv`, `mpc/<UUID>_LAN.csv`: MPC runtimes and
  communication amounts measured in the LAN network setting which are
  output by `benchmark_mpc.py`
- `mpc/silph_WAN.csv`, `mpc/<UUID>_WAN.csv`: like above, but MPC
  benchmark results from the WAN network setting

To plot your results, replace the respective files by the outputs of
`benchmark_mpc.py` or `benchmark_share_assignment.py`. Note that MPC
benchmarks for Silph and SING are output in a combined CSV file. The
Silph benchmark numbers need to be manually extracted to the
`mpc/silph_LAN.csv` and `mpc/silph_WAN.csv` files respectively.
