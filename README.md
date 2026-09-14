# EgoPathBench

**Evaluating Zero-Shot Egocentric Waypoint Decision-Making in Vision-Language Models**

Yang Zhao, Xubo Yang (corresponding author) · Shanghai Jiao Tong University

[Project page](https://egopathbench.github.io/EgoPathBench/) ·
[Dataset](https://huggingface.co/datasets/runder1/EgoPathBench)

EgoPathBench asks vision-language models to select traversable waypoints or form
goal-directed routes from a first-person RGB image, a natural-language goal, and
numbered visible candidates. Five tasks evaluate point/embodied traversability,
point/embodied paths, and intent-conditioned embodied paths.

| Split | Questions |
| --- | ---: |
| Train | 31,852 |
| Validation | 1,345 |
| Benchmark | 1,111 |

The formal release manifest is authoritative. Benchmark evaluation uses the
**evaluation-corrected** VQA and GT together with direct-pair sidecars and
acceptable goal IDs. Replacing these with a sparse graph changes the protocol.

## Download and evaluate

The data release is being uploaded; check its dataset card for completion status.

```bash
git clone https://github.com/EgoPathBench/EgoPathBench.git
cd EgoPathBench
python -m pip install huggingface_hub
hf download runder1/EgoPathBench --repo-type dataset --local-dir downloads
python scripts/unpack_dataset.py --source downloads --output dataset
```

Archive verification/extraction requires Python 3.12 or newer. Run commands **from the dataset directory**, because runtime paths are relative
to that directory. The evaluator itself uses the Python standard library.

```bash
cd dataset
python ../scripts/evaluate_next.py \
  --vqa benchmark_evalfix/benchmark/vqa/vqa_next_a2.jsonl \
  --gt benchmark_evalfix/benchmark/gt/gt_next_a2.jsonl \
  --predictions /path/to/predictions_a2.jsonl \
  --output metrics_a2.json
```

Each prediction is a JSONL record with `question_id` and `output`, where `output`
is a list of display IDs (or its JSON string). Use the matching files for each of
`a1`, `b1`, `a2`, `b2`, and `c`. The benchmark has 146, 146, 309, 309, and 201
questions in those tasks, respectively. Evaluate all questions; do not use
`--restrict-to-predictions` for the main benchmark.

EgoPath Score is the arithmetic mean of chance-adjusted A1/B1 balanced accuracy
(`2 * BA - 1`, with BA on a 0–1 scale) and A2/B2/C success rate. The project page reports these values on a 0–100 scale.

For API inference, install the matching provider client and use
`scripts/run_benchmark_next.py --help`. Provider settings are read from environment
variables. Never include ground-truth answers or evaluator sidecars in model input.
Full rendering/training dependencies are listed in `requirements.txt`; they are not
needed to score saved predictions.

## Repository and data layout

```text
scripts/        Inference, evaluation, construction, training/export utilities
tests/          Behavioral tests for the corresponding tools
configs/        Construction/training examples (adapt local paths)
paper/          Selected paper artifacts and reported result tables
website/        Static GitHub Pages project page
dataset/        Downloaded separately from Hugging Face
  release/              Frozen train, val, benchmark and selection manifests
  benchmark_evalfix/    Final benchmark evaluation contract
  sidecars/             Construction and reference metadata
  assets/               Referenced observations, candidates and graphs
```

The public repository is a clean export of the research workspace. Local reviews,
credentials, agent state, experiment backups, and development Git history are not
part of this release. Some construction/training utilities retain environment-specific
paths and need adaptation; the portable benchmark evaluation path is described above.

## Licenses and contact

Original code: [MIT](LICENSE). Data: **CC BY-NC-SA 4.0**, retaining applicable
upstream terms; original scene meshes are not redistributed. See
[third-party notices](THIRD_PARTY_NOTICES.md).

Contact: [Yang Zhao](mailto:runder1103@sjtu.edu.cn),
[Xubo Yang](mailto:yangxubo@sjtu.edu.cn).

arXiv submission is pending. No arXiv identifier has been assigned.
