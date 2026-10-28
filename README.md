# AIMO Interpretability Challenge 2026

Code, submission bundle and technical report for my entry to the AIMO Interpretability Challenge at NeurIPS 2026.

The task is to say, for a model and a math problem, whether the model's accuracy survives perturbations of the problem. The official training labels show that the answer depends on the problem far more than on the model, so the submission judges each problem once: gpt-oss-120b solves it sixteen times at low reasoning effort with the organizers' solve prompt, and the problem is called robust when at least 80% of the samples give the same answer. The method, how well it works, and what failed before it are in [`report/technical_report.md`](report/technical_report.md).

## Layout

| Path | What it holds |
|---|---|
| `submission/` | The submitted Codabench bundle, a single `solution.py`. For the Small track, add an empty `small.txt` at the root of the zip. |
| `submission_stability/` | The first submission (hidden-state stability under rewrites), kept because the report evaluates it. |
| `code/` | Every experiment in the report. The scripts import one another, so keep them in one directory. |
| `jobs/` | The Slurm scripts as they were run. They record the resources and the exact command of each run. |
| `data/answers_math.json` | Ground-truth answers for the 164 officially labelled MATH problems, read from the hendrycks MATH solutions. |
| `report/` | The technical report. |

## Reproducing the submission

The bundle has no trained artifact, so there is nothing to fit. To reproduce the numbers behind its threshold:

1. Download the five competition checkpoints and `aimo-interp/train-main-v2` from Hugging Face.
2. Sample every train problem with the judge, three runs:
   `python code/nosol_experiment.py --parquet <train-main-v2.parquet> --out-dir <out1>`
   (then `--seed 1 --variants 6` into `<out2>` and `--seed 2 --variants 6 --rewrite subtle` into `<out3>`)
3. Score the signals and the rule by resampling:
   `python code/judge_rules.py --parquet <train-main-v2.parquet> --runs <out1> <out2> <out3>`
4. Run the bundle the way the harness calls it:
   `python code/trainv2_bundle_test.py submission <train-main-v2.parquet> <predictions.jsonl>`

Steps 2 and 4 need a GPU that holds gpt-oss-120b (about 81 GiB at the submitted batch size). Set `AIMO_BASE` to the working directory that holds `data_trainv2/`, `results/` and the bundles (`export AIMO_BASE=/path/to/aimo`); the scripts in `code/` and `jobs/` read it. The job scripts carry the partition and GPU names of the cluster they ran on, so adjust the `#SBATCH` lines for yours, and create `logs/` in the directory you submit from.

## Environment

torch 2.12.1 with CUDA 13.2, transformers 5.13.0 and kernels 0.15.2, matching the competition image. The submitted bundle imports nothing beyond torch and transformers.
