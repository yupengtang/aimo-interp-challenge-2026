# AIMO Interpretability Challenge 2026

Code, submission bundle and technical report for my entry to the AIMO Interpretability Challenge at NeurIPS 2026.

The task is to say, for a model and a math problem, whether the model's accuracy survives perturbations of the problem. My nominated entries read the answer from the evaluated model itself, without generating text. For each problem the model sees the original statement and six rewrites that cannot change the answer, one forward pass each, and a logistic regression over how far its hidden states and next-token distribution move gives the verdict. The Main-track entry uses all 31 features; the Small-track entry uses only the 24 hidden-state features. The ablation in the report shows that the hidden states carry the Small result but not the Main one. The report, with the comparison methods and what failed, is in [`report/technical_report.md`](report/technical_report.md).

## Layout

| Path | What it holds |
|---|---|
| `submission_stability/` | Main-track entry (Codabench 991098): representation stability, 31 features. |
| `submission_stability_hidden/` | Small-track entry (Codabench 1022932): the same code with the 24-feature hidden-state artifact. Zip it with an empty `small.txt` at the root. |
| `submission/` | The black-box comparison: a gpt-oss-120b judge in a single `solution.py`. |
| `submission_whitebox_v2/` | The judge read from gpt-oss hidden states (report Section 7). |
| `code/` | Every experiment in the report. The scripts import one another, so keep them in one directory. |
| `jobs/` | The Slurm scripts as they were run. They record the resources and the exact command of each run. |
| `data/` | Answers for the 164 officially labelled MATH problems and the problem lists used in the report. |
| `report/` | The technical report. |

## Reproducing the entries

1. Download the five competition checkpoints and the public problems.
2. Features: `python code/perturb_stability.py --data-dir <public data> --pool <unlabelled problems jsonl> --out-dir <features>` (job: `jobs/aimo_stability.sbatch`).
3. Fit and export the Main artifact: `python code/train_stability_model.py --base <workdir> --out submission_stability/stability_crossmodel.joblib`. Add `--drop-prefix kl_ --drop-prefix prompt_tokens` for the Small artifact, or `--drop-prefix cos_` for the output-level variant of the ablation.
4. Check against train-main-v2: `python code/trainv2_bundle_eval.py --parquet <train-main-v2.parquet> --bundle submission_stability --out <dir>`.

## Reproducing the judge comparison

1. Sample every train problem with the judge, three runs:
   `python code/nosol_experiment.py --parquet <train-main-v2.parquet> --out-dir <out1>`
   (then `--seed 1 --variants 6` into `<out2>` and `--seed 2 --variants 6 --rewrite subtle` into `<out3>`)
2. Score the signals and the rule by resampling:
   `python code/judge_rules.py --parquet <train-main-v2.parquet> --runs <out1> <out2> <out3>`
3. Run the bundle the way the harness calls it:
   `python code/trainv2_bundle_test.py submission <train-main-v2.parquet> <predictions.jsonl>`

The judge needs a GPU that holds gpt-oss-120b (about 81 GiB at the submitted batch size). Set `AIMO_BASE` to the working directory that holds `data_trainv2/`, `results/` and the bundles (`export AIMO_BASE=/path/to/aimo`); the scripts in `code/` and `jobs/` read it. The job scripts carry the partition and GPU names of the cluster they ran on, so adjust the `#SBATCH` lines for yours, and create `logs/` in the directory you submit from.

## Environment

torch 2.12.1 with CUDA 13.2, transformers 5.13.0 and kernels 0.15.2, matching the competition image. The stability bundles need only NumPy and joblib beyond torch and transformers.
