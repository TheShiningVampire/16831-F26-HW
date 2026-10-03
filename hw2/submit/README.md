# HW2: Policy Gradients

Andrew ID: vawale

This folder has my code and experiment logs for HW2 (policy gradients with
reward-to-go, a neural-network baseline, and GAE). The report is submitted
separately as `hw2_submission.pdf`.

## What is in here

- `rob831/` is the completed code, with the same layout as the starter code.
  The parts carried over from HW1 and the new policy-gradient parts
  (`agents/pg_agent.py`, `policies/MLP_policy.py`) are filled in.
- `run_logs/` has one TensorBoard folder per run, copied from `rob831/data`
  with the names unchanged. Folder names start with the `exp_name` from the
  assignment (`q1_...`, `q2_...`, `q3_...`, `q4_...`, `q5_...`), followed by
  the environment and the time stamp that the logger adds. Video logging was
  off for every run.
- `plot_hw2.py` makes every figure in the report from `run_logs/`.

## How to run

All commands are run from the `hw2/` folder of the course repo with the
`rob831` conda environment active, exactly as given in the assignment. The
only additions are `--seed 5` for the InvertedPendulum and HalfCheetah runs
(stated in the report) and `--no_gpu` where the GPU was not available; the
networks are small, so this does not change the results.

The exact command for every run is listed in the report next to its figure.
The run folders in `run_logs/` are:

| Question | Runs | `exp_name` |
|---|---|---|
| Q1 CartPole | 6 | `q1_{sb,lb}_{no_rtg_dsa,rtg_dsa,rtg_na}` |
| Q2 InvertedPendulum | 28 | `q2_b<b>_r<r>`, b in {100, 200, 300, 500, 1000, 2000, 5000}, r in {0.005, 0.01, 0.02, 0.05}; the reported run is `q2_b300_r0.01` |
| Q3 LunarLander | 1 | `q3_b10000_r0.005` |
| Q4 HalfCheetah search | 9 | `q4_search_b<b>_lr<r>_rtg_nnbaseline`, b in {10000, 30000, 50000}, r in {0.005, 0.01, 0.02} |
| Q4 HalfCheetah, best b*, r* | 4 | `q4_b50000_r0.02`, `..._rtg`, `..._nnbaseline`, `..._rtg_nnbaseline` |
| Q4 HalfCheetah, extra comparison at b = 10000 | 4 | `q4_b10000_r0.02`, `..._rtg`, `..._nnbaseline`, `..._rtg_nnbaseline` |
| Q5 Hopper | 4 | `q5_b2000_r0.001_lambda<λ>`, λ in {0, 0.95, 0.99, 1} |

## How to make the figures

From this folder, with the `rob831` environment active (it needs
`matplotlib` and `tensorboard`):

```
python plot_hw2.py
```

writes these files to `figures/`:

| Figure | Report question | Runs used |
|---|---|---|
| `q1_small_batch.pdf` | Q5.1.2.1 | the three `q1_sb_*` runs |
| `q1_large_batch.pdf` | Q5.1.2.2 | the three `q1_lb_*` runs |
| `q2_b300_r0_01.pdf` | Q5.2.3 | `q2_b300_r0.01` |
| `q3_lunarlander.pdf` | Q7.1.2 | `q3_b10000_r0.005` |
| `q4_search.pdf` | Q7.2.2 | the nine `q4_search_*` runs |
| `q4_final_b10000_r0_02.pdf` | Q7.2.4 | the four `q4_b10000_r0.02*` runs |
| `q4_final_b50000_r0_02.pdf` | Q7.2.7 | the four `q4_b50000_r0.02*` runs |
| `q5_hopper.pdf` | Q8.1.2 | the four `q5_*` runs |

`python plot_hw2.py q2 q5` makes only the named figures (names: `q1`, `q2`,
`q3`, `q4_search`, `q4_final`, `q5`). Each plot shows the raw
`Eval_AverageReturn` faintly and a 10-iteration moving average in bold, with
the target score from the assignment as a dashed line. The numbers quoted in
the report (first iteration reaching the target, averages over the last 10 or
50 iterations) are read from the same `Eval_AverageReturn` series.
