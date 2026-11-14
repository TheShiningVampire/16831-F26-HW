# HW4: Model-Based Reinforcement Learning

Andrew ID: vawale

The report is in the separate file `hw4_submission.pdf`.

## Contents

- `rob831/hw4_part1/`: The completed code. The directory structure is the same as the starter code. The TODO sections are complete in these files:
  - `agents/mb_agent.py`, `models/ff_model.py`, `policies/MPC_policy.py`.
  - `infrastructure/utils.py`, `infrastructure/rl_trainer.py` (code from the previous homework).

  The other files have no changes.
- `run_logs/`: One TensorBoard directory for each run. The directories are copies from `rob831/data/`, with the original names. Each name starts with `hw4_` and the `exp_name` from the assignment. The environment name and a time stamp follow the `exp_name`. The value of `--video_log_freq` is -1 for all runs.
- `plot_hw4.py`: The script that makes all figures in the report from `run_logs/`.

## Commands for the runs

Do all commands from the `hw4/` directory of the course repository. Activate the `rob831` conda environment first. The commands are the same as the commands in the assignment.

Python must import `rob831.hw4_part1` from `hw4/`. If an editable install of `rob831` points to a different homework, put `PYTHONPATH=$PWD` before each command.

Q1, dynamics model on HalfCheetah:

```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n5_arch2x250 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch2x250 --env_name cheetah-hw4_part1-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

Q2, MPC action selection on Obstacles:

```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q2_obstacles_singleiteration --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

Q3, iterative model training:

```
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q3_obstacles --env_name obstacles-hw4_part1-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q3_reacher --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q3_cheetah --env_name cheetah-hw4_part1-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20 --video_log_freq -1 --mpc_action_sampling_strategy 'random'
```

Q4, hyperparameter comparison on Reacher: the eight `q4_reacher_*` commands in the assignment, with no changes.

## Commands for the figures

Go to the directory that contains `plot_hw4.py`. Activate the `rob831` conda environment. The environment must have `matplotlib` and `tensorboard`. Then do this command:

```
python plot_hw4.py
```

The script writes these files to `figures/`. Each figure is in `.pdf` format and in `.png` format, except the Q1 figures.

| Figure | Report question | Runs and data |
|---|---|---|
| `q1_cheetah_*.png` | Q1 | A copy of `itr_0_predictions.png` from each `q1_*` run. |
| `q2_obstacles` | Q2 | `q2_obstacles_singleiteration`. `Train_AverageReturn` and `Eval_AverageReturn` after 1 iteration. |
| `q3_obstacles`, `q3_reacher`, `q3_cheetah` | Q3 | The `q3_*` runs. `Train_AverageReturn` and `Eval_AverageReturn` at each iteration. |
| `q4_reacher_horizon`, `q4_reacher_numseq`, `q4_reacher_ensemble` | Q4 | The `q4_reacher_*` runs. `Eval_AverageReturn` at each iteration for each value. |
