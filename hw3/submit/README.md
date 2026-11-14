# HW3: Q-Learning and Actor-Critic

Andrew ID: vawale

The report is in the separate file `hw3_submission.pdf`.

## Contents

- `rob831/`: The completed code. The directory structure is the same as the starter code. The TODO sections are complete in these files:
  - Part 1: `agents/dqn_agent.py`, `critics/dqn_critic.py`, `policies/argmax_policy.py`.
  - Part 2: `agents/ac_agent.py`, `critics/bootstrapped_continuous_critic.py`, `policies/MLP_policy.py`.

  The other files have no changes.
- `run_logs/`: One TensorBoard directory for each run. The directories are copies from `data/`, with the original names. Each name starts with the `exp_name` from the assignment. The environment name and a time stamp follow the `exp_name`. The value of `--video_log_freq` is -1 for all runs.
- `plot_hw3.py`: The script that makes all figures in the report from `run_logs/`.

## Commands for the runs

Do all commands from the `hw3/` directory of the course repository. Activate the `rob831` conda environment first. The commands are the same as the commands in the assignment. All runs use the default hyperparameters.

Python must import `rob831` from `hw3/`. If an editable install of `rob831` points to a different homework, do one of these steps first:

- Do `pip install -e .` in `hw3/`.
- Put `PYTHONPATH=$PWD` before each command.

Q1, DQN and double DQN on LunarLander-v3 (seeds 1, 2 and 3):

```
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1_dqn_1 --seed 1 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1_dqn_2 --seed 2 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1_dqn_3 --seed 3 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1_doubledqn_1 --double_q --seed 1 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1_doubledqn_2 --double_q --seed 2 --no_gpu
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q1_doubledqn_3 --double_q --seed 3 --no_gpu
```

Q2, actor-critic on CartPole-v0:

```
python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q2_10_10 -ntu 10 -ngsptu 10 --no_gpu
```

Q3, actor-critic on InvertedPendulum-v4:

```
python rob831/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q3_10_10 -ntu 10 -ngsptu 10 --no_gpu
```

## Commands for the figures

Go to the directory that contains `plot_hw3.py`. Activate the `rob831` conda environment. The environment must have `matplotlib` and `tensorboard`. Then do this command:

```
python plot_hw3.py
```

The script writes these files to `figures/`. Each figure is in `.pdf` format and in `.png` format.

| Figure | Report question | Runs and data |
|---|---|---|
| `q1_dqn_vs_ddqn` | Q1 | The six `q1_*` runs. One plot shows the mean and the standard deviation of the 3 seeds. Solid lines: `Train_AverageReturn`. Dashed lines: `Train_BestReturn`. |
| `q2_cartpole` | Q2 | `q2_10_10`. The raw `Eval_AverageReturn` value at each iteration. |
| `q3_pendulum` | Q3 | `q3_10_10`. The raw `Eval_AverageReturn` value at each iteration. |

For Q2 and Q3, the script also prints three values of `Eval_AverageReturn`: the final value, the maximum value, and the mean of the last 10 iterations.
