# HW1: how to reproduce the results

Every command behind the numbers and figures in the report.

| Report item | Section |
| --- | --- |
| Table 1 (expert returns) | Part 2 |
| Table 2 (BC on Ant and Hopper) | Part 3 |
| Figure 1 (BC with less expert data) | Part 4 |
| Figure 2 (DAgger learning curves) | Question 2 |

## Before you start

Run everything from the `hw1/` folder with the conda environment active:

```bash
cd hw1
conda activate rob831
```

Run the sections in order, because the plots read the logs from Part 2 and Part 3.

Each command saves its output to `submit/run_logs/<name>.log` and writes a TensorBoard folder to `data/`. Copy those folders into `submit/run_logs/` afterwards. All runs use seed 1, so a rerun gives exactly the same numbers.

| Path | What it is |
| --- | --- |
| `RUN.md` | This file |
| `run_logs/` | Logs and TensorBoard folders for every run |
| `plot_part4.py` | Builds Figure 1 |
| `plot_q2.py` | Builds Figure 2 |
| `rob831/` | The completed code |

## Settings used everywhere

Every BC and DAgger run uses these settings, which keeps the comparison between tasks fair. Part 4 and Question 2 each change one thing, and say which.

| Setting | Flag | Value |
| --- | --- | --- |
| Hidden layers | `--n_layers` | 5 |
| Layer width | `--size` | 64 |
| Learning rate | `--learning_rate` | 4e-3 |
| Gradient steps | `--num_agent_train_steps_per_iter` | 10000 |
| Minibatch size | `--train_batch_size` | 100 |
| Eval transitions | `--eval_batch_size` | 5000 |
| Episode length | `--ep_len` | 1000 |
| Seed | `--seed` | 1 |

- Evaluation collects 5000 steps and an episode is at most 1000 steps, so every mean and standard deviation covers at least 5 rollouts.
- The training data is the provided expert file: 2 trajectories of 1000 steps, so 2000 transitions.

## Part 1: code

Nothing to run. The `TODO` sections are filled in, and the completed code is in the `rob831/` folder next to this file.

## Part 2: expert performance (Table 1)

One run per environment, about 2 seconds each. On the first iteration the training data is the expert data, so the expert's score is printed as `Train_AverageReturn` and `Train_StdReturn`.

Ant:

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 --exp_name expert_ant --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --video_log_freq -1 2>&1 | tee submit/run_logs/q1_expert_ant.log
```

Humanoid:

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 --exp_name expert_humanoid --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --video_log_freq -1 2>&1 | tee submit/run_logs/q1_expert_humanoid.log
```

Walker2d:

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Walker2d.pkl \
  --env_name Walker2d-v2 --exp_name expert_walker --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
  --video_log_freq -1 2>&1 | tee submit/run_logs/q1_expert_walker.log
```

Hopper:

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Hopper.pkl \
  --env_name Hopper-v2 --exp_name expert_hopper --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
  --video_log_freq -1 2>&1 | tee submit/run_logs/q1_expert_hopper.log
```

HalfCheetah:

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/HalfCheetah.pkl \
  --env_name HalfCheetah-v2 --exp_name expert_halfcheetah --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl \
  --video_log_freq -1 2>&1 | tee submit/run_logs/q1_expert_halfcheetah.log
```

To pull out the numbers:

```bash
grep -H -E "Train_AverageReturn|Train_StdReturn" submit/run_logs/q1_expert_*.log
```

Ignore the `Eval_*` lines in these logs. They come from a throwaway policy and are not used in the report.

## Part 3: behavioral cloning (Table 2)

Two runs with identical settings. BC reaches at least 30% of the expert on Ant, and stays below 30% on Hopper.

Ant:

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --n_layers 5 --size 64 --learning_rate 4e-3 \
    --num_agent_train_steps_per_iter 10000 \
    --train_batch_size 100 --eval_batch_size 5000 --ep_len 1000 \
    --seed 1 --video_log_freq -1 --no_gpu \
    2>&1 | tee submit/run_logs/q1_bc_ant.log
```

Hopper:

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Hopper.pkl \
    --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
    --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 \
    --n_layers 5 --size 64 --learning_rate 4e-3 \
    --num_agent_train_steps_per_iter 10000 \
    --train_batch_size 100 --eval_batch_size 5000 --ep_len 1000 \
    --seed 1 --video_log_freq -1 --no_gpu \
    2>&1 | tee submit/run_logs/q1_bc_hopper.log
```

To pull out the numbers:

```bash
grep -E "Eval_AverageReturn|Eval_StdReturn|Eval_AverageEpLen|Train_AverageReturn" \
    submit/run_logs/q1_bc_ant.log submit/run_logs/q1_bc_hopper.log
```

- `Eval_AverageReturn` and `Eval_StdReturn` are the BC policy's score.
- `Train_AverageReturn` is the expert's score. Dividing the two gives the percentage of expert.
- `Eval_AverageEpLen` well below 1000 means the robot falls over before the episode ends.

## Part 4: BC with less expert data (Figure 1)

Hypothesis: BC only works in states the demonstrations cover, so removing expert data should make the return drop sharply rather than gently.

BC is trained on Ant with 125, 250, 500, 1000 and 2000 expert transitions. Everything else stays as in Part 3.

The amount of data is set with `--max_replay_buffer_size`. The buffer keeps only the last N transitions it receives, so the agent trains on the last N transitions of the expert data, not a random sample.

```bash
for N in 125 250 500 1000 2000; do
  P=$(printf %04d $N)
  python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --env_name Ant-v2 --exp_name p4_bc_ant_data$P --n_iter 1 \
    --n_layers 5 --size 64 --learning_rate 4e-3 \
    --num_agent_train_steps_per_iter 10000 \
    --train_batch_size 100 --eval_batch_size 5000 --ep_len 1000 \
    --max_replay_buffer_size $N \
    --seed 1 --video_log_freq -1 --no_gpu \
    2>&1 | tee submit/run_logs/q1_p4_bc_ant_data$P.log
done
```

The `p4_` prefix marks the Part 4 runs, and N is zero padded (`data0125`) so the folders list in order. The run with 2000 transitions uses exactly the Part 3 settings, so it matches the Part 3 Ant result. That is expected.

To pull out the numbers:

```bash
grep -H -E "Eval_AverageReturn|Eval_StdReturn|Eval_AverageEpLen" submit/run_logs/q1_p4_bc_ant_data*.log
```

To build Figure 1:

```bash
python submit/plot_part4.py
```

This writes `q1_part4.pdf` next to the report. The expert line comes from `submit/run_logs/q1_expert_ant.log`, so run Part 2 first.

## Question 2: DAgger (Figure 2)

DAgger starts out exactly like BC. Then, in every later iteration, it:

1. runs the current policy in the environment,
2. asks the expert what it would have done in the states the policy reached,
3. adds those labelled states to the training data, and
4. retrains the policy.

So the policy gets corrected on its own mistakes instead of only ever seeing the expert's states.

Both runs use the Part 3 settings, plus:

| Setting | Flag | Value |
| --- | --- | --- |
| Turn on expert relabeling | `--do_dagger` | on |
| DAgger iterations | `--n_iter` | 10 |
| New transitions per iteration | `--batch_size` | at least 1000 |

Whole rollouts are collected, so an iteration can add more than 1000 transitions when episodes end early. Iteration 0 is plain BC, so it matches the Part 3 result. Each run takes about 3 minutes.

Ant:

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger \
    --n_layers 5 --size 64 --learning_rate 4e-3 \
    --num_agent_train_steps_per_iter 10000 --batch_size 1000 \
    --train_batch_size 100 --eval_batch_size 5000 --ep_len 1000 \
    --seed 1 --video_log_freq -1 --no_gpu \
    2>&1 | tee submit/run_logs/q2_dagger_ant.log
```

Hopper:

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Hopper.pkl \
    --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
    --env_name Hopper-v2 --exp_name dagger_hopper --n_iter 10 --do_dagger \
    --n_layers 5 --size 64 --learning_rate 4e-3 \
    --num_agent_train_steps_per_iter 10000 --batch_size 1000 \
    --train_batch_size 100 --eval_batch_size 5000 --ep_len 1000 \
    --seed 1 --video_log_freq -1 --no_gpu \
    2>&1 | tee submit/run_logs/q2_dagger_hopper.log
```

The script adds the `q2_` prefix by itself when `--do_dagger` is set, so the TensorBoard folders are `data/q2_dagger_ant_Ant-v2_<timestamp>/` and `data/q2_dagger_hopper_Hopper-v2_<timestamp>/`.

To pull out the numbers, one line per iteration:

```bash
grep -H -E "Eval_AverageReturn|Eval_StdReturn" submit/run_logs/q2_dagger_*.log
```

To build Figure 2:

```bash
python submit/plot_q2.py
```

This writes `q2_dagger.pdf` next to the report, with Ant on the left and Hopper on the right. Points are the mean return per iteration and error bars are the standard deviation. The expert line comes from the DAgger log and the BC line from the Part 3 logs, so run Part 3 first. Both y axes start at zero.
