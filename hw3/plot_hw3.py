"""Make the HW3 report figures from the TensorBoard logs.

Usage: activate the rob831 conda environment, go to the directory that
contains plot_hw3.py, and do this command:
    python plot_hw3.py

The script reads the logs from run_logs/. If run_logs/ does not exist, the
script reads the logs from data/. The script writes each figure to figures/
in .pdf format and in .png format:
  q1_dqn_vs_ddqn  Mean and standard deviation of 3 seeds. Solid lines show
                  Train_AverageReturn. Dashed lines show Train_BestReturn.
  q2_cartpole     Raw Eval_AverageReturn value at each iteration.
  q3_pendulum     Raw Eval_AverageReturn value at each iteration.
"""
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'run_logs')
if not os.path.isdir(DATA):
    DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200)
    plt.close(fig)


def run_dir(exp_name):
    pattern = re.compile(re.escape(exp_name) + r'_[A-Za-z]+-v\d+_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$')
    matches = sorted(d for d in os.listdir(DATA) if pattern.match(d))
    if not matches:
        raise FileNotFoundError('no run folder for ' + exp_name)
    return os.path.join(DATA, matches[-1])


def scalars(exp_name, y_tag, x_tag):
    event = glob.glob(os.path.join(run_dir(exp_name), 'events*'))[0]
    ea = EventAccumulator(event, size_guidance={'scalars': 0})
    ea.Reload()
    xs = {e.step: e.value for e in ea.Scalars(x_tag)}
    pts = [(xs[e.step], e.value) for e in ea.Scalars(y_tag) if e.step in xs]
    x, y = zip(*pts)
    return np.array(x), np.array(y)


def q1():
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for name, label, color in [('q1_dqn', 'DQN', 'C0'), ('q1_doubledqn', 'Double DQN', 'C1')]:
        for tag, ls, alpha, kind in [('Train_AverageReturn', '-', 0.25, 'average return'),
                                     ('Train_BestReturn', '--', 0.12, 'best return')]:
            runs = [scalars(f'{name}_{s}', tag, 'Train_EnvstepsSoFar') for s in (1, 2, 3)]
            # The first logged step can be different for each seed. Use only the steps that all seeds have.
            grid = sorted(set.intersection(*[set(x.tolist()) for x, _ in runs]))
            ys = np.stack([np.interp(grid, x, y) for x, y in runs])
            mean, std = ys.mean(0), ys.std(0)
            ax.plot(grid, mean, color=color, ls=ls, label=f'{label}, {kind}')
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=alpha, lw=0)
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Return (Train_AverageReturn / Train_BestReturn)')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title('LunarLander-v3: DQN vs Double DQN (mean $\\pm$ std over 3 seeds)')
    fig.tight_layout()
    save(fig, 'q1_dqn_vs_ddqn')


def ac(exp_name, title, fname, target):
    x, y = scalars(exp_name, 'Eval_AverageReturn', 'Train_EnvstepsSoFar')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(y)), y, marker='.', ms=3)
    ax.axhline(target, ls='--', color='gray', lw=1, label=f'target ({target})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval_AverageReturn')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    save(fig, fname)
    print(f'{exp_name}: final {y[-1]:.1f}, max {y.max():.1f}, mean of last 10 {y[-10:].mean():.1f}')


if __name__ == '__main__':
    q1()
    ac('q2_10_10', 'CartPole-v0 actor-critic (ntu=10, ngsptu=10)', 'q2_cartpole', 200)
    ac('q3_10_10', 'InvertedPendulum-v4 actor-critic (ntu=10, ngsptu=10)', 'q3_pendulum', 1000)
    print('figures written to', OUT)
