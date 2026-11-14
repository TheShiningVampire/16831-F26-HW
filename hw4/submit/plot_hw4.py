"""Make the HW4 report figures from the TensorBoard logs.

Usage: activate the rob831 conda environment, go to the directory that
contains plot_hw4.py, and do this command:
    python plot_hw4.py

The script reads the logs from run_logs/. If run_logs/ does not exist, the
script reads the logs from rob831/data/. The script writes each figure to
figures/ in .pdf format and in .png format:
  q1_<run>        Copy of itr_0_predictions.png for each Q1 run.
  q2_obstacles    Train_AverageReturn and Eval_AverageReturn after 1 iteration.
  q3_<env>        Train_AverageReturn and Eval_AverageReturn at each iteration.
  q4_<parameter>  Eval_AverageReturn at each iteration for each value.
"""
import glob
import os
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'run_logs')
if not os.path.isdir(DATA):
    DATA = os.path.join(HERE, 'rob831', 'data')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
MARKERS = ['o', 's', '^']

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#8a8a85',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.frameon': False,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200)
    plt.close(fig)


def run_dir(exp_name):
    pattern = re.compile(r'hw4_' + re.escape(exp_name)
                         + r'_[a-z]+-hw4_part1-v\d+_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$')
    matches = sorted(d for d in os.listdir(DATA) if pattern.match(d))
    if not matches:
        raise FileNotFoundError('no run folder for ' + exp_name)
    return os.path.join(DATA, matches[-1])


def scalars(exp_name, tag):
    event = glob.glob(os.path.join(run_dir(exp_name), 'events*'))[0]
    ea = EventAccumulator(event, size_guidance={'scalars': 0})
    ea.Reload()
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def style(ax, title, ylabel='Average return'):
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.8)


def q1():
    for name in ('q1_cheetah_n500_arch1x32', 'q1_cheetah_n5_arch2x250',
                 'q1_cheetah_n500_arch2x250'):
        src = os.path.join(run_dir(name), 'itr_0_predictions.png')
        shutil.copy(src, os.path.join(OUT, name + '.png'))


def q2():
    exp = 'q2_obstacles_singleiteration'
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for i, (tag, label) in enumerate([('Train_AverageReturn', 'Train_AverageReturn (random actions)'),
                                      ('Eval_AverageReturn', 'Eval_AverageReturn (MPC)')]):
        x, y = scalars(exp, tag)
        ax.scatter(x, y, s=80, color=COLORS[i], marker=MARKERS[i], label=label,
                   edgecolor='white', linewidth=1.5, zorder=3)
        ax.annotate(f'{y[0]:.1f}', (x[0], y[0]), xytext=(10, -4),
                    textcoords='offset points', fontsize=9, color='#333333')
    ax.set_xticks([0])
    ax.set_xlim(-0.5, 0.5)
    style(ax, 'Obstacles: random actions versus MPC after 1 iteration')
    ax.legend(loc='center right', fontsize=9)
    fig.tight_layout()
    save(fig, 'q2_obstacles')


def q3():
    for env, title in [('obstacles', 'Obstacles'), ('reacher', 'Reacher'),
                       ('cheetah', 'HalfCheetah')]:
        exp = 'q3_' + env
        fig, ax = plt.subplots(figsize=(8, 4.3))
        for i, tag in enumerate(['Eval_AverageReturn', 'Train_AverageReturn']):
            x, y = scalars(exp, tag)
            ax.plot(x, y, color=COLORS[i], marker=MARKERS[i], label=tag)
        style(ax, f'{title}: MBRL with on-policy data collection')
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        save(fig, exp)


def q4():
    groups = [
        ('horizon', 'planning horizon', [('horizon5', 'H = 5'), ('horizon15', 'H = 15'),
                                         ('horizon30', 'H = 30')]),
        ('numseq', 'number of candidate action sequences',
         [('numseq100', 'N = 100'), ('numseq1000', 'N = 1000')]),
        ('ensemble', 'ensemble size', [('ensemble1', '1 model'), ('ensemble3', '3 models'),
                                       ('ensemble5', '5 models')]),
    ]
    for key, desc, runs in groups:
        fig, ax = plt.subplots(figsize=(8, 4.3))
        for i, (suffix, label) in enumerate(runs):
            x, y = scalars('q4_reacher_' + suffix, 'Eval_AverageReturn')
            ax.plot(x, y, color=COLORS[i], marker=MARKERS[i], label=label)
        style(ax, f'Reacher: effect of {desc}', ylabel='Eval_AverageReturn')
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        save(fig, 'q4_reacher_' + key)


if __name__ == '__main__':
    for fn in (q1, q2, q3, q4):
        try:
            fn()
        except (FileNotFoundError, KeyError, IndexError) as err:
            print(f'{fn.__name__}: skipped ({err})')
