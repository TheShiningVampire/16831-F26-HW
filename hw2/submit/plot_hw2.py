"""Make the report figures from the TensorBoard logs.

Usage (from the folder containing this file, rob831 env active):
    python plot_hw2.py                 # all figures
    python plot_hw2.py q2 q4_search    # only some

Logs are read from run_logs/ next to this file (or data/ if there is no
run_logs/). Figures are written to figures/. Each curve shows the raw
Eval_AverageReturn faintly and a 10-iteration moving average in bold.
"""
import os
import re
import sys

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
WINDOW = 10
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def load(exp_name):
    """Return Eval_AverageReturn for the run named exp_name.

    The logger names a run folder <exp_name>_<EnvName-vN>_<date>_<time>, so
    match the whole exp_name followed by the environment and time stamp, not
    just the prefix (q4_b50000_r0.02 must not pick up q4_b50000_r0.02_rtg).
    """
    pattern = re.compile(re.escape(exp_name) + r'_[A-Za-z]+-v\d+_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$')
    matches = sorted(d for d in os.listdir(DATA) if pattern.match(d))
    if not matches:
        raise FileNotFoundError('no run folder for ' + exp_name)
    if len(matches) > 1:
        print('note: %d runs for %s, using the latest' % (len(matches), exp_name))
    ea = EventAccumulator(os.path.join(DATA, matches[-1]))
    ea.Reload()
    return np.array([e.value for e in ea.Scalars('Eval_AverageReturn')])


def smooth(y):
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = y[max(0, i - WINDOW + 1):i + 1].mean()
    return out


def plot(fname, title, runs, target, legend_cols=1, smoothed_only=False):
    """runs: list of (label, exp_name, color, linestyle)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, exp_name, color, ls in runs:
        y = load(exp_name)
        x = np.arange(len(y))
        if not smoothed_only:
            ax.plot(x, y, color=color, alpha=0.3, lw=0.8)
        ax.plot(x, smooth(y), color=color, lw=2, ls=ls, label=label)
    ax.axhline(target, color='black', ls='--', lw=0.8)
    ax.text(1, target, 'target %d' % target, fontsize=8, va='bottom')
    ax.set_title(title, loc='left')
    xlabel = 'Iteration   (bold: %d-iteration moving average' % WINDOW
    ax.set_xlabel(xlabel + (')' if smoothed_only else ', faint: raw)'))
    ax.set_ylabel('Eval average return')
    ax.grid(alpha=0.3)
    if len(runs) > 1:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  ncol=legend_cols, frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, fname + '.pdf'))
    plt.close(fig)
    print('wrote', fname)


def q1():
    for tag, b in (('sb', 1500), ('lb', 6000)):
        plot('q1_%s_batch' % ('small' if tag == 'sb' else 'large'),
             'CartPole-v0, %s batch (b = %d)' % ('small' if tag == 'sb' else 'large', b),
             [('no rtg, no std. (q1_%s_no_rtg_dsa)' % tag, 'q1_%s_no_rtg_dsa' % tag, COLORS[0], '-'),
              ('rtg, no std. (q1_%s_rtg_dsa)' % tag, 'q1_%s_rtg_dsa' % tag, COLORS[1], '-'),
              ('rtg, standardized (q1_%s_rtg_na)' % tag, 'q1_%s_rtg_na' % tag, COLORS[2], '-')],
             target=200)


def q2():
    plot('q2_b300_r0_01', 'InvertedPendulum-v4, b = 300, lr = 0.01',
         [('', 'q2_b300_r0.01', COLORS[0], '-')], target=1000)


def q2_search():
    runs = []
    lrs = [0.005, 0.01, 0.02, 0.05]
    styles = [':', '--', '-.', '-']
    for i, b in enumerate([100, 200, 300]):
        for lr, ls in zip(lrs, styles):
            runs.append(('b=%d, lr=%g' % (b, lr), 'q2_b%d_r%g' % (b, lr), COLORS[i], ls))
    plot('q2_search', 'InvertedPendulum-v4 search, small batches (color = batch, style = lr)',
         runs, target=1000, legend_cols=3, smoothed_only=True)


def q3():
    plot('q3_lunarlander', 'LunarLanderContinuous-v2 (rtg + baseline)',
         [('', 'q3_b10000_r0.005', COLORS[0], '-')], target=120)


def q4_search():
    runs = []
    for i, b in enumerate([10000, 30000, 50000]):
        for lr, ls in zip([0.005, 0.01, 0.02], [':', '--', '-']):
            runs.append(('b=%d, lr=%g' % (b, lr),
                         'q4_search_b%d_lr%g_rtg_nnbaseline' % (b, lr), COLORS[i], ls))
    plot('q4_search', 'HalfCheetah-v4 search (color = batch, style = lr)',
         runs, target=200, legend_cols=3, smoothed_only=True)


def q4_final():
    for b in (10000, 50000):
        base = 'q4_b%d_r0.02' % b
        plot('q4_final_b%d_r0_02' % b, 'HalfCheetah-v4, b = %d, lr = 0.02' % b,
             [('no rtg, no baseline', base, COLORS[0], '-'),
              ('rtg', base + '_rtg', COLORS[1], '-'),
              ('baseline', base + '_nnbaseline', COLORS[2], '-'),
              ('rtg + baseline', base + '_rtg_nnbaseline', COLORS[4], '-')],
             target=200, legend_cols=4)


def q5():
    plot('q5_hopper', 'Hopper-v4 with GAE (noisy actions)',
         [(r'$\lambda$ = %g' % lam, 'q5_b2000_r0.001_lambda%g' % lam, c, '-')
          for lam, c in zip([0, 0.95, 0.99, 1], [COLORS[0], COLORS[1], COLORS[2], COLORS[4]])],
         target=400, legend_cols=4)


FIGURES = {'q1': q1, 'q2': q2, 'q2_search': q2_search, 'q3': q3,
           'q4_search': q4_search, 'q4_final': q4_final, 'q5': q5}

if __name__ == '__main__':
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()
