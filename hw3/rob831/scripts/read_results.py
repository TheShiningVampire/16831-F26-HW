import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_section_results(file, x_tag='Train_EnvstepsSoFar', y_tag='Train_AverageReturn'):
    """
        Read the (x, y) scalar values logged to a tensorboard event file.
        Only requires the `tensorboard` package (no tensorflow needed).
    """
    ea = EventAccumulator(file, size_guidance={'scalars': 0})  # 0 = load all scalars
    ea.Reload()
    # align x and y by logging step (y may be missing at early steps, e.g. before
    # the first DQN episode finishes)
    x_by_step = {e.step: e.value for e in ea.Scalars(x_tag)}
    X, Y = [], []
    for e in ea.Scalars(y_tag):
        if e.step in x_by_step:
            X.append(x_by_step[e.step])
            Y.append(e.value)
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    parser.add_argument('--y_tag', type=str, default='Train_AverageReturn', help='e.g. Train_AverageReturn (DQN) or Eval_AverageReturn (actor-critic)')
    args = parser.parse_args()

    logdir = os.path.join(args.logdir, 'events*')
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile, y_tag=args.y_tag)

    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
