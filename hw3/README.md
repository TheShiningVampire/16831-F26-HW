## Setup

You can run this code on your own machine or on Google Colab.

1. **Local option:** Use the same `rob831` conda environment you created for homework 1 (see [installation.md](../hw1/installation.md)). If you completed that installation, you only need to install the extra packages for this homework (Box2D, used by `LunarLander`):

    ```bash
    conda activate rob831
    cd <path_to_hw3>
    pip install -r requirements.txt
    pip install -e .
    ```

    Note that `pip install -e .` points the `rob831` package to `hw3`. If you go back to a previous homework, re-run `pip install -e .` from that homework's folder.

    To test if the installation worked correctly, run
    ```bash
    python -c "import gym; from rob831.infrastructure.dqn_utils import register_custom_envs; register_custom_envs(); env = gym.make('LunarLander-v3'); print(env.reset().shape); env.close()"
    ```
    which should print `(9,)`.

2. **Colab:** The first few sections of the notebooks will install all required dependencies. You can try out the Colab option by clicking the badges below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-F26-HW/blob/main/hw3/rob831/scripts/run_hw3_dqn.ipynb) **Part I (Q-learning)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-F26-HW/blob/main/hw3/rob831/scripts/run_hw3_actor_critic.ipynb)     **Part II (Actor-critic)**

Note: If running in Colab and you pull the repo again or install new packages, you may need to restart the runtime and reconnect before being able to run.

## Complete the code

You will need to implement new routines in the following files for homework 3 part 1 (Q-learning):
- [agents/dqn_agent.py](rob831/agents/dqn_agent.py)
- [critics/dqn_critic.py](rob831/critics/dqn_critic.py)
- [policies/argmax_policy.py](rob831/policies/argmax_policy.py)

and in the following files for part 2 (actor-critic):
- [agents/ac_agent.py](rob831/agents/ac_agent.py)
- [critics/bootstrapped_continuous_critic.py](rob831/critics/bootstrapped_continuous_critic.py)
- [policies/MLP_policy.py](rob831/policies/MLP_policy.py)

The relevant sections are marked with `TODO`. Unlike homework 2, you do not need to copy over any code from previous homeworks.

You may also want to look through [scripts/run_hw3_dqn.py](rob831/scripts/run_hw3_dqn.py) and [scripts/run_hw3_actor_critic.py](rob831/scripts/run_hw3_actor_critic.py) (if running locally) or [scripts/run_hw3_dqn.ipynb](rob831/scripts/run_hw3_dqn.ipynb) and [scripts/run_hw3_actor_critic.ipynb](rob831/scripts/run_hw3_actor_critic.ipynb) (if running on Colab), though you will not need to edit these files beyond changing runtime arguments in the Colab notebook.

See the assignment PDF for more details on what files to edit and which experiments to run.

## Visualizing and plotting results

Logs are written to `hw3/data/`. You can visualize your runs using TensorBoard:
```bash
python -m tensorboard.main --logdir data
```

[scripts/read_results.py](rob831/scripts/read_results.py) shows how to read the logged returns from an event file into Python lists (e.g. to average over seeds and plot with matplotlib):
```bash
python rob831/scripts/read_results.py --logdir data/<your_run_folder>
```
Use `--y_tag Eval_AverageReturn` for the actor-critic runs.
