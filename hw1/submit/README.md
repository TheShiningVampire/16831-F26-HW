# HW1: Imitation Learning

This is my code and experiment logs for HW1 (behavioral cloning and DAgger).

All the commands I used are in `RUN.md`, in the same order as the report. Run them from the `hw1/` folder of the course repo with the `rob831` conda environment active. Every run uses seed 1, so running a command again should give the same numbers as in my report.

What is in here:

- `rob831/` is the completed code. It has the same layout as the starter code, with the TODO parts filled in.
- `run_logs/` has one TensorBoard folder per run, plus a `.log` file with what each run printed. Folders starting with `q1_` are from Question 1 and `q2_` are from Question 2. The ones named `q1_p4_...` are the Part 4 experiment.
- `plot_part4.py` and `plot_q2.py` make Figure 1 and Figure 2 of the report from the `.log` files.
- `RUN.md` has the commands.

I used Ant-v2 and Hopper-v2 for the BC and DAgger comparisons. Video logging was turned off for all runs.
