import matplotlib.pyplot as plt
import os
from pathlib import Path
import re

'''
Find loss values in the latest log file like the below:

2026-02-19 12:18:13 INFO: Finished STEP 6120/50000, loss = 0.491907 (0.263 sec/batch), lr: 2.000000

Extract them and plot over time.

We also find the LAS scores in lines like:

2026-02-19 12:25:34 INFO: Evaluating on dev set...
2026-02-19 12:25:42 INFO: LAS   MLAS    BLEX
2026-02-19 12:25:42 INFO: 94.27 93.26   94.90

We map these as values to the latest step, and plot them over time as well.
'''

LOGS = Path("logs/")
logs = os.listdir(LOGS)
logs = [log for log in logs if log.startswith("log")]

latest_log = sorted(logs, key=lambda x: os.path.getctime(LOGS / x))[-2]
print(f"Latest log: {latest_log}")

step_loss_dict = {}
step_las_dict = {}

latest_step = 0
with open(LOGS / latest_log, "r") as f:
    for line in f:
        if "Training ended" in line:
            break
        if "Finished STEP" in line and "loss =" in line:
            step_match = re.search(r"STEP (\d+)/\d+", line)
            loss_match = re.search(r"loss = ([\d.]+)", line)
            if step_match and loss_match:
                step = int(step_match.group(1))
                loss = float(loss_match.group(1))
                step_loss_dict[step] = loss
                latest_step = max(latest_step, step)
        if "LAS   MLAS    BLEX" in line:
            continue
        if "INFO: " in line and re.search(r"\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+", line):
            las_match = re.search(r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", line)
            if las_match:
                las = float(las_match.group(1))
                mlas = float(las_match.group(2))
                blex = float(las_match.group(3))
                step_las_dict[latest_step] = (las, mlas, blex)


# Sort by step
steps = sorted(step_loss_dict.keys())
losses = [step_loss_dict[step] for step in steps]

las_steps = sorted(step_las_dict.keys())
las_values = [step_las_dict[step][0] for step in las_steps]

# Plot loss and LAS together with shared x-axis and separate y-axes.
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.plot(steps, losses, label="Loss", color="tab:blue")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2.plot(las_steps, las_values, label="LAS", color="red")
ax2.set_ylabel("LAS", color="red")
ax2.tick_params(axis="y", labelcolor="red")

ax1.set_title("Loss and LAS over Training Steps")
ax1.grid()
fig.tight_layout()
fig.savefig(f"plot/plot_loss_las_{latest_log}.png", dpi=400)