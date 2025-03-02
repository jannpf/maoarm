"""
Helper module to experiment with cat mood visualization.
"""

import json
import sys
import select
import re
from moods import Cat

import matplotlib.pyplot as plt

cat_characteristics = json.load(open("moods/cat_characters/spot.json", "r"))
gaussians = cat_characteristics["gaussians"]


def parse_input():
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not rlist:
        return None
    cmd = sys.stdin.readline().strip()
    match = re.match(r"\(\-?\d+,\-?\d+\)", cmd)
    # match = re.match(r'([av])(\=)([\+\-])(\d)+', cmd)
    if match:
        v, a = eval(cmd)
        # ldict = {'v': 0, 'a': 0}
        # exec(cmd, globals(), ldict)
        # v,a = ldict['v'], ldict['a']
        print((v, a))
        return v / 100, a / 100

    return None


if __name__ == "__main__":
    cat = Cat(
        valence=0.0, arousal=0.0, proposal_sigma=0.2, gaussians=gaussians, plot=True
    )
    num_steps = 1000
    trace = []

    for _ in range(num_steps):
        cat.mood_iteration()

        offset = parse_input()
        if offset:
            v_new = cat.valence + offset[0]
            a_new = cat.arousal + offset[1]
            cat.override_mood(v_new, a_new)

        trace.append(cat.mood)
        plt.pause(0.05)

    # x_vals = [m[0] for m in trace]
    # y_vals = [m[1] for m in trace]

    # plt.figure(figsize=(6, 6))
    # plt.scatter(x_vals, y_vals, s=10, alpha=0.5,
    #             c=range(len(x_vals)), cmap='viridis')
    # plt.colorbar(label='Simulation Step')
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.title("Trajectory (Valence vs. Arousal)")
    # plt.xlabel("Valence (Negative --> Positive)")
    # plt.ylabel("Arousal (Low --> High)")
    # plt.grid(True)
    # plt.show()
