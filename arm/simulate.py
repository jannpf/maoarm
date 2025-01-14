import random
import json
import math
import sys
import select
import re
from maoarm.arm.cat import Cat

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

gaussians = json.load(open('spot.json', 'r'))


def plot_arrow(start, end):
    plt.arrow(start[0], start[1], end[0] - start[0], end[1]-start[1],
              shape='full', color='r', head_length=0.05, head_width=0.05, length_includes_head=True)
    # , lw=d['MAG']/2., zorder=0)


def parse_input():
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not rlist:
        return None
    cmd = sys.stdin.readline().strip()
    match = re.match(r'\(\-?\d+,\-?\d+\)', cmd)
    # match = re.match(r'([av])(\=)([\+\-])(\d)+', cmd)
    if match:
        v, a = eval(cmd)
        # ldict = {'v': 0, 'a': 0}
        # exec(cmd, globals(), ldict)
        # v,a = ldict['v'], ldict['a']
        print((v, a))
        return v/100, a/100

    return None


def plot_gaussian(ax, mx, my, sx, sy, rho):
    """
    Plot 1-, 2-, and 3-sigma ellipse contours for a correlated 2D Gaussian
    defined by (mx, my, sx, sy, rho) on the given Axes `ax`.
    """
    # Build the 2x2 covariance matrix
    cov = np.array([
        [sx*sx,       rho*sx*sy],
        [rho*sx*sy,   sy*sy     ]
    ])
    
    # We'll plot the 1-, 2-, 3-sigma ellipses
    for n_std in [1, 2, 3]:
        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)
        # Sort by eigenvalue descending
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        
        # Ellipse angle in degrees
        # the eigenvector for the largest eigenvalue
        theta = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        
        # ellipse axes (2 * sqrt(eigenvalue) for 1-sigma)
        # multiply by n_std for n-sigma
        width  = 2 * n_std * np.sqrt(vals[0])
        height = 2 * n_std * np.sqrt(vals[1])
        
        ellipse = Ellipse(
            (mx, my),
            width=width,
            height=height,
            angle=theta,
            fill=False,
            color='black',
            lw=0.5,
            alpha=0.3
        )
        ax.add_patch(ellipse)


if __name__ == "__main__":
    cat = Cat(valence=0.0, arousal=0.0,
              proposal_sigma=0.2, gaussians=gaussians)
    num_steps = 1000

    # plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence (Negative -> Positive)")
    ax.set_ylabel("Arousal (Low -> High)")
    ax.set_title("Mood Simulation")
    scatter_plot = ax.scatter([], [], c='blue', alpha=0.5)
    cbar = plt.colorbar(scatter_plot, ax=ax)
    cbar.set_label("Step")
    scatter_plot.set_clim(0, num_steps)

    for g in gaussians:
        mx, my = g['mu']
        sx, sy = g['sigma_x'], g['sigma_y']
        rho = g['rho']
        plot_gaussian(ax, mx, my, sx, sy, rho)

    trace = []
    for _ in range(num_steps):
        cat.update_mood()

        offset = parse_input()
        if offset:
            plot_arrow(
                cat.mood, (cat.valence+offset[0], cat.arousal+offset[1]))
            cat.valence += offset[0]
            cat.arousal += offset[1]

        trace.append(cat.mood)
        scatter_plot.set_offsets(trace)
        scatter_plot.set_array(np.arange(len(trace)))

        plt.pause(0.005)

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
