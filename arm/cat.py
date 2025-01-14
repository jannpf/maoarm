import random
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.axes import Axes


class Cat:
    def __init__(self, gaussians: list[dict], valence: float = 0.0, arousal: float = 0.0, proposal_sigma: float = 0.2, plot: bool = False, maxtracelen: int = 1000):
        """
        Initializes a new Cat with the given character profile.

        Args:
            gaussians:          The gaussian distributions describing the cats behavior, each as dict
            valence:            The initial valence, between -1 and 1
            arousal:            The initial arousal, between -1 and 1
            proposal_sigma:     The standard deviation of the proposal distribution for Metropolis Hastings
            plot:               Whether or not to plot the mood trace of the cat
            maxtracelen:        In case of plot=True, this determines how many values are displayed concurrently at max 
        """
        self.valence = valence
        self.arousal = arousal
        self.proposal_sigma = proposal_sigma
        self.gaussians = gaussians
        self.plot = plot
        self.mood_trace = None

        if plot:
            self.initialize_plotting()
            self.mood_trace = deque(maxlen=maxtracelen)

    @property
    def mood(self) -> tuple[float, float]:
        return self.valence, self.arousal

    @staticmethod
    def uncorr_gaussian_pdf(x: float, y: float, g: dict):
        """
        Density at (x,y) for gaussian g. Assumes no correlation between components.
        """
        w = g['weight']
        mx, my = g['mu']
        sx, sy = g['sigma_x'], g['sigma_y']

        # gaussian exponent (without constant factors - irrelevant for MH)
        dx = (x - mx)**2 / (2 * sx**2)
        dy = (y - my)**2 / (2 * sy**2)
        exponent = math.exp(- (dx + dy))

        # weighted exponent
        return w * exponent

    @staticmethod
    def corr_gaussian_pdf(x: float, y: float, g: dict):
        """
        Density at (x,y) for gaussian g. Correlation between components can be passed with key 'rho'.
        """
        w = g['weight']
        mx, my = g['mu']
        sx, sy = g['sigma_x'], g['sigma_y']
        rho = g['rho']

        # Translate point by mean
        dx = x - mx
        dy = y - my
        # 1 - rho^2
        one_minus_rho2 = 1.0 - (rho**2)
        if one_minus_rho2 <= 0:
            # fallback to avoid numeric issues if |rho| ~ 1
            return 0.0

        # The exponent in the correlated 2D Gaussian:
        #    E = 1/(2*(1-rho^2)) * [ (dx^2 / sx^2) - 2*rho*(dx*dy)/(sx*sy) + (dy^2 / sy^2 ) ]
        term1 = (dx*dx)/(sx*sx)
        term2 = -2.0*rho*(dx*dy)/(sx*sy)
        term3 = (dy*dy)/(sy*sy)

        exponent = 0.5 * (1.0/one_minus_rho2) * (term1 + term2 + term3)

        # exp(-E).
        return w * math.exp(-exponent)

    def _mood_target_density(self, x: float, y: float):
        """
        Returns a probability density for the gaussian mixture at (x,y)
        """
        total = 0.0
        for g in self.gaussians:
            density = self.corr_gaussian_pdf(x, y, g)
            total += density

        return total

    def _propose_new_mood(self) -> tuple[float, float]:
        """
        Propose a new (valence, arousal) tuple by sampling from a
        Gaussian centered at the current mood with std = self.propsal_sigma.
        Bound to [-1, 1].
        """
        v_new = random.gauss(self.valence, self.proposal_sigma)
        a_new = random.gauss(self.arousal, self.proposal_sigma)
        # Bound to [-1, 1]
        v_new = max(-1, min(1, v_new))
        a_new = max(-1, min(1, a_new))

        return v_new, a_new

    def mood_iteration(self):
        """
        Mood drift. Probabilistically update the mood state (valence, arousal), with one iteration of MH.
        """
        p_current = self._mood_target_density(self.valence, self.arousal)

        # Propose new point
        valence_new, arousal_new = self._propose_new_mood()
        p_proposed = self._mood_target_density(valence_new, arousal_new)

        if p_current == 0:  # avoid divide by zero
            alpha = 1.0
        else:
            alpha = p_proposed / p_current

        # Accept or reject
        if random.random() < alpha:
            # accept
            self.valence = valence_new
            self.arousal = arousal_new
            if self.plot:
                self.mood_trace.append(self.mood)
                self.scatter_plot.set_offsets(self.mood_trace)
        else:
            # reject
            pass

    def override_mood(self, v_new: float, a_new: float) -> None:
        """
        Use this function to update the mood of the cat. 
        If plotting is enabled, the override will be displayed as red arrow 
        and the trace of the visualization updated.

        Args:
            v_new:    The new valence value to use
            a_new:    The new arousal value to use
        """
        self.valence = max(-1, min(1, v_new))
        self.arousal = max(-1, min(1, a_new))

        if self.plot:
            self.plot_arrow(self.mood_trace[-1], self.mood)
            self.mood_trace.append(self.mood)
            self.scatter_plot.set_offsets(self.mood_trace)

    @staticmethod
    def plot_gaussian(ax: Axes, mx: float, my: float, sx: float, sy: float, rho: float):
        """
        Plot 1-, 2-, and 3-sigma ellipse contours for a correlated 2D Gaussian
        defined by (mx, my, sx, sy, rho) on the given Axes `ax`.
        """
        # Build the 2x2 covariance matrix
        cov = np.array([
            [sx*sx,       rho*sx*sy],
            [rho*sx*sy,   sy*sy]
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
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

            # ellipse axes (2 * sqrt(eigenvalue) for 1-sigma)
            # multiply by n_std for n-sigma
            width = 2 * n_std * np.sqrt(vals[0])
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

    def plot_arrow(self, start: tuple, end: tuple):
        self.ax.arrow(
            start[0],
            start[1],
            end[0]-start[0],
            end[1]-start[1],
            shape='full',
            color='r',
            head_length=0.05,
            head_width=0.05,
            length_includes_head=True
        )

    def initialize_plotting(self):
        # initialize plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel("Valence (Negative -> Positive)")
        self.ax.set_ylabel("Arousal (Low -> High)")
        self.ax.set_title("My Mood")

        # plot 1,2,3 sigma contour lines for all the configured gaussians
        for g in self.gaussians:
            mx, my = g['mu']
            sx, sy = g['sigma_x'], g['sigma_y']
            rho = g['rho']
            self.plot_gaussian(self.ax, mx, my, sx, sy, rho)

        self.scatter_plot = self.ax.scatter([], [], c='blue', alpha=0.5)
