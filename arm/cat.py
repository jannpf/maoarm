import random
import math


class Cat:
    def __init__(self, valence, arousal, proposal_sigma, gaussians):
        self.valence = valence
        self.arousal = arousal
        self.proposal_sigma = proposal_sigma
        self.gaussians = gaussians

    @property
    def mood(self) -> tuple[float, float]:
        return self.valence, self.arousal

    @staticmethod
    def uncorr_gaussian_pdf(x, y, g):
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
    def corr_gaussian_pdf(x, y, g):
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

    def _mood_target_density(self, x, y):
        """
        Returns a probability density for the gaussian mixture
        """
        total = 0.0
        for g in self.gaussians:
            exponent = self.corr_gaussian_pdf(x, y, g)
            total += exponent

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

    def update_mood(self):
        """
        Probabilistically update the mood state (valence, arousal), with one iteration of MH.
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
        else:
            # reject
            pass
