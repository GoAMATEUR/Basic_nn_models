import abc
import math
import torch
import numpy as np

class SDEBase(abc.ABC):
    def __init__(self, T):
        self.T = T
        self.dt = 1 / T

    @abs.abstractmethod
    def drift(self, x_t, t):
        pass

    @abs.abstractmethod
    def dispersion(self, x_t, t):
        pass
    
    def dw(self, x):
        return torch.rand_like(x) * math.sqrt(self.dt)

    def _single_step_forward(self, x_t, t):
        dx = self.drift(x_t, t) * self.dt + self.dispersion(x_t, t) * self.dw(x_t)
        return x_t + dx

    def forward(self, x_0):
        x_t = x_0
        for t in range(self.T):
            x_t = self.single_step_forward(x_t, t)
        return x_t

    def _single_reverse_sde(self, x_t, t, score):
        return x_t - (self.drift(x_t, t) * self.dispersion(x_t, t)**2 * score) * self.dt + self.dispersion(x_t, t) * self.dw(x_t) * (t > 0) # no dispersion at t=0

    def _single_reverse_ode(self, x_t, t, score):
        return x_t - (self.drift(x_t, t) * 0.5 * self.dispersion(x_t, t)**2 * score) * self.dt * (t > 0)

    def reverse(self, x_t, score, mode):
        """Reverse ODE

        Args:
            x_T (torch.Tensor): final state
            score (torch.Tensor): score
            mode (str): 'score' or 'state'
            
        """
        x_0 = x_t
        for t in reversed(range(self.T)):
            score_value = score(x_t, t)
            if mode == "sde":
                x_0 = self._single_reverse_sde(x_t, t, score)
            elif mode == "ode":
                x_0 = self._single_reverse_ode(x_t, t, score)


def vp_beta_schedule(T, dtype = torch.float32):
    t = np.arange(1, T + 1)
    b_max = 10.
    b_min = .1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    return torch.tensor(1-alpha, dtype=dtype)

class MySDE(SDEBase):
    def __init__(self, T, schedule):
        super().__init__(T)
        
        if schedule == "vp":
            self.thetas = vp_beta_schedule(T)
        self.sigmas = torch.sqrt(self.thetas)

    def drift(self, x_t, t):
        return -self.thetas[t] * x_t

    def dispersion(self, x_t, t):
        return self.sigmas[t]
        
