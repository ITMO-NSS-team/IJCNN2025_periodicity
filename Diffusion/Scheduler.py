import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Scheduler:
    def __init__(self, mode='cosine', params: dict = None):
        """
        Class for initialization noise scheduler with gamma function

        :param mode: gamma function name - 'cosine', 'linear', 'sigmoid' are available
        :param params: dict with parameters for cosine or sigmoid gamma function - "start", "end", "tau"
        """
        self.mode = mode
        self.params = params

    def step(self, t: float):
        """
        Function for calculating gamma coefficient tor diffusion step t depending on self.mode
        :param t: normalized step of diffusion, float [0, 1]
        :return: value of gamma function on step t
        """
        if self.mode == 'linear':
            return self.linear_schedule(t)
        if self.mode == 'cosine':
            if self.params is not None:
                return self.cosine_schedule(t, **self.params)
            else:
                return self.cosine_schedule(t)
        if self.mode == 'sigmoid':
            if self.params is not None:
                return self.sigmoid_schedule(t, **self.params)
            else:
                return self.sigmoid_schedule(t)

    def get_curve(self, total_t: int, plot=False):
        """
        Function for calculating values of gamma function for all staps of diffusion
        :param total_t: total number of diffusion steps, T
        :param plot: is there need to plot values of gamma function
        :return: normalized steps of diffusion, values of gamma function per step
        """
        steps = np.linspace(0, 1, total_t)
        if self.mode == 'linear':
            values = [self.linear_schedule(t) for t in steps]
        if self.mode == 'cosine':
            if self.params is not None:
                values = [self.cosine_schedule(t, **self.params) for t in steps]
            else:
                values = [self.cosine_schedule(t) for t in steps]
        if self.mode == 'sigmoid':
            if self.params is not None:
                values = [self.sigmoid_schedule(t, **self.params) for t in steps]
            else:
                values = [self.sigmoid_schedule(t) for t in steps]
        if plot:
            plt.plot(steps, values)
            plt.title(f'{self.mode} gamma function for T={total_t}')
            plt.xlabel('t')
            plt.ylabel('Gamma function value')
            plt.show()
        return steps, values

    @staticmethod
    def linear_schedule(t, clip_min=1e-9):
        """
        A gamma function that simply is 1-t
        """
        return np.clip(1 - t, clip_min, 1.)

    @staticmethod
    def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
        """
        A gamma function based on cosine function
        """
        v_start = math.cos(start * math.pi / 2) ** (2 * tau)
        v_end = math.cos(end * math.pi / 2) ** (2 * tau)
        output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
        output = (v_end - output) / (v_end - v_start)
        return np.clip(output, clip_min, 1.)

    @staticmethod
    def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
        """
        A gamma function based on sigmoid function
        """
        v_start = sigmoid(start / tau)
        v_end = sigmoid(end / tau)
        output = sigmoid((t * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start)
        return np.clip(output, clip_min, 1.)
