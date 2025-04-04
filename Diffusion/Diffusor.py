import math

import numpy as np
from matplotlib import pyplot as plt

from Scheduler import Scheduler


class Diffusor:
    def __init__(self, total_t: int, mode: str = 'cosine', params: dict = None):
        self.total_t = total_t
        self.scheduler = Scheduler(mode, params)

    def noise_images(self, images_array: np.ndarray):
        """
        Function for noising the images array for one random step of diffusion
        :param images_array: array with data
        :return: noised images list, steps of diffusion list , delta noise list
        """
        i_list = []
        x_t_list = []
        noise_t_list = []
        for im in images_array:
            noise = np.random.normal(0, 0.1, im.ravel().shape[0]).reshape(*im.shape)
            t = np.random.randint(1, self.total_t)  # select random diffusion step from T
            i = t / self.total_t
            beta = self.scheduler.step(i)
            i_list.append(i)
            x_t = self.noise_scheme(im, noise, beta)
            # to present noise of current step we calculate noise of previous step
            x_t_prev = self.noise_scheme(im, noise, self.scheduler.step((t - 1) / self.total_t))
            x_t_list.append(x_t)
            noise_t_list.append(x_t - x_t_prev)
            '''plt.imshow(x_t[0])
            plt.colorbar()
            plt.show()
            plt.imshow(x_t_prev[0])
            plt.colorbar()
            plt.show()
            plt.imshow(x_t[0]-x_t_prev[0])
            plt.colorbar()
            plt.show()'''

        return np.array(x_t_list), np.array(i_list), np.array(noise_t_list)

    def denoise_image(self, x_t: np.ndarray, t_noise: np.ndarray):
        """
        Function for removing one step of diffusion
        :param x_t: noised image at t step of diffusion
        :param t_noise: noise part on t step of diffusion (predicted noise)
        :return: denoised for one step of diffusion image
        """
        image = x_t - t_noise
        return image

    @staticmethod
    def noise_scheme(image: np.ndarray, noise: np.ndarray, i: float, b: float = 1):
        """
        Function for applying noise to image by equation
        :param b: scaling factor for clear image
        :param image: array with data
        :param noise: gaussian noise array
        :param i: normalized step of diffusion
        :return: noised data on i step of diffusion
        """
        x_t = i ** 0.5 * b * image + (1 - i) ** 0.5 * noise
        return x_t


