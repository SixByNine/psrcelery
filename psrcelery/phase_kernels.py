
import numpy as np

class GaussianPhaseKernel:
    """
    Computes the Fourier terms for a Gaussian phase kernel.
    """
    parameter_names = ("log10_width",)

    def __init__(self, nc):
        self.nc = nc

    def theta0(self, params):
        log10_width = params[0]
        width = 10 ** log10_width
        gauss_scale = np.sqrt(2 * np.pi * width ** 2)
        return gauss_scale

    def __call__(self, params):
        log10_width = params[0]
        width = 10 ** log10_width

        gauss_denominator = 2 * np.pi ** 2 * width ** 2
        gauss_scale = np.sqrt(2 * np.pi * width ** 2)
        n = np.arange(self.nc) + 1
        return n, gauss_scale * np.ones_like(n) * np.exp(-gauss_denominator * n ** 2)
