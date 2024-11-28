
import jax.numpy as jnp

class GaussianPhaseKernel:
    """
    Computes the Fourier terms for a Gaussian phase kernel.
    """

    def __init__(self, nc):
        self.nc = nc
        self.parameter_names=("log10_width",)

    def set_param_mask(self, allparams):
        self.param_mask = jnp.array([True if param in self.parameter_names else False for param in allparams])

    def set_params(self,params):
        self.params=params[self.param_mask]

    def theta0(self):
        log10_width = self.params[0]
        width = 10 ** log10_width
        gauss_scale = jnp.sqrt(2 * jnp.pi * width ** 2)
        return gauss_scale

    def __call__(self):
        log10_width = self.params[0]
        width = 10 ** log10_width

        gauss_denominator = 2 * jnp.pi ** 2 * width ** 2
        gauss_scale = jnp.sqrt(2 * jnp.pi * width ** 2)
        n = jnp.arange(self.nc) + 1
        return n, gauss_scale * jnp.ones_like(n) * jnp.exp(-gauss_denominator * n ** 2)
