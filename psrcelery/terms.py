import jax
import celerite2.jax
import jax.numpy as jnp


class CeleryTerm(celerite2.jax.terms.Term):
    def __init__(self, parameter_names,bounds):
        self.nparams=len(parameter_names)
        self.parameter_names=parameter_names
        lowbounds=[]
        hibounds=[]
        for parameter in parameter_names:
            lowbounds.append(bounds[parameter][0])
            hibounds.append(bounds[parameter][1])
        self.lowbounds=jnp.array(lowbounds)
        self.hibounds=jnp.array(hibounds)
        self.param_mask=None
        self.params=(self.hibounds+self.lowbounds)/2
    def set_param_mask(self, allparams):
        self.param_mask = jnp.array([True if param in self.parameter_names else False for param in allparams])

    def set_params(self,params):
        self.params=params[self.param_mask]

    
    


class Matern32Term(CeleryTerm):
    def __init__(self,bounds,eps=1e-5):
        parameter_names = ("log_sigma", "log_rho")
        super(Matern32Term, self).__init__(parameter_names,bounds)
        log_sigma, log_rho = self.params
        sigma = jnp.exp(log_sigma)
        rho = jnp.exp(log_rho)
        self.matern=celerite2.jax.terms.Matern32Term(sigma=sigma,rho=rho,eps=eps)

    def get_coefficients(self):
        log_sigma, log_rho = self.params
        self.matern.sigma = jnp.exp(log_sigma)
        self.matern.rho = jnp.exp(log_rho)
        return self.matern.get_coefficients()


class ExpTerm(CeleryTerm):
    """
    A simple exponential kernel.
    """
    def __init__(self,bounds):
        parameter_names = ("log_amp", "length")
        super(ExpTerm, self).__init__(parameter_names,bounds)


    def get_coefficients(self):
        log_amp, length = self.params
        e = jnp.empty(0)
        return (
            jnp.atleast_1d(jnp.exp(log_amp)), jnp.atleast_1d(1 / length),e,e,e,e
        )


class ExpTermLog(CeleryTerm):
    """
    A simple exponential kernel.
    """
    def __init__(self,bounds):
        parameter_names = ("log_amp", "log10_length")
        super(ExpTermLog, self).__init__(parameter_names,bounds)


    def get_coefficients(self):
        log_amp, log10_length = self.params
        length = 10**log10_length
        e = jnp.empty(0)
        return (
            jnp.atleast1d(jnp.exp(log_amp)), jnp.atleast_1d(1 / length),e,e,e,e
        )


class SimpleProfileTerm(CeleryTerm):
    """
    This class defines the 'simple as possible' kernel, with a Gaussian in the
    phase direction and an exponential in the time direction.
    """


    def __init__(self, ncoef, bounds):
        parameter_names = ("log_amp", "log10_width", "log10_length")
        super(SimpleProfileTerm, self).__init__(parameter_names,bounds)
        self.ncoef = ncoef


    def get_coefficients(self):
        log_amp, log_width, log10_length = self.params
        width = 10 ** log_width
        conc = 10**-log10_length
        zero_coef = jnp.sqrt(2 * jnp.pi * width ** 2)
        omega0 = 2 * jnp.pi

        nc = self.ncoef
        gauss_denominator = 2 * jnp.pi ** 2 * width ** 2
        gauss_scale = 2 * jnp.sqrt(2 * jnp.pi * width ** 2)
        n = jnp.arange(nc) + 1

        d = jnp.array(omega0 * n)

        a = jnp.exp(log_amp) * gauss_scale * jnp.ones_like(n) * jnp.exp(-gauss_denominator * n ** 2)
        b = jnp.zeros_like(n)
        c = jnp.ones_like(n) * conc

        return (
            jnp.atleast_1d(jnp.exp(log_amp) * zero_coef), jnp.atleast_1d(conc), a, b, c, d
        )


def make_custom_profile_term(phase_kernel, time_kernel, omega0=2 * jnp.pi, optimise_tau_real_only=False):
    pnames = time_kernel.parameter_names + phase_kernel.parameter_names

    #@todo: Work out how to check if the time kernel is real or complex.

    class HalfProfileTerm(CeleryTerm):


        def __init__(self,bounds):
            super(HalfProfileTerm, self).__init__(pnames,bounds)


        def set_param_mask(self, allparams):
            phase_kernel.set_param_mask(allparams)
            time_kernel.set_param_mask(allparams)
            return super().set_param_mask(allparams)

        def set_params(self, params):
            phase_kernel.set_params(params)
            time_kernel.set_params(params)
        
        def get_coefficients(self):

            p, r, _ ,_ ,_,_ = time_kernel.get_coefficients()
            ra = p * phase_kernel.theta0()
            rc = r

            n, theta_n = phase_kernel()
            a = 2 * p * theta_n
            b = jnp.zeros_like(n)
            c = r * jnp.ones_like(n)
            d = omega0 * n

            return (jnp.atleast_1d(ra),jnp.atleast_1d(rc),
                    jnp.atleast_1d(a),jnp.atleast_1d(b),jnp.atleast_1d(c),jnp.atleast_1d(d))

    class FullProfileTerm(CeleryTerm):

        def __init__(self,bounds):
            super(FullProfileTerm, self).__init__(pnames,bounds)
            phase_kernel.set_param_mask(pnames)
            time_kernel.set_param_mask(pnames)

        def set_params(self, params):
            phase_kernel.set_params(params)
            time_kernel.set_params(params)

        def set_param_mask(self, allparams):
            phase_kernel.set_param_mask(allparams)
            time_kernel.set_param_mask(allparams)
            return super().set_param_mask(allparams)

        def get_coefficients(self):
            # Complex Terms...

            rp, rr , p, q, r, s = time_kernel.get_coefficients()


            theta0 = phase_kernel.theta0()
            e = jnp.empty(0)

            n, theta_n = phase_kernel()
            theta_0 = phase_kernel.theta0()

            a = jnp.concatenate((p * theta_n, p * theta_n, jnp.atleast_1d(p * theta_0)))
            b = jnp.concatenate((q * theta_n, q * theta_n, jnp.atleast_1d(q * theta_0)))
            c = r * jnp.ones(2 * len(n) + 1)
            d = jnp.concatenate((s - omega0 * n, s + omega0 * n, jnp.atleast_1d(s)))
            return (e,e, a, b, c, d)

                

    if optimise_tau_real_only:
        return HalfProfileTerm
    else:
        return FullProfileTerm
    

class CeleryWhiteTransform:
    def __init__(self, parameter_names,bounds):
        self.nparams=len(parameter_names)
        self.parameter_names=parameter_names
        lowbounds=[]
        hibounds=[]
        for parameter in parameter_names:
            lowbounds.append(bounds[parameter][0])
            hibounds.append(bounds[parameter][1])
        self.lowbounds=jnp.array(lowbounds)
        self.hibounds=jnp.array(hibounds)
        self.param_mask=None
        self.params=(self.hibounds+self.lowbounds)/2

    def set_param_mask(self, allparams):
        self.param_mask = jnp.array([True if param in self.parameter_names else False for param in allparams])

    def set_params(self,params):
        self.params=params[self.param_mask]


class EquadWhiteTransform(CeleryWhiteTransform):
    def __init__(self,bounds):
        parameter_names = ("log_equad",)
        super(EquadWhiteTransform, self).__init__(parameter_names,bounds)

    def __call__(self,yerr):
        return jnp.sqrt(jnp.exp(2*self.params[0]) + yerr ** 2)