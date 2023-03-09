import celerite
import numpy as np


class ExpTerm(celerite.terms.Term):
    """
    A simple exponential kernel.
    """
    parameter_names = ("log_amp", "length")

    def get_real_coefficients(self, params):
        log_amp, length = params
        return (
            np.exp(log_amp), 1 / length
        )


class SimpleProfileTerm(celerite.terms.Term):
    """
    This class defines the 'simple as possible' kernel, with a Gaussian in the
    phase direction and an exponential in the time direction.
    """
    parameter_names = ("log_amp", "log10_width", "length")

    def __init__(self, *args, **kwargs):
        ncoef = kwargs.pop("ncoef", 32)
        super(SimpleProfileTerm, self).__init__(*args, **kwargs)
        self.ncoef = ncoef

    def get_real_coefficients(self, params):
        log_amp, log_width, length = params
        width = 10 ** log_width
        conc = 1 / length
        zero_coef = np.sqrt(2 * np.pi * width ** 2)
        return (
            np.exp(log_amp) * zero_coef, conc
        )

    def get_complex_coefficients(self, params):
        log_amp, log_width, length = params
        width = 10 ** log_width
        conc = 1 / length
        omega0 = 2 * np.pi

        nc = self.ncoef
        gauss_denominator = 2 * np.pi ** 2 * width ** 2
        gauss_scale = 2 * np.sqrt(2 * np.pi * width ** 2)
        n = np.arange(nc) + 1

        d = omega0 * n

        a = np.exp(log_amp) * gauss_scale * np.ones_like(n) * np.exp(-gauss_denominator * n ** 2)
        b = np.zeros_like(n)
        c = np.ones_like(n) * conc

        return (
            a, b, c, d
        )


def make_custom_profile_term(phase_kernel, time_kernel, omega0=2 * np.pi, optimise_tau_real_only=False):
    pnames = time_kernel.parameter_names + phase_kernel.parameter_names
    print(pnames)

    time_mask = np.array([k in time_kernel.parameter_names for k in pnames], dtype=bool)
    fourier_mask = np.array([k in phase_kernel.parameter_names for k in pnames], dtype=bool)

    def extract_time_args(kwargs):
        return {k: kwargs[k] for k in time_kernel.parameter_names}

    class HalfProfileTerm(celerite.terms.Term):
        parameter_names = pnames

        def __init__(self, *args, **kwargs):
            super(HalfProfileTerm, self).__init__(*args, **kwargs)
            self.timekernel = time_kernel(**extract_time_args(kwargs))

        def get_real_coefficients(self, params):
            time_params = params[time_mask]
            fourier_params = params[fourier_mask]
            p, r = self.timekernel.get_real_coefficients(time_params)
            a = p * phase_kernel.theta0(fourier_params)
            c = r
            return (a, c)

        def get_complex_coefficients(self, params):
            time_params = params[time_mask]
            fourier_params = params[fourier_mask]
            p, r = self.timekernel.get_real_coefficients(time_params)

            n, theta_n = phase_kernel(fourier_params)
            a = 2 * p * theta_n
            b = np.zeros_like(n)
            c = r * np.ones_like(n)
            d = omega0 * n

            return (
                a, b, c, d
            )

    class CmplxProfileTerm(celerite.terms.Term):
        parameter_names = pnames

        def __init__(self, *args, **kwargs):
            super(CmplxProfileTerm, self).__init__(*args, **kwargs)
            self.timekernel = time_kernel(**extract_time_args(kwargs))

        def get_complex_coefficients(self, params):
            time_params = params[time_mask]
            fourier_params = params[fourier_mask]
            p, q, r, s = self.timekernel.get_complex_coefficients(time_params)

            n, theta_n = phase_kernel(fourier_params)
            theta_0 = phase_kernel.theta0(fourier_params)

            a = np.concatenate((p * theta_n, p * theta_n, [p * theta_0]))
            b = np.concatenate((q * theta_n, q * theta_n, [q * theta_0]))
            c = r * np.ones(2 * len(n) + 1)
            d = np.concatenate((s - omega0 * n, s + omega0 * n, [s]))

            return (
                a, b, c, d
            )

    class FullProfileTerm(celerite.terms.Term):
        parameter_names = pnames

        def __init__(self, *args, **kwargs):
            super(FullProfileTerm, self).__init__(*args, **kwargs)
            self.timekernel = time_kernel(**extract_time_args(kwargs))

        def get_real_coefficients(self, params):
            time_params = params[time_mask]
            fourier_params = params[fourier_mask]
            test, _, _, _ = self.timekernel.get_complex_coefficients(time_params)
            if test.size > 0:
                # There are complex terms - we only will have complex terms
                return (np.empty(0), np.empty(0))
            else:
                # The kernel only has real terms, so we have some real terms.
                theta0 = phase_kernel.theta0(fourier_params)
                p, r = self.timekernel.get_real_coefficients(time_params)
                a = p * theta0
                c = r
                return (a, c)

        def get_complex_coefficients(self, params):
            time_params = params[time_mask]
            fourier_params = params[fourier_mask]
            p, q, r, s = self.timekernel.get_complex_coefficients(time_params)
            if p.size == 0:
                # Only real Terms...
                p, r = self.timekernel.get_real_coefficients(time_params)
                n, theta_n = phase_kernel(fourier_params)
                a = 2 * np.outer(p, theta_n).flatten()
                b = np.zeros_like(a)
                c = np.outer(r, np.ones_like(theta_n)).flatten()
                d = np.outer(omega0 * n, np.ones_like(p)).flatten()
                return (a, b, c, d)
            else:
                # Complex terms...
                # TODO: Handle the case where p,q,r,s have more than one value.
                n, theta_n = phase_kernel(fourier_params)
                theta_0 = phase_kernel.theta0(fourier_params)

                a = np.concatenate((p * theta_n, p * theta_n, [p * theta_0]))
                b = np.concatenate((q * theta_n, q * theta_n, [q * theta_0]))
                c = r * np.ones(2 * len(n) + 1)
                d = np.concatenate((s - omega0 * n, s + omega0 * n, [s]))
                return (a, b, c, d)

    if optimise_tau_real_only:
        return HalfProfileTerm
    else:
        return FullProfileTerm