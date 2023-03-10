import numpy as np
import scipy.optimize as opt
import scipy.stats
import celerite
from matplotlib import pyplot as plt


def align_and_scale(data, template, nharm=None):
    if nharm == "auto":
        power_template = np.absolute(np.fft.rfft(template)) ** 2
        k = (np.arange(len(power_template)) + 1)
        ex = np.cumsum(power_template[::-1])
        A2 = ex / k
        prob = scipy.stats.chi2.cdf(power_template[::-1], df=2, loc=0, scale=A2)
        t = np.argmax(prob > 0.99)
        nharm = len(power_template) - t

    return np.apply_along_axis(align_and_scale_one, 1, data, template=template, nharm=nharm)


def align_and_scale_one(prof, template, nharm=None):
    # Equation A7 in Taylor 1992
    def get_dchi(tau, N, nbin):
        dphi = np.angle(xspec)[1:N]

        k = np.arange(1, N)

        dchi = np.sum(k * np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.sin(dphi + 2 * np.pi * k * tau / nbin))
        return dchi

    # Equation A9 in Taylor 1992
    def get_b(tau, N, nbin):
        dphi = np.angle(xspec)[1:N]
        k = np.arange(1, N)
        scale = np.sum(np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.cos(dphi + 2 * np.pi * k * tau / nbin))
        scale /= np.sum(np.abs(f_template[1:N]) ** 2)
        return scale

    # Equation A10 in Taylor 1992
    def get_sigma_tau(tau, N, nbin, b):
        dphi = np.angle(xspec)[1:N]
        k = np.arange(1, N)
        chi2 = np.sum(np.abs(f_prof[1:N]) ** 2 + b ** 2 * np.abs(f_template[1:N])) - 2 * b * np.sum(
            np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.cos(dphi + 2 * np.pi * k * tau / nbin))
        sigma2 = chi2 / (N - 1)
        de = np.sum(
            (k ** 2) * np.abs(f_prof[1:N]) * np.abs(f_template[1:N]) * np.cos(dphi + 2 * np.pi * k * tau / nbin))
        fac = nbin / (2 * np.pi)
        return np.sqrt(sigma2 / (2 * b * de)) * fac

    def rotate_phs(ff, phase_shift):
        fr = ff * np.exp(1.0j * 2 * np.pi * np.arange(len(ff)) * phase_shift)
        return np.fft.irfft(fr)

    nbin = len(prof)
    # We are going to do a cross correlation by means of the Fourier transform and the Wiener-Kinchin theorem
    f_template = np.fft.rfft(template)
    f_prof = np.fft.rfft(prof)

    min_spectral_bins = min(len(f_template),len(f_prof))

    # The cross correlation of a and b is the inverse transform of FT(a) times the conjugate of FT(b)
    xspec = f_template[:min_spectral_bins] * f_prof[:min_spectral_bins].conj()  # "cross spectrum"
    xcor = np.fft.irfft(xspec)  # Cross correlation

    ishift = np.argmax(np.abs(xcor))  # estimate of the shift directly from the peak cross-correlation

    # We need to define some bounds to search. (Actually this might not be optimal)
    lo = ishift - 1
    hi = ishift + 1
    if nharm is None or nharm > len(xspec):
        nh = len(xspec)
    else:
        nh = nharm
    # We minimise the chisquare parameter by findng the root of it's derivatiive following Taylor 1992
    # This root_scalar method uses the 'Brent 1973' algorithm for root finding.
    ret = opt.root_scalar(get_dchi, bracket=(lo, hi), x0=ishift, args=(nh, nbin), method='brentq')

    # tau is the bin shift between data and template, which will become our ToA
    tau = ret.root
    # Again folow the math of Taylor 1992 to get the scale factor, which it calls 'b'
    scale = get_b(tau, nh, nbin)
    # And finally given the shift and scale we can find the uncertainty on the shift.
    sigma_tau = get_sigma_tau(tau, nh, nbin, scale)

    # Phase shift is bin shift divided by nbins
    phase_shift = tau / nbin

    scaled_and_shifted = rotate_phs(f_prof, -phase_shift) / scale
    return scaled_and_shifted


class Celery:
    def __init__(self, data, mjd):
        self.offmask = None
        self.onmask = None
        self.cel_gp = None
        self.data = data
        self.mjd = mjd
        nsub, nbin = self.data.shape
        self.phs = np.linspace(0, 1, nbin, endpoint=False)

        self.avgprof = np.median(self.data, axis=0)
        self.reset_onoff()


    def reset_onoff(self):
        nsub, nbin = self.data.shape
        self.onmask = np.zeros(nbin, dtype=bool)
        self.offmask = np.ones(nbin, dtype=bool)

    def set_on_phase(self, start, end):
        phs = self.phs
        match = (phs > start) & (phs < end)
        # Add these phases to the onmask
        self.onmask = np.logical_or(self.onmask, match)
        # Remove the bins from the offmask!
        self.offmask = np.logical_and(self.offmask, np.logical_not(match))

    def remove_off_phase(self, start, end):
        """Remove these phases from the 'offpulse' phase"""
        phs = self.phs
        match = (phs > start) & (phs < end)
        # Remove the bins from the offmask!
        self.offmask = np.logical_and(self.offmask, np.logical_not(match))

    def make_xydata(self):
        nsub, nbin = self.data.shape
        self.subdata = self.data - np.tile(self.avgprof, nsub).reshape((nsub, nbin))
        offrms = np.std(self.subdata[:, self.offmask], axis=1)

        flatdata = self.subdata.flatten()

        Nonbins = np.sum(self.onmask)
        self.ymask = np.tile(self.onmask, nsub)

        # Round to the nearest day ... in future we should have this scaleable.
        rmjd = np.round(self.mjd)
        rmjd -= rmjd[0]
        if (np.any(np.diff(rmjd) < 1)):
            raise ValueError("Observations on the same day cause a problem (tofix)")
        self.x = np.tile(np.arange(Nonbins) / Nonbins, nsub) + np.repeat(rmjd, Nonbins)

        self.y = flatdata[self.ymask]
        self.yerr = np.repeat(offrms, nbin)[self.ymask]
        return self.x, self.y, self.yerr

    def set_gp_model(self, kernel):
        self.cel_gp = celerite.GP(kernel, mean=np.mean(self.y))
        self.cel_gp.compute(self.x, self.yerr)

    def log_likelihood(self, params):
        for i, ip in enumerate(self.cel_gp.get_parameter_bounds()):
            if params[i] < ip[0] or params[i] > ip[1]:
                return -np.inf
        self.cel_gp.set_parameter_vector(params)
        return self.cel_gp.log_likelihood(self.y)

    def sample_uniform(self, nsamples):
        low = np.array(self.cel_gp.get_parameter_bounds()).T[0]
        up = np.array(self.cel_gp.get_parameter_bounds()).T[1]

        p0 = np.random.uniform(0, 1, nsamples * len(self.cel_gp.get_parameter_names())) * np.tile((up - low),
                                                                                                  nsamples) + np.tile(
            low,
            nsamples)
        return p0.reshape(nsamples, -1)

    def set_parameter_vector(self, params):
        self.cel_gp.set_parameter_vector(params)

    def predict_profiles(self):
        self.pred_mean, pred_var = self.cel_gp.predict(self.y, self.x, return_var=True)
        self.pred_std = np.sqrt(pred_var)
        return self.pred_mean, self.pred_std

    def predict_resampled(self, number_output_days=256):
        rmjd = np.round(self.mjd)
        rmjd -= rmjd[0]
        Nonbins = np.sum(self.onmask)

        z = np.round(np.linspace(rmjd[0], rmjd[-1], number_output_days))
        self.x_resampled = np.tile(np.linspace(0, 1, Nonbins, endpoint=False), number_output_days)
        self.x_resampled += np.repeat(z, Nonbins)
        print(len(self.x_resampled))

        self.pred_mean_resample, pred_var = self.cel_gp.predict(self.y, self.x_resampled, return_var=True)
        self.pred_std_resample = np.sqrt(pred_var)
        self.number_output_days = number_output_days
        return self.pred_mean_resample, self.pred_std_resample

    def get_phase_factor(self):
        dp = self.phs[1] - self.phs[0]
        return dp / (self.x_resampled[1] - self.x_resampled[0])

    def data_model_plot(self,outname=None):
        nsub, nbin = self.data.shape
        model_profiles_at_data = np.reshape(self.pred_mean, (nsub, -1))

        plt.figure(figsize=(18, 18), facecolor='lightgray')
        plt.subplot(132)
        plt.title("GP Model")
        plt.imshow(model_profiles_at_data, aspect='auto', origin='lower', cmap='magma')
        plt.subplot(131)
        plt.title("Data")
        plt.imshow(self.subdata[:, self.onmask], aspect='auto', origin='lower', cmap='magma')
        plt.subplot(133)
        plt.title("Diff")
        plt.imshow(self.subdata[:, self.onmask] - model_profiles_at_data, aspect='auto', origin='lower', cmap='magma')

        if not (outname is None):
            plt.savefig(outname)
        else:
            plt.show()

    def rainbowplot(self, outname=None):
        nsub, nbin = self.data.shape
        model_profiles = np.reshape(self.pred_mean_resample, (self.number_output_days, -1))
        model_profile_std = np.reshape(self.pred_std_resample, (self.number_output_days, -1))
        model_profiles_at_data = np.reshape(self.pred_mean, (nsub, -1))

        plt.rcParams.update({'font.size': 12})

        sigma = np.abs(model_profiles / model_profile_std)

        vm = np.amax(np.abs(model_profiles))

        extent = (self.phs[self.onmask][0], self.phs[self.onmask][-1], self.mjd[0], self.mjd[-1])

        kvals = np.reshape(self.cel_gp.kernel.get_value(self.x_resampled), (self.number_output_days, -1))
        _, outphs = kvals.shape
        kvals = np.roll(kvals, outphs // 2, 1)
        kvals = np.vstack((np.flip(kvals, axis=0), kvals))
        dp = self.phs[1] - self.phs[0]
        ds = np.reshape(self.x_resampled, (self.number_output_days, -1))[1, 0]
        height_ratio = 4

        alpha = np.ones_like(model_profiles)
        alpha[sigma < 3] = scipy.special.erf(sigma[sigma < 3] / 2)

        fig, ((profile_plot, kernel_plot), (main_plot, err_plot)) = plt.subplots(2, 2, facecolor='white',
                                                                                 gridspec_kw={'width_ratios': [3, 1],
                                                                                              'height_ratios': [1,
                                                                                                                height_ratio],
                                                                                              'wspace': 0.02},
                                                                                 figsize=(12, 18))
        main_plot.set_title("GP Model")
        main_plot.imshow(model_profiles, aspect='auto', origin='lower', interpolation='nearest', cmap='rainbow',
                         alpha=alpha,
                         extent=extent, vmin=-vm, vmax=vm)
        main_plot.set_xlabel("Phase")
        main_plot.set_ylabel("MJD")

        main_plot.set_ylim(extent[2], extent[3])
        main_plot.set_xlim(extent[0], extent[1])

        err = np.mean(model_profile_std, axis=1)
        err_plot.plot(err, np.linspace(self.mjd[0], self.mjd[1], self.number_output_days), color='k', alpha=0.5)
        err_plot.plot(-err, np.linspace(self.mjd[0], self.mjd[1], self.number_output_days), color='k', alpha=0.5)
        err_plot.plot(2 * err, np.linspace(self.mjd[0], self.mjd[1], self.number_output_days), color='k', ls=':',
                      alpha=0.5)
        err_plot.plot(-2 * err, np.linspace(self.mjd[0], self.mjd[1], self.number_output_days), color='k', ls=':',
                      alpha=0.5)
        err_plot.set_ylim(self.mjd[0], self.mjd[1])
        err_plot.set_xlabel("Signal (peak flux)")

        a1data = np.tile(np.linspace(-vm, vm, 256), self.number_output_days).reshape(self.number_output_days, -1)

        a1sigma = np.abs(a1data.T / err).T
        a1alpha = np.ones_like(a1data)
        a1alpha[a1sigma < 3] = scipy.special.erf(a1sigma[a1sigma < 3] / 2)

        err_plot.set_title("Uncertainty & Colour scale")

        err_plot.imshow(a1data, cmap='rainbow', aspect='auto', extent=(-vm, vm, self.mjd[0], self.mjd[1]),
                        alpha=a1alpha,
                        interpolation='nearest', origin='lower')

        err_plot.yaxis.set_label_position("right")
        err_plot.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')

        err_plot.set_ylabel("MJD")

        # Find phase with most variation
        most_variable_phase_bin = np.argmax(np.std(model_profiles, axis=0))

        lowprof = model_profiles_at_data[:, most_variable_phase_bin] < -0.01
        hiprof = model_profiles_at_data[:, most_variable_phase_bin] > 0.01

        profile_plot.plot(self.phs[self.onmask], np.mean(self.data[lowprof][:, self.onmask], axis=0), color='b',
                          alpha=0.5, label='low')
        profile_plot.plot(self.phs[self.onmask], np.mean(self.data[hiprof][:, self.onmask], axis=0), color='r',
                          alpha=0.5, label='high')
        profile_plot.plot(self.phs[self.onmask], self.avgprof[self.onmask], color='k', lw=2, label='total')
        profile_plot.axvline(self.phs[self.onmask][most_variable_phase_bin] , color='gray', ls='--', alpha=0.4,
                             label='most variable phase')
        profile_plot.legend()

        profile_plot.set_xlim(extent[0], extent[1])
        profile_plot.set_xlabel("Phase")
        profile_plot.set_ylabel("Amplitude")
        profile_plot.set_title("Average Profile")

        kernel_plot.set_title("Kernel")
        kernel_plot.imshow(kvals, aspect='auto', origin='lower',
                           extent=(-outphs * dp / 2, outphs * dp / 2, -self.number_output_days * ds,
                                   self.number_output_days * ds), cmap='Greys')
        kernel_plot.set_xlabel("Phase Lag")
        kernel_plot.set_ylabel("Time Lag (days)")

        kernel_plot.yaxis.set_label_position("right")
        kernel_plot.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')

        rx = (extent[1] - extent[0]) / 3
        ry = (extent[3] - extent[2]) / height_ratio
        kernel_plot.set_xlim(-rx, rx)
        kernel_plot.set_ylim(-ry, ry)

        if not (outname is None):
            plt.savefig(outname)
        else:
            plt.show()
