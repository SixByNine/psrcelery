import multiprocessing

import celerite
import numpy as np
import sklearn.decomposition
from matplotlib import pyplot as plt
import scipy.stats
from psrcelery import terms, celery_threads


def loadCelery(fname):
    datfile = np.load(fname)
    data = datfile['data']
    mjd = datfile['mjd']
    celery = Celery(data, mjd)
    for k in datfile.keys():
        celery.__dict__[k] = datfile[k]
    return celery


class Celery:
    def __init__(self, data, mjd):
        self.mjd_resampled = None
        self.eigenvalues_resample = None
        self.eigenprofiles_resample = None
        self.eigenprofiles = None
        self.eigenvalues = None
        self.eigenvalues_err = None
        self.eigenvalues_resample_err = None
        self.pred_mean_resample = None
        self.pred_std_resample = None
        self.pred_block_cov_resample = None
        self.pred_std = None
        self.pred_mean = None
        self.pred_block_cov = None
        self.nudot_val = self.nudot_err = self.nudot_mjd = None
        self.x = self.y = self.yerr = None
        self.subdata = None
        self.offmask = None
        self.onmask = None
        self.cel_gp = None
        self.data = data
        self.mjd = mjd
        nsub, nbin = self.data.shape
        self.phs = np.linspace(0, 1, nbin, endpoint=False)
        self.avgprof = np.median(self.data, axis=0)
        self.reset_onoff()

    def save(self, fname):
        data = {}
        for k in self.__dict__:
            if self.__dict__[k].__class__ is np.ndarray:
                data[k] = self.__dict__[k]
            if self.__dict__[k].__class__ is list:
                data[k] = self.__dict__[k]
            if self.__dict__[k].__class__ is float:
                data[k] = self.__dict__[k]
            if self.__dict__[k].__class__ is int:
                data[k] = self.__dict__[k]
        np.savez(fname, **data)

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

    def clean_data(self, iqr_cut):
        nsub, nbin = self.data.shape
        metric = []
        ii = np.arange(nsub)
        for i in ii:
            xcor = np.correlate(self.data[i, self.offmask], self.data[i, self.offmask], mode='full')
            l = len(xcor)
            metric.append(np.sum((xcor[l // 2 + 1:]) ** 2) / xcor[l // 2] ** 2)
        metric = np.array(metric)
        u = np.percentile(metric, 25)
        med = np.percentile(metric, 50)
        l = np.percentile(metric, 75)
        iqr = u - l
        zap = np.logical_or(metric > (med - iqr_cut * iqr), metric < (med + iqr_cut * iqr))
        good = np.logical_not(zap)
        self.data = self.data[good]
        self.mjd = self.mjd[good]

    def set_nudot(self, nudot_mjd, nudot_val, nudot_err=None):
        self.nudot_mjd = nudot_mjd
        self.nudot_val = nudot_val
        self.nudot_err = nudot_err

    def make_xydata(self):
        nsub, nbin = self.data.shape
        self.number_profiles = nsub
        template_subtracted_data = self.data - np.tile(self.avgprof, nsub).reshape((nsub, nbin))
        self.subdata = template_subtracted_data - np.mean(template_subtracted_data[:, self.onmask], axis=1).reshape(-1,
                                                                                                                    1)
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

    def use_standard_kernel(self, log_min_width=-1.5, min_length=10, max_length=4000, min_lnamp=-12, max_lnamp=-2):
        nc = int(10 ** (
            -log_min_width) / 2)  # This is determined empirically - I'm sure there is a better logical way to do it.
        print(f"nc={nc}")
        prior_bounds = {'log_amp': (min_lnamp, max_lnamp), 'log10_width': (log_min_width, -0.5),
                        'log10_length': (np.log10(min_length), np.log10(max_length))}
        celkern = terms.SimpleProfileTerm(log_amp=-8,
                                          log10_width=log_min_width,
                                          log10_length=3,
                                          ncoef=nc,
                                          bounds=prior_bounds)

        celkern += celerite.terms.JitterTerm(log_sigma=-6, bounds={'log_sigma': (-10, -1)})
        self.set_gp_model(celkern)

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
        self.parameter_vector = np.array(params)
        self.cel_gp.set_parameter_vector(self.parameter_vector)

    def predict_profiles_blockwise(self):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        self.pred_block_cov = []
        xstack = self.x.reshape((self.number_profiles, -1))
        pred_y = np.zeros_like(xstack)
        pred_yerr = np.zeros_like(xstack)
        for i, x in enumerate(xstack):
            print(f"{i}/{self.number_profiles}")
            pred_y[i], cov = self.cel_gp.predict(self.y, x, return_cov=True)
            pred_yerr[i] = np.sqrt(np.diag(cov))
            self.pred_block_cov.append(cov)
        self.pred_mean = pred_y
        self.pred_std = pred_yerr
        return self.pred_mean, self.pred_std

    def predict_profiles_resampled_blockwise(self, number_output_days=256):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        self.pred_block_cov_resample = []
        rmjd = np.round(self.mjd)
        rmjd -= rmjd[0]
        Nonbins = np.sum(self.onmask)
        self.number_output_days = number_output_days
        self.mjd_resampled = np.round(np.linspace(rmjd[0], rmjd[-1], number_output_days))
        self.x_resampled = np.tile(np.linspace(0, 1, Nonbins, endpoint=False), number_output_days)
        self.x_resampled += np.repeat(self.mjd_resampled, Nonbins)
        xstack = self.x_resampled.reshape((self.number_output_days, -1))
        pred_y = np.zeros_like(xstack)
        pred_yerr = np.zeros_like(xstack)
        for i, x in enumerate(xstack):
            print(f"{i}/{self.number_output_days}")
            pred_y[i], cov = self.cel_gp.predict(self.y, x, return_cov=True)
            pred_yerr[i] = np.sqrt(np.diag(cov))
            self.pred_block_cov_resample.append(cov)
        self.pred_mean_resample = pred_y
        self.pred_std_resample = pred_yerr
        self.kernel_values = np.reshape(self.cel_gp.kernel.get_value(self.x_resampled), (self.number_output_days, -1))
        return self.pred_mean_resample, self.pred_std_resample

    def predict_profiles_resampled_blockwise_THREADS(self, number_output_days=256, nthread=4):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        self.pred_block_cov_resample = []
        rmjd = np.round(self.mjd)
        rmjd -= rmjd[0]
        Nonbins = np.sum(self.onmask)
        self.number_output_days = number_output_days
        self.mjd_resampled = np.round(np.linspace(rmjd[0], rmjd[-1], number_output_days))
        self.x_resampled = np.tile(np.linspace(0, 1, Nonbins, endpoint=False), number_output_days)
        self.x_resampled += np.repeat(self.mjd_resampled, Nonbins)
        xstack = self.x_resampled.reshape((self.number_output_days, -1))
        pred_y = np.zeros_like(xstack)
        pred_yerr = np.zeros_like(xstack)

        with multiprocessing.Pool(nthread, initializer=celery_threads.init,
                                  initargs=(self.cel_gp, xstack, self.y)) as pool:
            ret = pool.map(celery_threads.run, range(len(xstack)))
            for i, x in enumerate(xstack):
                pred_y[i], pred_yerr[i], cov = ret[i]
                self.pred_block_cov_resample.append(cov)
        self.pred_mean_resample = pred_y
        self.pred_std_resample = pred_yerr
        self.kernel_values = np.reshape(self.cel_gp.kernel.get_value(self.x_resampled), (self.number_output_days, -1))
        return self.pred_mean_resample, self.pred_std_resample

    def predict_profiles_blockwise_THREADS(self, number_output_days=256, nthread=4):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        self.pred_block_cov = []
        xstack = self.x.reshape((self.number_profiles, -1))
        pred_y = np.zeros_like(xstack)
        pred_yerr = np.zeros_like(xstack)

        with multiprocessing.Pool(nthread, initializer=celery_threads.init,
                                  initargs=(self.cel_gp, xstack, self.y)) as pool:
            ret = pool.map(celery_threads.run, range(len(xstack)))
            for i, x in enumerate(xstack):
                pred_y[i], pred_yerr[i], cov = ret[i]
                self.pred_block_cov.append(cov)
        self.pred_mean = pred_y
        self.pred_std = pred_yerr
        return self.pred_mean, self.pred_std

    def predict_profiles(self, ignore_covariance=False):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        self.pred_block_cov = None
        if ignore_covariance:
            self.pred_mean, pred_var = self.cel_gp.predict(self.y, return_var=True)
            self.pred_std = np.sqrt(pred_var)
        else:
            self.pred_mean, pred_cov = self.cel_gp.predict(self.y, return_cov=True)
            self.pred_std = np.sqrt(np.diag(pred_cov))
            Nonbins = np.sum(self.onmask)
            self.pred_block_cov = [pred_cov[i * Nonbins:(i + 1) * Nonbins, i * Nonbins:(i + 1) * Nonbins] for i in
                                   range(pred_cov.shape[0] // Nonbins)]
        return self.pred_mean, self.pred_std

    def predict_resampled(self, number_output_days=256, ignore_covariance=False):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        self.pred_block_cov_resample = None
        rmjd = np.round(self.mjd)
        rmjd -= rmjd[0]
        Nonbins = np.sum(self.onmask)

        self.mjd_resampled = np.round(np.linspace(rmjd[0], rmjd[-1], number_output_days))
        self.x_resampled = np.tile(np.linspace(0, 1, Nonbins, endpoint=False), number_output_days)
        self.x_resampled += np.repeat(self.mjd_resampled, Nonbins)
        print(len(self.x_resampled))
        if ignore_covariance:
            self.pred_mean_resample, pred_var = self.cel_gp.predict(self.y, self.x_resampled, return_var=True)
            self.pred_std_resample = np.sqrt(pred_var)
        else:
            self.pred_mean_resample, pred_cov_resample = self.cel_gp.predict(self.y, self.x_resampled, return_cov=True)
            Nonbins = np.sum(self.onmask)
            self.pred_block_cov_resample = [
                pred_cov_resample[i * Nonbins:(i + 1) * Nonbins, i * Nonbins:(i + 1) * Nonbins] for i in
                range(pred_cov_resample.shape[0] // Nonbins)]
            self.pred_std_resample = np.sqrt(np.diag(pred_cov_resample))

        self.number_output_days = number_output_days

        self.kernel_values = np.reshape(self.cel_gp.kernel.get_value(self.x_resampled), (self.number_output_days, -1))
        return self.pred_mean_resample, self.pred_std_resample

    def make_resampled_block_matricies(self):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        xstack = self.x_resampled.reshape((self.number_output_days, -1))
        self.pred_block_cov_resample = []
        for i, x in enumerate(xstack):
            print(f"{i}/{self.number_output_days}")
            _, cov = self.cel_gp.predict(self.y, x, return_cov=True)
            self.pred_block_cov_resample.append(cov)

    def get_phase_factor(self):
        dp = self.phs[1] - self.phs[0]
        return dp / (self.x_resampled[1] - self.x_resampled[0])

    def compute_eigenprofiles(self):
        def run_pca(data):
            pca = sklearn.decomposition.PCA(n_components=10)
            eigenvalues = pca.fit_transform(data).T
            eigenprofiles = pca.components_
            return eigenvalues, eigenprofiles

        if not self.pred_mean_resample is None:
            self.eigenvalues_resample, self.eigenprofiles_resample = run_pca(
                np.reshape(self.pred_mean_resample, (self.number_output_days, -1)))
            if not self.pred_block_cov_resample is None:
                ## NOTE - is this really correct?
                self.eigenvalues_resample_err = []
                for c in self.eigenprofiles_resample:
                    self.eigenvalues_resample_err.append(np.sqrt(
                        c.T.dot(self.pred_block_cov_resample).dot(c)))

        if not self.pred_mean is None:
            self.eigenvalues, self.eigenprofiles = run_pca(np.reshape(self.pred_mean, (self.number_profiles, -1)))
            if not self.pred_block_cov is None:
                ## NOTE - is this really correct?
                self.eigenvalues_err = []
                for c in self.eigenprofiles:
                    self.eigenvalues_err.append(np.sqrt(
                        c.T.dot(self.pred_block_cov).dot(c)))

    def data_model_plot(self, outname=None):
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

    def rainbowplot(self, outname=None, show_pca=True, show_nudot=True, figsize=(12, 18), pca_comps=[0],
                    interpolation=None, scale_plots=False):
        if self.nudot_val is None:
            show_nudot = False
        if self.eigenvalues is None and self.eigenvalues_resample is None:
            show_pca = False
        threecolumn = show_nudot or show_pca  ## we use a three column layout

        nsub, nbin = self.data.shape

        model_profiles = np.reshape(self.pred_mean_resample, (self.number_output_days, -1))
        model_profile_std = np.reshape(self.pred_std_resample, (self.number_output_days, -1))
        model_profiles_at_data = np.reshape(self.pred_mean, (nsub, -1))

        plt.rcParams.update({'font.size': 12})

        sigma = np.abs(model_profiles / model_profile_std)

        vm = np.amax(np.abs(model_profiles))

        extent = (self.phs[self.onmask][0], self.phs[self.onmask][-1], self.mjd[0], self.mjd[-1])
        kvals = self.kernel_values
        _, outphs = kvals.shape
        kvals = np.roll(kvals, outphs // 2, 1)
        kvals = np.vstack((np.flip(kvals, axis=0), kvals))
        dp = self.phs[1] - self.phs[0]
        ds = np.reshape(self.x_resampled, (self.number_output_days, -1))[1, 0]
        height_ratio = 4

        alpha = np.ones_like(model_profiles)
        alpha[sigma < 3] = scipy.special.erf(sigma[sigma < 3] / 2)

        if threecolumn:
            width_ratio = 6
            fig, ((top_left_plot, profile_plot, kernel_plot), (left_plot, main_plot, err_plot)) = plt.subplots(2, 3,
                                                                                                               facecolor='white',
                                                                                                               gridspec_kw={
                                                                                                                   'width_ratios': [
                                                                                                                       1,
                                                                                                                       3,
                                                                                                                       3 / width_ratio],
                                                                                                                   'height_ratios': [
                                                                                                                       1,
                                                                                                                       height_ratio],
                                                                                                                   'wspace': 0.02},
                                                                                                               figsize=figsize)
            fig.delaxes(top_left_plot)  ## we don't use this one right now

        else:
            width_ratio = 3
            fig, ((profile_plot, kernel_plot), (main_plot, err_plot)) = plt.subplots(2, 2, facecolor='white',
                                                                                     gridspec_kw={'width_ratios': [3,
                                                                                                                   3 / width_ratio],
                                                                                                  'height_ratios': [1,
                                                                                                                    height_ratio],
                                                                                                  'wspace': 0.02},
                                                                                     figsize=figsize)

        ## MAIN RAINBOW PLOT
        main_plot.set_title("GP Model")
        main_plot.imshow(model_profiles, aspect='auto', origin='lower', interpolation=interpolation, cmap='rainbow',
                         alpha=alpha,
                         extent=extent, vmin=-vm, vmax=vm)
        main_plot.set_xlabel("Phase")

        if threecolumn:
            ## no y labels as they are on the left panel.
            main_plot.yaxis.set_tick_params(labelleft=False, labelright=False, right=True, left=True, direction='in')
        else:
            main_plot.set_ylabel("MJD")

        main_plot.set_ylim(extent[2], extent[3])
        main_plot.set_xlim(extent[0], extent[1])

        for mjd in self.mjd:
            main_plot.axhline(mjd, 0, 0.05, ls='-', color='gray', alpha=0.5)

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
                        interpolation=interpolation, origin='lower')

        err_plot.yaxis.set_label_position("right")
        err_plot.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')

        err_plot.set_ylabel("MJD")

        # Find phase with most variation
        most_variable_phase_bin = np.argmax(np.std(model_profiles, axis=0))

        lowprof = model_profiles_at_data[:, most_variable_phase_bin] < -0.01
        hiprof = model_profiles_at_data[:, most_variable_phase_bin] > 0.01

        if not show_pca:
            profile_plot.plot(self.phs[self.onmask], np.mean(self.data[lowprof][:, self.onmask], axis=0), color='b',
                              alpha=0.5, label='low')
            profile_plot.plot(self.phs[self.onmask], np.mean(self.data[hiprof][:, self.onmask], axis=0), color='r',
                              alpha=0.5, label='high')

        profile_plot.axhline(0, color='gray')
        profile_plot.plot(self.phs[self.onmask], self.avgprof[self.onmask], color='k', lw=2, label='total')
        profile_plot.axvline(self.phs[self.onmask][most_variable_phase_bin], color='gray', ls='--', alpha=0.4,
                             label='most variable phase')

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

        rx = (extent[1] - extent[0]) / width_ratio
        ry = (extent[3] - extent[2]) / height_ratio
        kernel_plot.set_xlim(-rx, rx)
        kernel_plot.set_ylim(-ry, ry)

        if threecolumn:
            left_plot.set_ylabel("MJD")
            left_plot.set_ylim(extent[2], extent[3])
            left_plot.yaxis.set_tick_params(labelleft=True, labelright=False, right=True, left=True, direction='in')
            if show_pca:
                for icomp in pca_comps:
                    if not self.eigenvalues_resample is None:
                        eigenvalues = self.eigenvalues_resample[icomp]
                        eigenprofile = self.eigenprofiles_resample[icomp]
                        eigenvalues_err = None if self.eigenvalues_resample_err is None else \
                            self.eigenvalues_resample_err[icomp]
                        e_mjd = self.mjd_resampled + self.mjd[0]
                    else:
                        eigenvalues = self.eigenvalues[icomp]
                        eigenprofile = self.eigenprofiles[icomp]
                        eigenvalues_err = None if self.eigenvalues_err is None else self.eigenvalues_err[icomp]
                        e_mjd = self.mjd
                    if show_nudot:
                        nudot_resamp = np.interp(self.mjd_resampled + self.mjd[0], self.nudot_mjd,
                                                 self.nudot_val)
                        r, _ = scipy.stats.pearsonr(eigenvalues, nudot_resamp)
                        if r < 0:
                            eigenprofile *= -1
                            eigenvalues *= -1
                        if icomp == pca_comps[0]:
                            nudot_eigen_convert = np.poly1d(np.polyfit(eigenvalues, nudot_resamp, 1))
                            eigen_nudot_convert = np.poly1d([1 / nudot_eigen_convert.coef[0],
                                                             -nudot_eigen_convert.coef[1] / nudot_eigen_convert.coef[
                                                                 0]])
                    left_plot.plot(eigenvalues, e_mjd, label=f'Eigenprofile({icomp})')
                    left_plot.set_xlabel(r"$\lambda_n$")
                    profile_plot.plot(self.phs[self.onmask], eigenprofile)
                    if not eigenvalues_err is None:
                        left_plot.fill_betweenx(e_mjd, eigenvalues - eigenvalues_err, eigenvalues + eigenvalues_err,
                                                alpha=0.5)
                    if icomp == pca_comps[0]:
                        proflo = self.avgprof[self.onmask] + eigenprofile * np.amin(eigenvalues)
                        profhi = self.avgprof[self.onmask] + eigenprofile * np.amax(eigenvalues)
                        profile_plot.plot(self.phs[self.onmask], proflo, color='b', alpha=0.5, label='low')
                        profile_plot.plot(self.phs[self.onmask], profhi, color='r', alpha=0.5, label='high')
            if show_nudot:
                ax_nudot = left_plot.twiny()
                ax_nudot.set_xlabel("nudot")
                ax_nudot.plot(self.nudot_val, self.nudot_mjd, color='k', ls='--')
                if not self.nudot_err is None:
                    ax_nudot.fill_betweenx(self.nudot_mjd, self.nudot_val - self.nudot_err,
                                           self.nudot_val + self.nudot_err,
                                           color='k', alpha=0.3)
                nudots = self.nudot_val[(self.nudot_mjd > extent[2]) & (self.nudot_mjd < extent[3])]
                nudot_max = np.amax(nudots)
                nudot_min = np.amin(nudots)
                if scale_plots and show_pca:
                    cmax = max(nudot_max, np.amax(nudot_eigen_convert(eigenvalues)))
                    cmin = min(nudot_min, np.amin(nudot_eigen_convert(eigenvalues)))

                    rng = cmax - cmin
                    cmax += 0.1 * rng
                    cmin -= 0.1 * rng
                    ax_nudot.set_xlim(cmin, cmax)
                    left_plot.set_xlim(eigen_nudot_convert(cmin), eigen_nudot_convert(cmax))

                else:
                    rng = nudot_max - nudot_min
                    ax_nudot.set_xlim(nudot_min - 0.1 * rng, nudot_max + 0.1 * rng)

        profile_plot.legend()
        if not (outname is None):
            plt.savefig(outname)
        else:
            plt.show()
