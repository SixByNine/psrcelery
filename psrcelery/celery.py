import celerite
import numpy as np
import sklearn.decomposition
from matplotlib import pyplot as plt
import scipy.stats

def loadCelery(fname):
    datfile = np.load(fname)
    data = datfile['data']
    mjd = datfile['mjd']
    celery = Celery(data,mjd)
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
        self.nudot = self.nudot_err = self.nudot_mjd = None
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

    def save(self,fname):
        data = {}
        for k in self.__dict__:
            if self.__dict__[k].__class__ is np.ndarray:
                data[k] = self.__dict__[k]
            if self.__dict__[k].__class__ is list:
                data[k] = self.__dict__[k]
        np.savez(fname,**data)



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
        self.number_profiles = nsub
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

    def predict_profiles(self, ignore_covariance=False):
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
        return self.pred_mean_resample, self.pred_std_resample

    def get_phase_factor(self):
        dp = self.phs[1] - self.phs[0]
        return dp / (self.x_resampled[1] - self.x_resampled[0])

    def compute_eigenprofiles(self):
        def run_pca(data):
            pca = sklearn.decomposition.PCA(n_components=10)
            eigenvalues = pca.fit_transform(data)
            eigenprofiles = pca.components_
            return eigenprofiles, eigenvalues

        if not self.pred_mean_resample is None:
            self.eigenvalues_resample, self.eigenprofiles_resample = run_pca(np.reshape(self.pred_mean_resample,(self.number_output_days,-1)))
            if not self.pred_block_cov_resample is None:
                ## NOTE - is this really correct?
                self.eigenvalues_resample_err = np.sqrt(
                    self.eigenprofiles_resample.T.dot(self.pred_block_cov_resample).dot(self.eigenprofiles_resample))

        if not self.pred_mean is None:
            self.eigenvalues, self.eigenprofiles = run_pca(self.reshape(self.pred_mean,(self.number_profiles,-1)))
            if not self.pred_block_cov is None:
                ## NOTE - is this really correct?
                self.eigenvalues_err = np.sqrt(
                    self.eigenprofiles.T.dot(self.pred_block_cov).dot(self.eigenprofiles))

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

    def rainbowplot(self, outname=None, show_pca = True, show_nudot = True, figsize=(12,18),pca_comps=[0]):
        if self.nudot is None:
            show_nudot=False
        if self.eigenvalues is None and self.eigenvalues_resample is None:
            show_pca = False
        threecolumn = show_nudot or show_pca ## we use a three column layout

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

        if threecolumn:
            width_ratio=6
            fig, ((top_left_plot, profile_plot, kernel_plot), (left_plot,main_plot, err_plot)) = plt.subplots(2, 3, facecolor='white',
                                                                                     gridspec_kw={
                                                                                         'width_ratios': [1,3,3/width_ratio],
                                                                                         'height_ratios': [1,
                                                                                                           height_ratio],
                                                                                         'wspace': 0.02},
                                                                                     figsize=figsize)
            fig.delaxes(top_left_plot) ## we don't use this one right now

        else:
            width_ratio=3
            fig, ((profile_plot, kernel_plot), (main_plot, err_plot)) = plt.subplots(2, 2, facecolor='white',
                                                                                 gridspec_kw={'width_ratios': [3, 3/width_ratio],
                                                                                              'height_ratios': [1,
                                                                                                                height_ratio],
                                                                                              'wspace': 0.02},
                                                                                 figsize=figsize)

        ## MAIN RAINBOW PLOT
        main_plot.set_title("GP Model")
        main_plot.imshow(model_profiles, aspect='auto', origin='lower', interpolation='nearest', cmap='rainbow',
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
        profile_plot.axvline(self.phs[self.onmask][most_variable_phase_bin], color='gray', ls='--', alpha=0.4,
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

        rx = (extent[1] - extent[0]) / width_ratio
        ry = (extent[3] - extent[2]) / height_ratio
        kernel_plot.set_xlim(-rx, rx)
        kernel_plot.set_ylim(-ry, ry)

        if threecolumn:
            left_plot.set_ylabel("MJD")
            left_plot.set_ylim(extent[2], extent[3])
            left_plot.yaxis.set_tick_params(labelleft=True, labelright=False, right=True, left=True, direction='in')
            if show_nudot:
                ax_nudot = left_plot.twiny()
                ax_nudot.set_xlabel("nudot")
                ax_nudot.plot(self.nudot,self.nudot_mjd, color='k', ls='--')
                if not self.nudot_err is None:
                    ax_nudot.fill_betweenx(self.nudot_mjd,self.nudot-self.nudot_err,self.nudot+self.nudot_err,color='k',alpha=0.3)
            if show_pca:
                for icomp in pca_comps:
                    if not self.eigenvalues_resample is None:
                        eigenvalues = self.eigenvalues_resample[icomp]
                        eigenvalues_err = self.eigenvalues_resample_err[icomp]
                        e_mjd = self.mjd_resampled
                    else:
                        eigenvalues = self.eigenvalues[icomp]
                        eigenvalues_err = self.eigenvalues_err[icomp]
                        e_mjd = self.mjd
                    plt.plot(eigenvalues,e_mjd)
                    if not eigenvalues_err is None:
                        plt.fill_betweenx(e_mjd,eigenvalues-eigenvalues_err,eigenvalues+eigenvalues_err,alpha=0.5)

        if not (outname is None):
            plt.savefig(outname)
        else:
            plt.show()
