import multiprocessing

import celerite
import numpy as np
import sklearn.decomposition
from matplotlib import pyplot as plt
import scipy.stats
from psrcelery import terms, phase_kernels, celery_threads
import pickle
import dill
from matplotlib.ticker import FormatStrFormatter


def loadCelery(fname):
    datfile = np.load(fname)
    data = datfile['data']
    mjd = datfile['mjd']
    celery = Celery(data, mjd)
    for k in datfile.keys():
        if k == "cel_gp_bytes":
            celery.cel_gp = pickle.loads(datfile[k].tobytes())
        else:
            celery.__dict__[k] = datfile[k]

    # Fixes for reading some older versions...
    if celery.onmask is not None and celery.number_onpulse_bins is None:
        celery.number_onpulse_bins = np.sum(celery.onmask)
    if 'pred_block_cov_resample' in celery.__dict__:
        print("Converting pre1.0 format psrcelery file")
        celery.pred_phase_cov_function_resample = np.copy(celery.pred_block_cov_resample[0][0])
        celery.pred_phase_cov_function_resample /= celery.pred_phase_cov_function_resample[0]
        celery.pred_time_variance_resample = np.copy(celery.pred_block_cov_resample[:,0,0])
        del celery.pred_block_cov_resample
    if 'pred_block_cov' in celery.__dict__:
        celery.pred_phase_cov_function = np.copy(celery.pred_block_cov[0][0])
        celery.pred_phase_cov_function /= celery.pred_phase_cov_function[0]
        celery.pred_time_variance = np.copy(celery.pred_block_cov[:,0,0])
        del celery.pred_block_cov  

    return celery


class Celery:
    def __init__(self, data, mjd):
        self.celery_version=1.0
        self.eigenvalues_data = None
        self.eigenvalue_data_err = None
        self.kernel_values = None
        self.x_resampled = None
        self.number_output_days = None
        self.parameter_vector = None
        self.ymask = None
        self.number_profiles = None
        self.number_onpulse_bins = None
        self.mjd_resampled = None
        self.eigenvalues_resample = None
        self.eigenprofiles_resample = None
        self.eigenprofiles = None
        self.eigenvalues = None
        self.eigenvalues_err = None
        self.eigenvalues_resample_err = None

        self.pred_mean_resample = None
        self.pred_time_variance_resample = None
        self.pred_phase_cov_function_resample = None

        self.pred_mean = None
        self.pred_time_variance = None
        self.pred_phase_cov_function = None

        self.nudot_val = self.nudot_err = self.nudot_mjd = None
        self.x = self.y = self.yerr = None
        self.subdata = None
        self.offmask = None
        self.onmask = None
        self.cel_gp = None
        self.cel_gp_pack = None
        self.data = data
        self.mjd = mjd
        self.rounded_mjd_factor = 1
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

        # if self.cel_gp is not None:
        #    data['cel_gp_bytes'] = np.frombuffer(pickle.dumps(self.cel_gp), dtype=np.uint8)
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

    def set_nudot(self, nudot_mjd, nudot_val, nudot_err=None):
        self.nudot_mjd = nudot_mjd
        self.nudot_val = nudot_val
        self.nudot_err = nudot_err

    def make_xydata(self, rounded_mjd_factor=1):
        nsub, nbin = self.data.shape
        self.rounded_mjd_factor = rounded_mjd_factor
        self.number_profiles = nsub
        template_subtracted_data = self.data - np.tile(self.avgprof, nsub).reshape((nsub, nbin))
        self.subdata = template_subtracted_data - np.mean(template_subtracted_data[:, self.onmask], axis=1).reshape(-1, 1)
        offrms = np.std(self.subdata[:, self.offmask], axis=1)

        # Remove the weighted mean profile, since this is what the GP kind of expects to be zero.
        weights = (1 / offrms ** 2).reshape(-1,1)
        weighted_mean_profile = np.sum(self.subdata*weights, axis=0)/np.sum(weights)
        print(weights.shape, self.subdata.shape, weighted_mean_profile.shape)
        self.subdata -= weighted_mean_profile

        flatdata = self.subdata.flatten()

        self.number_onpulse_bins = np.sum(self.onmask)
        self.ymask = np.tile(self.onmask, nsub)

        # Round to the nearest day ... in future we should have this scaleable.
        rmjd = np.round(self.mjd * self.rounded_mjd_factor)
        rmjd -= rmjd[0]
        if np.any(np.diff(rmjd) < 1):
            raise ValueError("Observations on the same day cause a problem (tofix)")
        self.x = np.tile(np.arange(self.number_onpulse_bins) / self.number_onpulse_bins, nsub) \
                 + np.repeat(rmjd, self.number_onpulse_bins)

        self.y = flatdata[self.ymask]
        self.yerr = np.repeat(offrms, nbin)[self.ymask]
        return self.x, self.y, self.yerr

    def use_standard_kernel(self, log_min_width=-1.5, log_max_width=-0.5, 
                            min_length=10, max_length=4000, min_lnamp=-12, max_lnamp=-2,
                            matern=False):
        nc = int(10 ** (
            -log_min_width) / 2)  # This is determined empirically - I'm sure there is a better logical way to do it.
        print(f"nc={nc}")

        if matern:
            CustomTerm = terms.make_custom_profile_term(phase_kernels.GaussianPhaseKernel(nc),
                                                        celerite.terms.Matern32Term)
            celkern = CustomTerm(log_sigma=-4, log10_width=log_min_width, log_rho=np.log(0.5*(max_length+min_length)),
                                 bounds={'log_sigma': (min_lnamp / 2, max_lnamp / 2),
                                         'log10_width': (log_min_width, log_max_width),
                                         'log_rho': (np.log(min_length), np.log(max_length))},eps=1e-5)

        else:
            prior_bounds = {'log_amp': (min_lnamp, max_lnamp), 'log10_width': (log_min_width, log_max_width),
                            'log10_length': (np.log10(min_length), np.log10(max_length))}
            celkern = terms.SimpleProfileTerm(log_amp=-8,
                                              log10_width=log_min_width,
                                              log10_length=np.log10(0.5*(max_length+min_length)),
                                              ncoef=nc,
                                              bounds=prior_bounds)

        celkern += celerite.terms.JitterTerm(log_sigma=-6, bounds={'log_sigma': (-12, -1)})
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

    def _predict_blocks(self, xstack, nthread=1):
        nprof, nbin = xstack.shape
        pred_y = np.zeros_like(xstack)

        if nthread > 1:
            with multiprocessing.Pool(nthread, initializer=celery_threads.init,
                                      initargs=(dill.dumps(self.cel_gp), xstack, self.y)) as pool:
                ret = pool.map(celery_threads.run, range(len(xstack)))
                for i, x in enumerate(xstack):
                    pred_y[i] = ret[i]
        else:
            for i, x in enumerate(xstack):
                print(f"{i}/{self.number_profiles}")
                pred_y[i] = self.cel_gp.predict(self.y, x, return_cov=False)

        # Compute the error in each profile from the error in the central bin
        print("Computing variances etc")
        _, pred_1d_var = self.cel_gp.predict(self.y, xstack[:,nbin//2], return_cov=False,return_var=True)
        pred_cov_function= self.cel_gp.kernel.get_value(xstack[0])
        pred_cov_function/=pred_cov_function[0]
        return pred_y, pred_1d_var, pred_cov_function

    def predict_profiles_blockwise(self, nthread=1):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        xstack = self.x.reshape((self.number_profiles, -1))
        self.pred_mean, self.pred_time_variance, self.pred_phase_cov_function = self._predict_blocks(xstack, nthread)
        return self.pred_mean

    def predict_profiles_resampled_blockwise(self, number_output_days=256, nthread=1):
        self.cel_gp.set_parameter_vector(self.parameter_vector)
        rmjd = np.round(self.mjd * self.rounded_mjd_factor)
        rmjd -= rmjd[0]
        self.number_output_days = number_output_days
        self.rmjd_resampled = np.round(np.linspace(rmjd[0], rmjd[-1], number_output_days))
        self.mjd_resampled = self.rmjd_resampled / self.rounded_mjd_factor
        self.x_resampled = np.tile(np.linspace(0, 1, self.number_onpulse_bins, endpoint=False), number_output_days)
        self.x_resampled += np.repeat(self.rmjd_resampled, self.number_onpulse_bins)
        xstack = self.x_resampled.reshape((self.number_output_days, -1))

        self.pred_mean_resample, self.pred_time_variance_resample, self.pred_phase_cov_function_resample = self._predict_blocks(xstack,
                                                                                                             nthread)

        self.kernel_values = np.reshape(self.cel_gp.kernel.get_value(self.x_resampled), (self.number_output_days, -1))
        return self.pred_mean_resample




    def get_phase_factor(self):
        dp = self.phs[1] - self.phs[0]
        return dp / (self.x_resampled[1] - self.x_resampled[0])

    def compute_eigenprofiles(self):
        def run_pca(data):
            pca = sklearn.decomposition.PCA(n_components=10)
            eigenvalues = pca.fit_transform(data).T
            eigenprofiles = pca.components_
            self.pca_explained_variance = pca.explained_variance_ratio_
            return eigenvalues, eigenprofiles
        
        def get_eigenvalue_errors(phase_cov_func, time_variance,eigenprofs):
            # More time and memory efficient to do it day by day than to construct
            # a giant covariance matrix
            nprof=len(time_variance)
            nbin=len(phase_cov_func)
            covariance_matrix = np.zeros((nbin, nbin))
            for i in range(nbin):
                covariance_matrix[i] = np.roll(phase_cov_func, i)                
            
            eigenvalues_err = np.zeros((len(eigenprofs),nprof))
            
            for iday in range(nprof):
                C = covariance_matrix * time_variance[iday]
                for i, ep in enumerate(eigenprofs):
                    eigenvalues_err[i,iday] = np.sqrt(ep.T.dot(C).dot(ep))
            return eigenvalues_err
        
                    
        if self.pred_mean_resample is not None:
            # Do PCA on the resampled profiles
            self.pred_mean_resample[np.isnan(self.pred_mean_resample)] = 0
            self.pred_mean_resample[np.isinf(self.pred_mean_resample)] = 0
            self.eigenvalues_resample, self.eigenprofiles_resample = run_pca(
                np.reshape(self.pred_mean_resample, (self.number_output_days, -1)))

            self.eigenvalues_data = np.dot(self.eigenprofiles_resample, self.subdata[:, self.onmask].T)
            dataV = np.var(self.subdata[:, self.offmask], axis=1)
            self.eigenvalue_data_err = np.zeros_like(self.eigenvalues_data)
            for i, ep in enumerate(self.eigenprofiles_resample):
                self.eigenvalue_data_err[i] = (np.sqrt(dataV * np.dot(ep, ep)))
                
            if self.pred_phase_cov_function_resample is not None:
                self.eigenvalues_resample_err = get_eigenvalue_errors(self.pred_phase_cov_function_resample,
                                                                    self.pred_time_variance_resample,
                                                                    self.eigenprofiles_resample)   
        if self.pred_mean is not None:
            # Do PCA on the reconstructed profiles
            self.pred_mean[np.isnan(self.pred_mean)] = 0
            self.pred_mean[np.isinf(self.pred_mean)] = 0
            self.eigenvalues, self.eigenprofiles = run_pca(np.reshape(self.pred_mean, (self.number_profiles, -1)))
            if self.pred_phase_cov_function is not None:
                self.eigenvalues_err = get_eigenvalue_errors(self.pred_phase_cov_function,
                                                                    self.pred_time_variance,
                                                                    self.eigenprofiles)


    def dampen_edges(self, threshold=0.05):
        edge_mask = self.avgprof[self.onmask] < threshold
        self.pred_mean -= np.mean(self.pred_mean[:, edge_mask], axis=1).reshape((-1, 1))
        self.pred_mean_resample -= np.mean(self.pred_mean_resample[:, edge_mask], axis=1).reshape((-1, 1))

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







    def rainbowplot(self, outname=None, show_pca=True, show_nudot=True, figsize=(7, 14), pca_comps=(0,),
                    interpolation=None, scale_plots=False, eigenvalue_colors=None, title="Profile", cmap='rainbow',
                    glitches=[],nudot_bounds=None):
        if self.nudot_val is None:
            show_nudot = False
        if self.eigenvalues is None and self.eigenvalues_resample is None:
            show_pca = False
        if eigenvalue_colors is None:
            eigenvalue_colors = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                                 (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                                 (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                                 (0.8352941176470589, 0.3686274509803922, 0.0),
                                 (0.8, 0.47058823529411764, 0.7372549019607844),
                                 (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
                                 (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
                                 (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
                                 (0.9254901960784314, 0.8823529411764706, 0.2),
                                 (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)]
        threecolumn = show_nudot or show_pca  # we use a three column layout

        def translate_phase(phase):
            return 360 * (phase - 0.5)

        nsub, nbin = self.data.shape

        model_profiles = np.reshape(self.pred_mean_resample, (self.number_output_days, -1))
        
        #model_profile_std = np.reshape(self.pred_std_resample, (self.number_output_days, -1))
        model_profile_std = np.repeat(np.sqrt(self.pred_time_variance_resample), self.number_onpulse_bins).reshape(self.number_output_days, -1)
        model_profiles_at_data = np.reshape(self.pred_mean, (nsub, -1))

        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'font.family': 'serif'})
        sigma = np.abs(model_profiles / model_profile_std)

        vm = np.amax(np.abs(model_profiles))

        extent = (
            translate_phase(self.phs[self.onmask][0]), translate_phase(self.phs[self.onmask][-1]), self.mjd[0],
            self.mjd[-1])
        kvals = self.kernel_values
        _, outphs = kvals.shape
        kvals = np.roll(kvals, outphs // 2, 1)
        kvals = np.vstack((np.flip(kvals, axis=0), kvals))
        dp = translate_phase(self.phs[1]) - translate_phase(self.phs[0])
        ds = np.reshape(self.x_resampled, (self.number_output_days, -1))[1, 0]
        height_ratio = 4

        alpha = np.ones_like(model_profiles)
        alpha[sigma < 3] = scipy.special.erf(sigma[sigma < 3] / 2)

        if threecolumn:
            width_ratio = 4
            fig, ((top_left_plot, profile_plot, kernel_plot),
                  (left_plot, main_plot, err_plot)) = plt.subplots(2, 3,
                                                                   facecolor='white',
                                                                   gridspec_kw={
                                                                       'width_ratios': [1, 3, 3 / width_ratio],
                                                                       'height_ratios': [1, height_ratio],
                                                                       'wspace': 0.02},
                                                                   figsize=figsize)
            fig.delaxes(top_left_plot)  # we don't use this one right now

        else:
            width_ratio = 3
            fig, ((profile_plot, kernel_plot),
                  (main_plot, err_plot)) = plt.subplots(2, 2, facecolor='white',
                                                        gridspec_kw={'width_ratios': [3, 3 / width_ratio],
                                                                     'height_ratios': [1, height_ratio],
                                                                     'wspace': 0.02},
                                                        figsize=figsize)


        # MAIN RAINBOW PLOT
        # main_plot.set_title("GP Model")
        main_plot.imshow(model_profiles, aspect='auto', origin='lower', interpolation=interpolation, cmap=cmap,
                         alpha=alpha, extent=extent, vmin=-vm, vmax=vm)
        main_plot.set_xlabel("Phase (deg)")

        if threecolumn:
            # no y labels as they are on the left panel.
            main_plot.yaxis.set_tick_params(labelleft=False, labelright=False, right=True, left=True, direction='in')
        else:
            main_plot.set_ylabel("MJD")

        main_plot.set_ylim(extent[2], extent[3])
        main_plot.set_xlim(extent[0], extent[1])

        for mjd in self.mjd:
            main_plot.axhline(mjd, 0, 0.02, ls='-', color='gray', alpha=0.5)
            if threecolumn:
                left_plot.axhline(mjd, 0.95, 1.0, ls='-', color='gray', alpha=0.5)
        err = np.mean(model_profile_std, axis=1)
        err_plot.plot(err, self.mjd_resampled + self.mjd[0], color='k', alpha=0.5)
        err_plot.plot(-err, self.mjd_resampled + self.mjd[0], color='k', alpha=0.5)
        err_plot.plot(2 * err, self.mjd_resampled + self.mjd[0], color='k', ls=':',
                      alpha=0.5)
        err_plot.plot(-2 * err, self.mjd_resampled + self.mjd[0], color='k', ls=':',
                      alpha=0.5)
        err_plot.set_ylim(extent[2], extent[3])
        err_plot.set_xlabel("Signal (peak flux)")

        a1data = np.tile(np.linspace(-vm, vm, 256), self.number_output_days).reshape(self.number_output_days, -1)

        a1sigma = np.abs(a1data.T / err).T
        a1alpha = np.ones_like(a1data)
        a1alpha[a1sigma < 3] = scipy.special.erf(a1sigma[a1sigma < 3] / 2)

        # err_plot.set_title("Uncertainty & Colour scale")

        err_plot.imshow(a1data, cmap=cmap, aspect='auto', extent=(-vm, vm, extent[2], extent[3]),
                        alpha=a1alpha,
                        interpolation=interpolation, origin='lower')

        err_plot.yaxis.set_label_position("right")
        err_plot.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
        err_plot.tick_params(axis='y', labelrotation=-50, direction='inout')

        err_plot.set_ylabel("MJD")
        err_plot.xaxis.set_major_formatter(FormatStrFormatter('%g'))

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
        profile_plot.plot(translate_phase(self.phs[self.onmask]), self.avgprof[self.onmask], color='k', lw=2,
                          label='Median\nProfile')
        profile_plot.axvline(translate_phase(self.phs[self.onmask][most_variable_phase_bin]), color='gray', ls='--',
                             alpha=0.4,
                             label='MVP')

        profile_plot.set_xlim(extent[0], extent[1])
        profile_plot.set_xlabel("Phase (deg)")
        # profile_plot.set_ylabel("Amplitude")
        profile_plot.set_title(title)

        kernel_plot.set_title("Kernel")
        kernel_plot.imshow(kvals, aspect='auto', origin='lower',
                           extent=(-outphs * dp / 2, outphs * dp / 2, -self.number_output_days * ds,
                                   self.number_output_days * ds), cmap='Greys')
        kernel_plot.set_xlabel("Phase Lag")
        kernel_plot.set_ylabel("Time Lag (days)")

        kernel_plot.yaxis.set_label_position("right")
        kernel_plot.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')

        rx = (extent[1] - extent[0]) / width_ratio / 2
        ry = (extent[3] - extent[2]) / height_ratio / 2

        #     kernel_plot.set_yticks([])
        kernel_plot.set_xlim(-rx, rx)
        kernel_plot.set_ylim(-ry, ry)
        kernel_plot.xaxis.set_major_formatter(FormatStrFormatter('%g'))

        if threecolumn:
            left_plot.set_ylabel("MJD")
            left_plot.tick_params(axis='y', labelrotation=50, direction='inout')
            left_plot.set_ylim(extent[2], extent[3])
            left_plot.yaxis.set_tick_params(labelleft=True, labelright=False, right=True, left=True, direction='in')
            left_plot.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            for glitch in glitches:
                left_plot.axhline(glitch,color='salmon',ls='--')
            if show_pca:
                for icomp in pca_comps:
                    if self.eigenvalues_resample is not None:
                        eigenvalues = self.eigenvalues_resample[icomp].copy()
                        eigenprofile = self.eigenprofiles_resample[icomp].copy()
                        eigenvalues_err = None if self.eigenvalues_resample_err is None else \
                            self.eigenvalues_resample_err[icomp]
                        e_mjd = self.mjd_resampled + self.mjd[0]
                    else:
                        eigenvalues = self.eigenvalues[icomp].copy()
                        eigenprofile = self.eigenprofiles[icomp].copy()
                        eigenvalues_err = None if self.eigenvalues_err is None else self.eigenvalues_err[icomp]
                        e_mjd = self.mjd
                    elab = ""
                    if self.eigenprofiles is not None and np.sum(eigenprofile * self.eigenprofiles[icomp]) < 0:
                        # We have accidently inverted one set of eigenprofiles
                        # This was a bug in early codes.
                        print("Note eigenprofiles have been sign corrected for consistency.")
                        eigenprofile *=-1
                        eigenvalues *= -1
                    if show_nudot:
                        nudot_resamp2 = np.interp(self.mjd, self.nudot_mjd,
                                                  self.nudot_val)
                        r2, _ = scipy.stats.pearsonr(self.eigenvalues[icomp], nudot_resamp2)
                        #r2, _ = scipy.stats.spearmanr(self.eigenvalues[icomp], nudot_resamp2)
                        nudot_resamp = np.interp(self.mjd_resampled + self.mjd[0], self.nudot_mjd,
                                                 self.nudot_val)
                        r, _ = scipy.stats.pearsonr(eigenvalues, nudot_resamp)
                        #r, _ = scipy.stats.spearmanr(eigenvalues, nudot_resamp)
                        if r2 < 0:
                            eigenprofile *= -1
                            eigenvalues *= -1
                            r2*=-1
                            r*=-1
                        if icomp == pca_comps[0]:
                            nudot_eigen_convert = np.poly1d(np.polyfit(eigenvalues, nudot_resamp, 1))
                            eigen_nudot_convert = np.poly1d([1 / nudot_eigen_convert.coef[0],
                                                             -nudot_eigen_convert.coef[1] / nudot_eigen_convert.coef[
                                                                 0]])
                        #elab = f" [r={r:.2f},{r2:.2f}]"
                        elab = f" [r={r2:.2f}]"
                        elab = ""
                    left_plot.plot(eigenvalues, e_mjd, label=f'$\\lambda_$({icomp})',
                                   color=eigenvalue_colors[icomp % len(eigenvalue_colors)])
                    left_plot.set_xlabel(r"$\lambda_n$")
                    profile_plot.plot(translate_phase(self.phs[self.onmask]), eigenprofile,
                                      color=eigenvalue_colors[icomp % len(eigenvalue_colors)],
                                      label=f"$\\mathbf{{e}}_{icomp}$" + elab)
                    # left_plot.errorbar(sign*self.eigenvalues[icomp], self.mjd, xerr=self.eigenvalues_err[icomp],
                    #                   ls='None', marker='x', color=eigenvalue_colors[icomp % len(eigenvalue_colors)])
                    #                 profile_plot.plot(self.phs[self.onmask], eigenprofile,
                    #                                   color=eigenvalue_colors[icomp % len(eigenvalue_colors)],
                    #                                   label=f"$\\mathbf{{e}}_{icomp}$")
                    if eigenvalues_err is not None:
                        left_plot.fill_betweenx(e_mjd, eigenvalues - eigenvalues_err, eigenvalues + eigenvalues_err,
                                                alpha=0.5, color=eigenvalue_colors[icomp % len(eigenvalue_colors)])
                    if icomp == pca_comps[0]:
                        proflo = self.avgprof[self.onmask] + eigenprofile * np.amin(eigenvalues)
                        profhi = self.avgprof[self.onmask] + eigenprofile * np.amax(eigenvalues)
                        profile_plot.plot(translate_phase(self.phs[self.onmask]), proflo, color='b', alpha=0.5,
                                          label=f"$+\\lambda_{{min}}\\mathbf{{e}}_{icomp}$")
                        profile_plot.plot(translate_phase(self.phs[self.onmask]), profhi, color='r', alpha=0.5,
                                          label=f"$+\\lambda_{{max}}\\mathbf{{e}}_{icomp}$")
            if show_nudot:
                ax_nudot = left_plot.twiny()
                ax_nudot.set_xlabel(r"$\dot{\nu}$ $(10^{-15}\mathrm{Hz^2})$")
                ax_nudot.plot(self.nudot_val, self.nudot_mjd, color='k', ls='--')
                if self.nudot_err is not None:
                    ax_nudot.fill_betweenx(self.nudot_mjd, self.nudot_val - self.nudot_err,
                                           self.nudot_val + self.nudot_err,
                                           color='k', alpha=0.3)
                nudots = self.nudot_val[(self.nudot_mjd > extent[2]) & (self.nudot_mjd < extent[3])]
                if nudot_bounds is None:
                    nudot_max = np.amax(nudots)
                    nudot_min = np.amin(nudots)
                else:
                    nudot_min,nudot_max = nudot_bounds
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

        profile_plot.legend(bbox_to_anchor=(-0.1, 1), loc='upper right', prop={'size': 12})

        bbox = kernel_plot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width1, height1 = bbox.width, bbox.height
        pwidth1 = kernel_plot.get_xlim()[1] - kernel_plot.get_xlim()[0]
        pheight1 = kernel_plot.get_ylim()[1] - kernel_plot.get_ylim()[0]

        bbox = main_plot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        width2, height2 = bbox.width, bbox.height
        pwidth2 = main_plot.get_xlim()[1] - main_plot.get_xlim()[0]
        pheight2 = main_plot.get_ylim()[1] - main_plot.get_ylim()[0]
        rwidth = (pwidth2 / width2) / (pwidth1 / width1)
        rheight = (pheight2 / height2) / (pheight1 / height1)
        if rwidth != 1.0 or rheight != 1.0:
            print("Notice Kernel scale wrong:", rwidth, rheight)

        if not (outname is None):
            plt.savefig(outname, bbox_inches='tight')
        else:
            plt.show()

    def piecewise_pca_analysis(self, bounds):
        def run_pca(data):
            pca = sklearn.decomposition.PCA(n_components=3)
            data -= np.median(data, axis=0)
            eigenvalues = pca.fit_transform(data).T
            eigenprofiles = pca.components_
            return eigenvalues, eigenprofiles

        nsub = len(self.mjd)
        model_profiles_at_data = np.reshape(self.pred_mean, (nsub, -1))
        _,nout = model_profiles_at_data.shape
        nudot_resamp2 = np.interp(self.mjd, self.nudot_mjd,
                                  self.nudot_val)
        nudot_resamp2 -= np.mean(nudot_resamp2)
        model_profiles = np.reshape(self.pred_mean_resample, (self.number_output_days, -1))
        mjd = self.mjd_resampled + self.mjd[0]
        chunks = []
        for i, bound in enumerate(bounds):
            mask = (mjd > bound[0]) & (mjd <= bound[1])
            mask_at_data = (self.mjd > bound[0]) & (self.mjd <= bound[1])
            data_eigenvals,eigenprofiles = run_pca(model_profiles_at_data[mask_at_data])
            eigenvals = np.dot(eigenprofiles, model_profiles[mask].T)
            r,_ = scipy.stats.pearsonr(data_eigenvals[0], nudot_resamp2[mask_at_data])
            if r < 0:
                eigenvals *= -1
                eigenprofiles *= -1
                data_eigenvals *= -1
                r*=-1
            dchunk = self.data[mask_at_data][:, self.onmask]
            chunks.append((mask, bound, eigenvals, eigenprofiles, data_eigenvals,r,dchunk))
        return chunks


    def get_correlation(self,icomp=0):
        nudot_resamp2 = np.interp(self.mjd, self.nudot_mjd,
                                  self.nudot_val)
        pearsonr = scipy.stats.pearsonr(self.eigenvalues[icomp], nudot_resamp2)
        spearmanr = scipy.stats.spearmanr(self.eigenvalues[icomp], nudot_resamp2)
        return pearsonr,spearmanr

    def pack_gp(self):
        """Pack the GP object into a byte array for storage"""
        print("Pack GP")
        self.cel_gp_pack = dill.dumps(self.cel_gp)
        self.cel_gp=None

    def unpack_gp(self):
        """Unpack the GP object from a byte array"""
        if self.cel_gp_pack is not None:
            print("Unpack GP")
            self.cel_gp = dill.loads(self.cel_gp_pack)
            self.cel_gp_pack=None