import numpy as np
import scipy.optimize as opt
import scipy.stats


def align_and_scale(data, template, nharm=None,max_ishift=None):
    if nharm == "auto":
        power_template = np.absolute(np.fft.rfft(template)) ** 2
        k = (np.arange(len(power_template)) + 1)
        ex = np.cumsum(power_template[::-1])
        A2 = ex / k
        prob = scipy.stats.chi2.cdf(power_template[::-1], df=2, loc=0, scale=A2)
        t = np.argmax(prob > 0.99)
        nharm = len(power_template) - t

    return np.apply_along_axis(align_and_scale_one, 1, data, template=template, nharm=nharm,max_ishift=max_ishift)


def align_and_scale_one(prof, template, nharm=None,max_ishift=None):
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

    min_spectral_bins = min(len(f_template), len(f_prof))

    # The cross correlation of a and b is the inverse transform of FT(a) times the conjugate of FT(b)
    xspec = f_template[:min_spectral_bins] * f_prof[:min_spectral_bins].conj()  # "cross spectrum"
    xcor = np.fft.irfft(xspec)  # Cross correlation

    ishift = np.argmax(np.abs(xcor))  # estimate of the shift directly from the peak cross-correlation
    if max_ishift is not None:
        ishift = np.argmax(np.abs(xcor[:max_ishift]))
    # We need to define some bounds to search. (Actually this might not be optimal)
    for window in (np.arange(nbin // 4) + 1):
        lo = ishift - window
        hi = ishift + window
        if nharm is None or nharm > len(xspec):
            nh = len(xspec)
        else:
            nh = nharm
        lo_c = get_dchi(lo, nh, nbin)
        hi_c = get_dchi(hi, nh, nbin)
        if np.sign(lo_c) != np.sign(hi_c):
            break  # we found a good window

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


