import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def rainbowplot(phs, onmask, xo2, pred_mean2, pred_std2, outsub, outname):
    zz2 = np.reshape(pred_mean2, (outsub, -1))
    ze2 = np.reshape(pred_std2, (outsub, -1))

    plt.rcParams.update({'font.size': 12})

    sigma = np.abs(zz2 / ze2)

    vm = np.amax(np.abs(zz2))

    extent = (phs[onmask][0] - 0.5, phs[onmask][-1] - 0.5, mjd[0], mjd[-1])

    kvals = np.reshape(cel_gp.kernel.get_value(xo2), (outsub, -1))
    _, outphs = kvals.shape
    kvals = np.roll(kvals, outphs // 2, 1)
    kvals = np.vstack((np.flip(kvals, axis=0), kvals))
    dp = phs[1] - phs[0]
    ds = np.reshape(xo2, (outsub, -1))[1, 0]
    height_ratio = 4
    phase_factor = dp / (xo2[1] - xo2[0])

    print(dp, ds)

    alpha = np.ones_like(zz2)
    alpha[sigma < 3] = scipy.special.erf(sigma[sigma < 3] / 2)

    fig, ((prof, kerplt), (a0, a1)) = plt.subplots(2, 2, facecolor='white', gridspec_kw={'width_ratios': [3, 1],
                                                                                         'height_ratios': [1,
                                                                                                           height_ratio],
                                                                                         'wspace': 0.02},
                                                   figsize=(12, 18))
    a0.set_title("GP Model")
    a0.imshow(zz2, aspect='auto', origin='lower', interpolation='nearest', cmap='rainbow', alpha=alpha,
                   extent=extent, vmin=-vm, vmax=vm)
    a0.set_xlabel("Phase")
    a0.set_ylabel("MJD")

    a0.set_ylim(extent[2], extent[3])
    a0.set_xlim(extent[0], extent[1])

    err = np.mean(ze2, axis=1)
    a1.plot(err, np.linspace(mjd[0], mjd[1], outsub), color='k', alpha=0.5)
    a1.plot(-err, np.linspace(mjd[0], mjd[1], outsub), color='k', alpha=0.5)
    a1.plot(2 * err, np.linspace(mjd[0], mjd[1], outsub), color='k', ls=':', alpha=0.5)
    a1.plot(-2 * err, np.linspace(mjd[0], mjd[1], outsub), color='k', ls=':', alpha=0.5)
    a1.set_ylim(mjd[0], mjd[1])
    a1.set_xlabel("Signal (peak flux)")

    a1data = np.tile(np.linspace(-vm, vm, 256), outsub).reshape(outsub, -1)

    a1sigma = np.abs(a1data.T / err).T
    a1alpha = np.ones_like(a1data)
    a1alpha[a1sigma < 3] = scipy.special.erf(a1sigma[a1sigma < 3] / 2)

    a1.set_title("Uncertanty & Colourscale")

    a1.imshow(a1data, cmap='rainbow', aspect='auto', extent=(-vm, vm, mjd[0], mjd[1]), alpha=a1alpha,
              interpolation='nearest', origin='lower')

    a1.yaxis.set_label_position("right")
    a1.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')

    a1.set_ylabel("MJD")

    # Find phase with most variation
    most_variable_phase_bin = np.argmax(np.std(zz2, axis=0))

    print(most_variable_phase_bin)
    lowprof = zz[:, most_variable_phase_bin] < -0.01
    hiprof = zz[:, most_variable_phase_bin] > 0.01

    print(lowprof.shape, data.shape)
    prof.plot(phs[onmask] - 0.5, np.mean(data[lowprof][:, onmask], axis=0), color='b', alpha=0.5, label='low')
    prof.plot(phs[onmask] - 0.5, np.mean(data[hiprof][:, onmask], axis=0), color='r', alpha=0.5, label='high')
    prof.plot(phs[onmask] - 0.5, avgprof[onmask], color='k', lw=2, label='total')
    prof.axvline(phs[onmask][most_variable_phase_bin] - 0.5, color='gray', ls='--', alpha=0.4,
                 label='most variable phase')
    prof.legend()

    prof.set_xlim(extent[0], extent[1])
    prof.set_xlabel("Phase")
    prof.set_ylabel("Amplidute")
    prof.set_title("Average Profile")

    kerplt.set_title("Kernel")
    kerplt.imshow(kvals, aspect='auto', origin='lower',
                  extent=(-outphs * dp / 2, outphs * dp / 2, -outsub * ds, outsub * ds), cmap='Greys')
    kerplt.set_xlabel("Phase Lag")
    kerplt.set_ylabel("Time Lag (days)")

    kerplt.yaxis.set_label_position("right")
    kerplt.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')

    rx = (extent[1] - extent[0]) / 3
    ry = (extent[3] - extent[2]) / height_ratio
    kerplt.set_xlim(-rx, rx)
    kerplt.set_ylim(-ry, ry)

    if not (outname is None):
        plt.savefig(outname)
    else:
        plt.show()
