import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import psrcelery
import scipy.stats
import seaborn
import glob
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import argparse

parser  = argparse.ArgumentParser(description="Make plots from the output of psrcelery")
parser.add_argument('--psr', type=str, help='pulsar')
parser.add_argument('input', type=str, help='Input file')
args = parser.parse_args()


B=seaborn.color_palette("crest_r", as_cmap=True).colors
A=seaborn.color_palette("flare", as_cmap=True).colors
C = A[:127]+B[-127:]
cmap = ListedColormap(C[::-1])
B=seaborn.color_palette("crest_r", as_cmap=True).colors
A=seaborn.color_palette("flare", as_cmap=True).colors
C = B[-127:]+A[:127]
cmap2 = ListedColormap(C[::-1])
A=seaborn.color_palette("flare", as_cmap=True).colors
B=seaborn.color_palette("crest_r", as_cmap=True).colors
C = B[::2]+A[::2]
cmap3 = ListedColormap(C[::-1])



psr=args.psr
nudot = f"{psr}/nudot.asc"
glitches= np.atleast_1d(np.loadtxt(f"{psr}/glitch.txt",usecols=(1,)))

celpredict=args.input

lab=''
if 'afb' in celpredict:
    lab='_afb'
if 'combo' in celpredict:
    lab='_combo'
if 'cobra2' in celpredict:
    lab='_42ft'

celery = psrcelery.celery.loadCelery(celpredict)
celery.data_model_plot(outname=f"{celpredict}.pdf")

celery.dampen_edges()
celery.compute_eigenprofiles()

nn=1
if psr=="B1540-06" or psr=="B2035+36":
    nn=3
nudot_mjd, nudot_val, nudot_err ,nudot_other= np.loadtxt(nudot,usecols=(0,nn,4,3),unpack=True)
# if psr=="B0919+06":
#     mm=nudot_val > 0.5
#     nudot_val[mm]=0.5
if nn==1:
    nudot_val += nudot_other[0]-nudot_val[0]

celery.set_nudot(nudot_mjd,nudot_val,nudot_err)




#celery.kernel_values = np.zeros_like(celery.x_resampled).reshape((celery.number_output_days,-1))
lab=''
if 'afb' in celpredict:
    lab='_afb'
if 'combo' in celpredict:
    lab='_combo'
if 'cobra2' in celpredict:
    lab='_42ft'
print(f"{psr}_rainbowplot{lab}.pdf")

icomps=[0]
nudot_bound=None
if psr=="B1740-03":
    nudot_bound=[-9.5,-6.5]
if psr=="B1826-17":
    icomps=[3]
if psr=="B1828-11":
    nudot_bound=[-367.5, -364.0]
if psr=="B1917+00":
    nudot_bound=[-4.744,-4.738]
if psr=="B0919+06":
    nudot_bound=[-74.3,-73.5]
    nudot_bound=[-74.4,-73.55]
if psr=="B1818-04":
    nudot_bound=[-17.8,-17.62]
    icomps=[0]
if psr=="J2043+2740":
    icomps=[0,1]
if psr=="B1540-06":
    nudot_bound=[-1.76, -1.725]
if psr=="B1917+00" and (lab=="_combo" or lab=='_afb'):
    print("test: swap")
    celery.eigenprofiles[0], celery.eigenprofiles[1] = celery.eigenprofiles[1], celery.eigenprofiles[0]
    celery.eigenvalues[0], celery.eigenvalues[1] = celery.eigenvalues[1], celery.eigenvalues[0]
    celery.eigenvalues_err[0], celery.eigenvalues_err[1] = celery.eigenvalues_err[1], celery.eigenvalues_err[0]
    celery.eigenprofiles_resample[0], celery.eigenprofiles_resample[1] = celery.eigenprofiles_resample[1], celery.eigenprofiles_resample[0]
    celery.eigenvalues_resample[0], celery.eigenvalues_resample[1] = celery.eigenvalues_resample[1], celery.eigenvalues_resample[0]
    celery.eigenvalues_resample_err[0], celery.eigenvalues_resample_err[1] = celery.eigenvalues_resample_err[1], celery.eigenvalues_resample_err[0]

print(celery.eigenvalues_resample.shape)
pearson,spearman = celery.get_correlation(icomps[0])

# Get errors on correlation coefficient.
import scipy.interpolate
celery.use_standard_kernel(matern=True)
celery.set_parameter_vector(celery.parameter_vector)
celery.cel_gp.compute(celery.x, celery.yerr)


X = np.round(celery.mjd)
X-=X[0]
xx=np.arange(np.amax(X)+1)
z = celery.cel_gp.kernel.get_value(xx)
c = scipy.interpolate.interp1d(xx, z, kind='linear')
def covariance_function(a,b):
    return c(np.abs(a-b))

n = len(X)

# Initialize the Cholesky factor L as a zero matrix
L = np.zeros((n, n))

# Compute the Cholesky factor L incrementally
for i in range(n):
    for j in range(i + 1):
        if i == j:
            # Diagonal elements
            sum_k = np.sum(L[i, :j] ** 2)
            L[i, j] = np.sqrt(covariance_function(X[i], X[i]) - sum_k)
        else:
            # Off-diagonal elements
            sum_k = np.sum(L[i, :j] * L[j, :j])
            L[i, j] = (covariance_function(X[i], X[j]) - sum_k) / L[j, j]
nudot_resamp2 = np.interp(celery.mjd, celery.nudot_mjd,
                                          celery.nudot_val)

stats_pearson=[]
stats_spearman=[]
for i in range(1000):
    sample = L.dot(np.random.normal(size=n))
    s,p = scipy.stats.pearsonr(sample,nudot_resamp2)
    s2,p2 = scipy.stats.spearmanr(sample,nudot_resamp2)
    stats_pearson.append(s)
    stats_spearman.append(s2)

r_error = np.std(stats_pearson)
bins=np.linspace(-1,1,61)
s_error = np.std(stats_spearman)

from uncertainties import ufloat
pearson = ufloat(pearson[0],r_error)
spearman = ufloat(spearman[0],s_error)

print(f"pearson: {pearson} spearman: {spearman}")
with open(f"{psr}_correlation{lab}.txt","w") as f:
    f.write(f"{psr} {lab} {pearson} {spearman}")


celery.rainbowplot(f"plots/{psr}_rainbowplot{lab}.pdf",pca_comps=icomps,title=psr,cmap=cmap3,glitches=glitches,nudot_bounds=nudot_bound,interpolation='antialiased')

