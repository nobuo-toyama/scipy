# ====================================
#     Scipy
# ====================================
# ====================================
#     Scipy:stats
# ====================================
# ==== Random variables
from scipy import stats
from scipy.stats import norm

# ---- Getting help
print('bounds of distribution lower: %s, upper: %s' % (norm.a, norm.b))
rv = norm()
dir(rv)  # reformatted

# ---- Common methods
norm.cdf(0)
# To compute the cdf at a number of points, we can pass a list or a numpy array.
norm.cdf([-1., 0, 1])
import numpy as np
norm.cdf(np.array([-1., 0, 1]))
norm.rvs(size=3)
# To achieve reproducibility, you can explicitly seed a global variable
np.random.seed(1234)
# Relying on a global state is not recommended, though.
norm.rvs(size=5, random_state=1234)

# ---- Shifting and scaling
# All continuous distributions take loc and scale as keyword parameters
# to adjust the location and scale of the distribution,
# e.g., for the standard normal distribution,
# the location is the mean and the scale is the standard deviation.
norm.stats(loc=3, scale=4, moments="mv")

from scipy.stats import expon
expon.mean(scale=3.)

from scipy.stats import uniform
uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4)

# ---- Shape parameters
from scipy.stats import gamma
gamma.numargs
gamma.shapes
gamma(1, scale=2.).stats(moments="mv")
gamma(a=1, scale=2.).stats(moments="mv")

# ==== Analysing one sample
np.random.seed(282629734)
x = stats.t.rvs(10, size=1000)

# ---- Descriptive statistics
print(x.min())
print(x.max())
print(x.mean())
print(x.var())

# How do the sample properties compare to their theoretical counterparts?
m, v, s, k = stats.t.stats(10, moments='mvsk')
n, (smin, smax), sm, sv, ss, sk = stats.describe(x)
sstr = '%-14s mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
print(sstr % ('distribution:', m, v, s ,k))
print(sstr % ('sample:', sm, sv, ss, sk))

# ---- T-test and KS-test
print('t-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_1samp(x, m))

tt = (sm-m)/np.sqrt(sv/float(n))  # t-statistic for mean
pval = stats.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
print('t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))

print('KS-statistic D = %6.3f pvalue = %6.4f' % stats.kstest(x, 't', (10,)))
print('KS-statistic D = %6.3f pvalue = %6.4f' % stats.kstest(x, 'norm'))
d, pval = stats.kstest((x-x.mean())/x.std(), 'norm')
print('KS-statistic D = %6.3f pvalue = %6.4f' % (d, pval))

# ---- Tails of the distribution
crit01, crit05, crit10 = stats.t.ppf([1-0.01, 1-0.05, 1-0.10], 10)
print('critical values from ppf at 1%%, 5%% and 10%% %8.4f %8.4f %8.4f' % (crit01, crit05, crit10))
print('critical values from isf at 1%%, 5%% and 10%% %8.4f %8.4f %8.4f' % tuple(stats.t.isf([0.01,0.05,0.10],10)))
freq01 = np.sum(x>crit01) / float(n) * 100
freq05 = np.sum(x>crit05) / float(n) * 100
freq10 = np.sum(x>crit10) / float(n) * 100
print('sample %%-frequency at 1%%, 5%% and 10%% tail %8.4f %8.4f %8.4f' % (freq01, freq05, freq10))

freq05l = np.sum(stats.t.rvs(10, size=10000) > crit05) / 10000.0 * 100
print('larger sample %%-frequency at 5%% tail %8.4f' % freq05l)

# The chisquare test can be used to test whether for a finite number of bins,
# the observed frequencies differ significantly from
# the probabilities of the hypothesized distribution.
quantiles = [0.0, 0.01, 0.05, 0.1, 1-0.10, 1-0.05, 1-0.01, 1.0]
crit = stats.t.ppf(quantiles, 10)
crit
n_sample = x.size
freqcount = np.histogram(x, bins=crit)[0]
tprob = np.diff(quantiles)
nprob = np.diff(stats.norm.cdf(crit))
tch, tpval = stats.chisquare(freqcount, tprob*n_sample)
nch, npval = stats.chisquare(freqcount, nprob*n_sample)
print('chisquare for t:      chi2 = %6.2f pvalue = %6.4f' % (tch, tpval))
print('chisquare for normal: chi2 = %6.2f pvalue = %6.4f' % (nch, npval))

tdof, tloc, tscale = stats.t.fit(x)
nloc, nscale = stats.norm.fit(x)
tprob = np.diff(stats.t.cdf(crit, tdof, loc=tloc, scale=tscale))
nprob = np.diff(stats.norm.cdf(crit, loc=nloc, scale=nscale))
tch, tpval = stats.chisquare(freqcount, tprob*n_sample)
nch, npval = stats.chisquare(freqcount, nprob*n_sample)
print('chisquare for t:      chi2 = %6.2f pvalue = %6.4f' % (tch, tpval))
print('chisquare for normal: chi2 = %6.2f pvalue = %6.4f' % (nch, npval))

# ---- Special tests for normal distributions
# First, we can test if skew and kurtosis of our sample differ
# significantly from those of a normal distribution:
print('normal skewtest teststat = %6.3f pvalue = %6.4f' % stats.skewtest(x))
print('normal kurtosistest teststat = %6.3f pvalue = %6.4f' % stats.kurtosistest(x))

# These two tests are combined in the normality test
print('normaltest teststat = %6.3f pvalue = %6.4f' % stats.normaltest(x))

# Since skew and kurtosis of our sample are based on central moments,
# we get exactly the same results if we test the standardized sample:
print('normaltest teststat = %6.3f pvalue = %6.4f' % stats.normaltest((x-x.mean())/x.std()))

# Because normality is rejected so strongly, we can check
# whether the normaltest gives reasonable results for other cases:
print('normaltest teststat = %6.3f pvalue = %6.4f' % stats.normaltest(stats.t.rvs(10, size=100)))
print('normaltest teststat = %6.3f pvalue = %6.4f' % stats.normaltest(stats.norm.rvs(size=1000)))

# ---- Comparing two samples
# ---- Comparing means
# Test with sample with identical means:
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
stats.ttest_ind(rvs1, rvs2)

# Test with sample with different means:
rvs3 = stats.norm.rvs(loc=8, scale=10, size=500)
stats.ttest_ind(rvs1, rvs3)

# ---- Kolmogorov-Smirnov test for two samples ks_2samp
stats.ks_2samp(rvs1, rvs2)
stats.ks_2samp(rvs1, rvs3)

# ==== Kernel density estimation
# ---- Univariate estimation
from scipy import stats
import matplotlib.pyplot as plt

x1 = np.array([-7, -5, 1, 4, 5], dtype=np.float64)
kde1 = stats.gaussian_kde(x1)
kde2 = stats.gaussian_kde(x1, bw_method='silverman')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # rug plot
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott's Rule")
ax.plot(x_eval, kde2(x_eval), 'r-', label="Silverman's Rule")
plt.show()

def my_kde_bandwidth(obj, fac=1./5):
    """We use Scott's Rule, multiplied by a constant factor."""
    return np.power(obj.n, -1. / (obj.d + 4)) * fac


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # rug plot
kde3 = stats.gaussian_kde(x1, bw_method=my_kde_bandwidth)
ax.plot(x_eval, kde3(x_eval), 'g-', label="With smaller BW")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(12456)
x1 = np.random.normal(size=200)  # random data, normal distribution
xs = np.linspace(x1.min()-1, x1.max()+1, 200)
kde1 = stats.gaussian_kde(x1)
kde2 = stats.gaussian_kde(x1, bw_method='silverman')

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(211)
ax1.plot(x1, np.zeros(x1.shape), 'b+', ms=12)  # rug plot
ax1.plot(xs, kde1(xs), 'k-', label="Scott's Rule")
ax1.plot(xs, kde2(xs), 'b-', label="Silverman's Rule")
ax1.plot(xs, stats.norm.pdf(xs), 'r--', label="True PDF")

ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.set_title("Normal (top) and Student's T$_{df=5}$ (bottom) distributions")
ax1.legend(loc=1)

x2 = stats.t.rvs(5, size=200)  # random data, T distribution
xs = np.linspace(x2.min() - 1, x2.max() + 1, 200)

kde3 = stats.gaussian_kde(x2)
kde4 = stats.gaussian_kde(x2, bw_method='silverman')

ax2 = fig.add_subplot(212)
ax2.plot(x2, np.zeros(x2.shape), 'b+', ms=12)  # rug plot
ax2.plot(xs, kde3(xs), 'k-', label="Scott's Rule")
ax2.plot(xs, kde4(xs), 'b-', label="Silverman's Rule")
ax2.plot(xs, stats.t.pdf(xs, 5), 'r--', label="True PDF")

ax2.set_xlabel('x')
ax2.set_ylabel('Density')

plt.show()

# We now take a look at a bimodal distribution with one wider
# and one narrower Gaussian feature.
from functools import partial

loc1, scale1, size1 = (-2, 1, 175)
loc2, scale2, size2 = (2, 0.2, 50)
x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),
                     np.random.normal(loc=loc2, scale=scale2, size=size2)])
x_eval = np.linspace(x2.min() - 1, x2.max() + 1, 500)

kde = stats.gaussian_kde(x2)
kde2 = stats.gaussian_kde(x2, bw_method='silverman')
kde3 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.2))
kde4 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.5))

pdf = stats.norm.pdf
bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1) * float(size1) / x2.size + \
              pdf(x_eval, loc=loc2, scale=scale2) * float(size2) / x2.size

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ax.plot(x2, np.zeros(x2.shape), 'b+', ms=12)
ax.plot(x_eval, kde(x_eval), 'k-', label="Scott's Rule")
ax.plot(x_eval, kde2(x_eval), 'b-', label="Silverman's Rule")
ax.plot(x_eval, kde3(x_eval), 'g-', label="Scott * 0.2")
ax.plot(x_eval, kde4(x_eval), 'c-', label="Scott * 0.5")
ax.plot(x_eval, bimodal_pdf, 'r--', label="Actual PDF")

ax.set_xlim([x_eval.min(), x_eval.max()])
ax.legend(loc=2)
ax.set_xlabel('x')
ax.set_ylabel('Density')
plt.show()

# ---- Multivariate estimation
# With gaussian_kde we can perform multivariate, as well as univariate estimation.
def measure(n):
    """Measurement model, return two coupled measurements."""
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1 + m2, m1 - m2


m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

# Then we apply the KDE to the data:
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel.evaluate(positions).T, X.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
ax.plot(m1, m2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()

# ---- Multiscale Graph Correlation (MGC)
# With multiscale_graphcorr, we can test for independence on high dimensional and nonlinear data.
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('classic')
from scipy.stats import multiscale_graphcorr

def mgc_plot(x, y, sim_name, mgc_dict=None, only_viz=False, only_mgc=False):
    """Plot sim and MGC-plot"""
    if not only_mgc:    # simulation
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_title(sim_name + " Simulation", fontsize=20)
        ax.scatter(x, y)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.axis('equal')
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        plt.show()
    if not only_viz:    # local correlation map
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        mgc_map = mgc_dict["mgc_map"]    # draw heatmap
        ax.set_title("Local Correlation Map", fontsize=20)
        im = ax.imshow(mgc_map, cmap='YlGnBu')
        cbar = ax.figure.colorbar(im, ax=ax)    # colorbar
        cbar.ax.set_ylabel("", rotation=-90, va="bottom")
        ax.invert_yaxis()
        for edge, spine in ax.spines.items():    # Turn spines off and create white grid.
            spine.set_visible(False)
        opt_scale = mgc_dict["opt_scale"]    # optimal scale
        ax.scatter(opt_scale[0], opt_scale[1],
                   marker='X', s=200, color='red')
        ax.tick_params(bottom="off", left="off")    # other formatting
        ax.set_xlabel('#Neighbors for X', fontsize=15)
        ax.set_ylabel('#Neighbors for Y', fontsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.show()


np.random.seed(12345678)
x = np.linspace(-1, 1, num=100)
y = x + 0.3 * np.random.random(x.size)

mgc_plot(x, y, "Linear", only_viz=True)

# Now, we can see the test statistic, p-value, and MGC map visualized below.
stat, pvalue, mgc_dict = multiscale_graphcorr(x, y)
print("MGC test statistic: ", round(stat, 1))
print("P-value: ", round(pvalue, 1))
mgc_plot(x, y, "Linear", mgc_dict, only_mgc=True)

# The same can be done for nonlinear data sets.
np.random.seed(12345678)
