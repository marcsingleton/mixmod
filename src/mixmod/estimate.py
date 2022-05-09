"""Functions for estimators of distributions with weighted data."""

import numpy as np
import scipy.optimize as opt
from math import exp, log, pi, sqrt
from scipy.special import digamma


# Functions for MLEs with no closed-form solutions
def create_fisk_scale(data, expt=None):
    expt = np.full(len(data), 1) if expt is None else expt

    def fisk_scale(scale):
        # Compute sums
        e = expt.sum()
        q = ((expt * data) / (scale + data)).sum()

        return 2 * q - e

    return fisk_scale


def create_fisk_shape(data, expt=None, scale=1):
    expt = np.full(len(data), 1) if expt is None else expt

    def fisk_shape(c):
        # Compute summands
        r = data / scale
        s = 1 / c + np.log(r) - 2 * np.log(r) * r ** c / (1 + r ** c)

        return (expt * s).sum()

    return fisk_shape


def create_gamma_shape(data, expt=None):
    expt = np.full(len(data), 1) if expt is None else expt

    def gamma_shape(a):
        # Compute sums
        e = expt.sum()
        ed = (expt * data).sum()
        elogd = (expt * np.log(data)).sum()

        return elogd - e * log(ed / e) + e * (log(a) - digamma(a))

    return gamma_shape


# Maximum likelihood estimators
def mle_expon(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Scale
    if 'scale' not in param_fix:
        e = expt.sum()
        ed = (expt * data).sum()
        scale = ed / e
        ests['scale'] = scale

    return ests


def mle_fisk(data, param_fix={}, expt=None, initial=None):
    expt = np.full(len(data), 1) if expt is None else expt
    initial = mm_fisk(data) if initial is None else initial
    ests = {}

    # Scale
    if 'scale' not in param_fix:
        fisk_scale = create_fisk_scale(data, expt)
        scale = opt.newton(fisk_scale, initial['scale'])
        ests['scale'] = scale
    else:
        scale = param_fix['scale']

    # Shape
    if 'c' not in param_fix:
        fisk_shape = create_fisk_shape(data, expt, scale)
        c = opt.newton(fisk_shape, initial['c'])
        ests['c'] = c

    return ests


def mle_gamma(data, param_fix={}, expt=None, initial=None):
    expt = np.full(len(data), 1) if expt is None else expt
    initial = mm_gamma(data) if initial is None else initial
    ests = {}

    # Shape
    if 'a' not in param_fix:
        gamma_shape = create_gamma_shape(data, expt)
        try:
            a = opt.newton(gamma_shape, initial['a'])
        except ValueError:  # Catch an error raised by a trial value below zero
            # Half and double endpoints until a sign difference
            lower = initial['a'] / 2
            upper = initial['a'] * 2
            while np.sign(gamma_shape(lower)) == np.sign(gamma_shape(upper)):
                lower /= 2
                upper *= 2
            a = opt.brentq(gamma_shape, lower, upper)
        ests['a'] = a
    else:
        a = param_fix['a']

    # Scale
    if 'scale' not in param_fix:
        scale = (expt * data).sum() / (a * expt.sum())
        ests['scale'] = scale

    return ests


def mle_laplace(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt[data.argsort()]  # Sort expt with data
    data = np.sort(data)  # Re-assign using np.sort() to prevent in-place sort
    ests = {}

    # Loc
    if 'loc' not in param_fix:
        # Find index of first point greater than center of mass
        cm = expt.sum() / 2
        e_cum = expt.cumsum()
        idx = np.argmax(e_cum > cm)

        # Linear interpolation as needed
        if data[idx] == data[idx - 1]:
            loc = data[idx]
        else:
            m = (e_cum[idx] - e_cum[idx - 1]) / (data[idx] - data[idx - 1])
            b = e_cum[idx] - m * data[idx]
            loc = (cm - b) / m
        ests['loc'] = loc
    else:
        loc = param_fix['loc']

    # Scale
    if 'scale' not in param_fix:
        e = expt.sum()
        d_abserr = abs(data - loc)
        scale = (expt * d_abserr).sum() / e
        ests['scale'] = scale

    return ests


def mle_levy(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Scale
    if 'scale' not in param_fix:
        e = expt.sum()
        edivd = (expt / data).sum()
        scale = e / edivd
        ests['scale'] = scale

    return ests


def mle_lognorm(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Scale
    if 'scale' not in param_fix:
        e = expt.sum()
        elogd = (expt * np.log(data)).sum()
        scale = exp(elogd / e)
        ests['scale'] = scale
    else:
        scale = param_fix['scale']

    # Shape
    if 's' not in param_fix:
        e = expt.sum()
        logd_sqerr = (np.log(data) - log(scale)) ** 2
        s = sqrt((expt * logd_sqerr).sum() / e)
        ests['s'] = s

    return ests


def mle_norm(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Loc
    if 'loc' not in param_fix:
        e = expt.sum()
        ed = (expt * data).sum()
        loc = ed / e
        ests['loc'] = loc
    else:
        loc = param_fix['loc']

    # Scale
    if 'scale' not in param_fix:
        e = expt.sum()
        d_sqerr = (data - loc) ** 2
        scale = np.sqrt((expt * d_sqerr).sum() / e)
        ests['scale'] = scale

    return ests


def mle_pareto(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Scale
    if 'scale' not in param_fix:
        scale = min(data)
        ests['scale'] = scale
    else:
        scale = param_fix['scale']

    # Shape
    if 'b' not in param_fix:
        e = expt.sum()
        elogd = (expt * np.log(data)).sum()
        b = e / (elogd - e * log(scale))
        ests['b'] = b

    return ests


def mle_uniform(data, param_fix={}, **kwargs):
    ests = {}

    # Loc
    if 'loc' not in param_fix:
        loc = min(data)
        ests['loc'] = loc
    else:
        loc = param_fix['loc']

    # Scale
    if 'scale' not in param_fix:
        scale = max(data) - loc
        ests['scale'] = scale

    return ests


# Method of moments estimators (for providing initial values for MLEs without closed forms)
def mm_fisk(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Moments
    logdata = np.log(data)
    m1 = (logdata * expt).sum() / expt.sum()
    m2 = (logdata ** 2 * expt).sum() / expt.sum()

    # Estimators
    if 'c' not in param_fix:
        ests['c'] = pi / sqrt(3 * (m2 - m1 ** 2))
    if 'scale' not in param_fix:
        ests['scale'] = exp(m1)

    return ests


def mm_gamma(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Moments
    m1 = (data * expt).sum() / expt.sum()
    m2 = (data ** 2 * expt).sum() / expt.sum()

    # Estimators
    if 'a' not in param_fix:
        ests['a'] = m1 ** 2 / (m2 - m1 ** 2)
    if 'scale' not in param_fix:
        ests['scale'] = (m2 - m1 ** 2) / m1

    return ests


def mm_lognorm(data, param_fix={}, expt=None, **kwargs):
    expt = np.full(len(data), 1) if expt is None else expt
    ests = {}

    # Moments
    m1 = (data * expt).sum() / expt.sum()
    m2 = (data ** 2 * expt).sum() / expt.sum()

    # Parameters of transformed lognorm
    mu = 2 * log(m1) - 0.5 * log(m2)
    var = log(m2) - 2 * log(m1)

    # Estimators
    if 's' not in param_fix:
        ests['s'] = sqrt(var)
    if 'scale' not in param_fix:
        ests['scale'] = exp(mu)

    return ests


# MLEs and MMEs for access by distribution name
mles = {'expon': mle_expon,
        'fisk': mle_fisk,
        'gamma': mle_gamma,
        'laplace': mle_laplace,
        'levy': mle_levy,
        'lognorm': mle_lognorm,
        'norm': mle_norm,
        'pareto': mle_pareto,
        'uniform': mle_uniform}

mmes = {'fisk': mm_fisk,
        'gamma': mm_gamma,
        'lognorm': mm_lognorm}

# Closed-form estimators
cfes = {**mles,
        'fisk': mm_fisk,
        'gamma': mm_gamma}
