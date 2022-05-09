"""Fit mixture models with arbitrary components to data."""

import numpy as np
from src.mixmod import estimate
from random import random
from scipy.stats import rv_continuous


def get_loglikelihood(data, dists, params, params_fix, weights):
    p = 0
    model_params = zip(dists, params, params_fix, weights)
    for dist, param, param_fix, weight in model_params:
        p += weight * dist.pdf(data, **param_fix, **param)  # This step is not log b/c we are summing the contribution of each component
    return np.log(p).sum()


def get_posterior(data, dists, params, params_fix, weights):
    p = get_pdfstack(data, dists, params, params_fix, weights)
    return p / p.sum(axis=0)  # Normalize stack to yield posterior


def get_cdfstack(data, dists, params, params_fix, weights):
    """Return array of cdf components evaluated at data."""

    model_params = zip(dists, params, params_fix, weights)
    ps = [weight * dist.cdf(data, **param, **param_fix) for dist, param, param_fix, weight in model_params]
    return np.stack(ps, axis=0)  # Stack results


def get_pdfstack(data, dists, params, params_fix, weights):
    """Return array of pdf components evaluated at data."""

    model_params = zip(dists, params, params_fix, weights)
    ps = [weight * dist.pdf(data, **param, **param_fix) for dist, param, param_fix, weight in model_params]
    return np.stack(ps, axis=0)  # Stack results


class MixtureModel(rv_continuous):

    def __init__(self, dists, params=None, params_fix=None, weights=None, name='mixture'):
        # Initialize base class
        super().__init__(self, name=name)
        self.a = min([dist.a for dist in dists])
        self.b = max([dist.b for dist in dists])

        # Model parameters
        self.dists = dists.copy()
        self.params = [{} for _ in range(len(self.dists))] if params is None else params.copy()
        self.params_fix = [{} for _ in range(len(self.dists))] if params_fix is None else params_fix.copy()
        self.weights = np.full(len(self.dists), 1 / len(self.dists)) if weights is None else weights.copy()
        self.converged = False

    def __repr__(self):
        pad = 13 * ' '
        dists = [dist.name for dist in self.dists]
        return f'MixtureModel(dists={dists},\n'\
               f'{pad}params={self.params},\n'\
               f'{pad}params_fix={self.params_fix},\n'\
               f'{pad}weights={self.weights},\n'\
               f'{pad}name={self.name})'

    def clear(self):
        self.params = [{} for _ in range(len(self.dists))]
        self.weights = np.full(len(self.dists), 1 / len(self.dists))
        self.converged = False

    def fit(self, data, max_iter=250, tol=1E-3, verbose=False):
        # Initialize params, using temporary values to preserve originals in case of error
        weights_opt = self.weights.copy()
        params_opt = []
        for dist, param, param_fix in zip(self.dists, self.params, self.params_fix):
            sample = np.random.choice(data, max(1, int(len(data) * random())))  # Use random sample to initialize
            cfe = estimate.cfes[dist.name]  # Get closed-form estimator
            param_init = {**cfe(sample, param_fix=param_fix), **param}  # Overwrite random initials with any provided initials
            params_opt.append(param_init)

        for i in range(1, max_iter + 1):
            ll0 = get_loglikelihood(data, self.dists, params_opt, self.params_fix, weights_opt)

            # Expectation
            expts = get_posterior(data, self.dists, params_opt, self.params_fix, weights_opt)
            weights_opt = expts.sum(axis=1) / expts.sum()

            # Maximization
            for dist, param_opt, param_fix, expt in zip(self.dists, params_opt, self.params_fix, expts):
                mle = estimate.mles[dist.name]  # Get MLE function
                opt = mle(data, param_fix=param_fix, expt=expt, initial=param_opt)  # Get updated parameters
                param_opt.update(opt)  # Update is dict built-in method
            ll = get_loglikelihood(data, self.dists, params_opt, self.params_fix, weights_opt)

            # Print output
            if verbose:
                print(i, ll, sep=': ')

            # Test numerical exception then convergence
            if np.isnan(ll) or np.isinf(ll):
                break
            if ll - ll0 < tol:
                self.converged = True
                break

        self.params = params_opt
        self.weights = weights_opt.tolist()

        return i, ll

    def loglikelihood(self, data):
        return get_loglikelihood(data, self.dists, self.params, self.params_fix, self.weights)

    def posterior(self, data):
        return get_posterior(data, self.dists, self.params, self.params_fix, self.weights)

    def cdf_comp(self, x, comp=None):
        if comp is None:
            p = get_cdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return p.sum(axis=0)
        elif comp == 'all':
            p = get_cdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return p
        else:
            model_params = zip(self.dists, self.params, self.params_fix, self.weights)
            dist, param, param_fix, weight = list(model_params)[comp]
            p = weight * dist.cdf(x, **param_fix, **param)
            return p

    def pdf_comp(self, x, comp=None):
        if comp is None:
            p = get_pdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return p.sum(axis=0)
        elif comp == 'all':
            p = get_pdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return p
        else:
            model_params = zip(self.dists, self.params, self.params_fix, self.weights)
            dist, param, param_fix, weight = list(model_params)[comp]
            p = weight * dist.pdf(x, **param_fix, **param)
            return p

    def _cdf(self, x):
        return self.cdf_comp(x)

    def _pdf(self, x):
        return self.pdf_comp(x)
