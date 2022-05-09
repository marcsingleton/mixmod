"""Class and functions for performing calculations with mixture models."""

from random import random

import numpy as np
from . import estimate


def _get_loglikelihood(data, dists, params, params_fix, weights):
    """Return log-likelihood of data according to mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    dists: list of rv_continuous instances
        Components of mixture model.
    params: list of dicts
        Free parameters of components of mixture model.
    params_fix: list of dicts
        Fixed parameters of components of mixture model.
    weights: list of floats
        Weights of components of mixture model.

    Returns
    -------
    p: float
        Log-likelihood of data.
    """
    p = 0
    model_params = zip(dists, params, params_fix, weights)
    for dist, param, param_fix, weight in model_params:
        p += weight * dist.pdf(data, **param_fix, **param)  # This step is not log b/c we are summing the contribution of each component
    return np.log(p).sum()


def _get_posterior(data, dists, params, params_fix, weights):
    """Return array of posterior probabilities of data for each component of mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    dists: list of rv_continuous instances
        Components of mixture model.
    params: list of dicts
        Free parameters of components of mixture model.
    params_fix: list of dicts
        Fixed parameters of components of mixture model.
    weights: list of floats
        Weights of components of mixture model.

    Returns
    -------
    ps: ndarray
        Array of posterior probabilities of data for each component of mixture
        model. Shape is (len(data), len(dists)).
    """
    ps = _get_pdfstack(data, dists, params, params_fix, weights)
    return ps / ps.sum(axis=0)  # Normalize stack to yield posterior


def _get_cdfstack(data, dists, params, params_fix, weights):
    """Return array of cdfs evaluated at data for each component of mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    dists: list of rv_continuous instances
        Components of mixture model.
    params: list of dicts
        Free parameters of components of mixture model.
    params_fix: list of dicts
        Fixed parameters of components of mixture model.
    weights: list of floats
        Weights of components of mixture model.

    Returns
    -------
    ps: ndarray
        Array of cdfs evaluated at data for each component of mixture model.
        Shape is (len(data), len(dists)).
    """
    model_params = zip(dists, params, params_fix, weights)
    ps = [weight * dist.cdf(data, **param, **param_fix) for dist, param, param_fix, weight in model_params]
    return np.stack(ps, axis=0)


def _get_pdfstack(data, dists, params, params_fix, weights):
    """Return array of pdfs evaluated at data for each component of mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    dists: list of rv_continuous instances
        Components of mixture model.
    params: list of dicts
        Free parameters of components of mixture model.
    params_fix: list of dicts
        Fixed parameters of components of mixture model.
    weights: list of floats
        Weights of components of mixture model.

    Returns
    -------
    ps: ndarray
        Array of pdfs evaluated at data for each component of mixture model.
        Shape is (len(data), len(dists)).
    """
    model_params = zip(dists, params, params_fix, weights)
    ps = [weight * dist.pdf(data, **param, **param_fix) for dist, param, param_fix, weight in model_params]
    return np.stack(ps, axis=0)


class MixtureModel:
    """Class for performing calculations with mixture models.

    Parameters
    ----------
    dists: list of rv_continuous instances
        Components of mixture model. Formally, the components are rv_continuous
        instances as defined in the scipy stats module. However, for most
        calculations only a pdf and cdf method are needed. The fit method
        requires rv_continuous instances since it uses their name attribute to
        select the correct estimator functions. It also uses the parameter
        names for each distribution as defined in the scipy stats module to set
        the correct keys in each param dict.
    params: list of dicts
        Free parameters of components of mixture model.
    params_fix: list of dicts
        Fixed parameters of components of mixture model.
    weights:
        Initial weights of components of model. Uses uniform distribution if
        None.
    name: str
        Name of mixture model for display when printing.
    """
    def __init__(self, dists, params=None, params_fix=None, weights=None, name='mixture'):
        # Check arguments
        if params is None:
            params = [{} for _ in range(len(dists))]
        elif len(params) != len(dists):
            raise RuntimeError('len(params) does not equal len(dists)')
        else:
            params = params.copy()

        if params_fix is None:
            params_fix = [{} for _ in range(len(dists))]
        elif len(params_fix) != len(dists):
            raise RuntimeError('len(params_fix) does not equal len(dists)')
        else:
            params_fix = params_fix.copy()

        if weights is None:
            weights = np.full(len(dists), 1 / len(dists))
        elif len(weights) != len(dists):
            raise RuntimeError('len(weights) does not equal len(dists)')
        else:
            weights = weights.copy()

        # Set instance attributes
        self.dists = dists.copy()
        self.params = params
        self.params_fix = params_fix
        self.weights = weights
        self.name = name
        self.converged = False

    def __repr__(self):
        pad = 13 * ' '
        dists = [dist.name for dist in self.dists]
        return (f'MixtureModel(dists={dists},\n'
                f'{pad}params={self.params},\n'
                f'{pad}params_fix={self.params_fix},\n'
                f'{pad}weights={self.weights},\n'
                f'{pad}name={self.name})')

    def clear(self):
        """Reset free parameters and weights of mixture model."""
        self.params = [{} for _ in range(len(self.dists))]
        self.weights = np.full(len(self.dists), 1 / len(self.dists))
        self.converged = False

    def fit(self, data, max_iter=250, tol=1E-3, verbose=False):
        """Fit components of mixture model with EM algorithm.

        Parameters
        ----------
        data: 1-D ndarray
            Data to fit mixture model.
        max_iter: int
            Maximum number of iterations. Must be at least 1.
        tol: positive int or float
            Optimization stops if the difference between log-likelihoods is less
            than tol between subsequent iterations.
        verbose: bool
            Prints log-likelihood at each iteration if True.

        Returns
        -------
        i, ll: (int, float)
            The number of iterations before a stop conditions was reached, and
            the final log-likelihood.
        """
        # Check arguments
        if max_iter < 1:
            raise ValueError('max_iter must be at least 1')
        if tol <= 0:
            raise ValueError('tol must be positive')

        # Initialize params, using temporary values to preserve originals in case of error
        weights_opt = self.weights.copy()
        params_opt = []
        for dist, param, param_fix in zip(self.dists, self.params, self.params_fix):
            sample = np.random.choice(data, max(1, int(len(data) * random())))  # Use random sample to initialize
            cfe = estimate.cfes[dist.name]  # Get closed-form estimator
            param_init = {**cfe(sample, param_fix=param_fix), **param}  # Overwrite random initials with any provided initials
            params_opt.append(param_init)

        for i in range(1, max_iter + 1):
            ll0 = _get_loglikelihood(data, self.dists, params_opt, self.params_fix, weights_opt)

            # Expectation
            expts = _get_posterior(data, self.dists, params_opt, self.params_fix, weights_opt)
            weights_opt = expts.sum(axis=1) / expts.sum()

            # Maximization
            for dist, param_opt, param_fix, expt in zip(self.dists, params_opt, self.params_fix, expts):
                mle = estimate.mles[dist.name]  # Get MLE function
                opt = mle(data, param_fix=param_fix, expt=expt, initial=param_opt)  # Get updated parameters
                param_opt.update(opt)
            ll = _get_loglikelihood(data, self.dists, params_opt, self.params_fix, weights_opt)

            # Print output
            if verbose:
                print(i, ll, sep=': ')

            # Test numerical exception then convergence
            if np.isnan(ll) or np.isinf(ll):
                break
            if abs(ll - ll0) < tol:
                self.converged = True
                break

        self.params = params_opt
        self.weights = weights_opt.tolist()

        return i, ll

    def loglikelihood(self, data):
        """Return log-likelihood of data according to mixture model.

        Parameters
        ----------
        data: 1-D ndarray
            Values at which to evaluate components of mixture model.

        Returns
        -------
        p: float
            Log-likelihood of data.
        """
        return _get_loglikelihood(data, self.dists, self.params, self.params_fix, self.weights)

    def posterior(self, data):
        """Return array of posterior probabilities of data for each component of mixture model.

        Parameters
        ----------
        data: 1-D ndarray
            Values at which to evaluate components of mixture model.

        Returns
        -------
        ps: ndarray
            Array of posterior probabilities of data for each component of mixture
            model. Shape is (len(data), len(self.dists)).
        """
        return _get_posterior(data, self.dists, self.params, self.params_fix, self.weights)

    def cdf(self, x, component='sum'):
        """Return cdf evaluated at x.

        Parameters
        ----------
        x: 1-D ndarray
            Values at which to evaluate components of mixture model.
        component: 'sum', 'all', or int
            If 'sum', the cdfs are summed across components. If 'all', the cdf
            of each component is returned as an ndarray with shape (len(x),
            len(self.dists)). If component is an int, the cdf of the
            corresponding component is returned.

        Returns
        -------
        ps: ndarray
            cdf evaluated at data.
        """
        # Check arguments
        if (component not in ['sum', 'all']) or isinstance(int, component):
            raise ValueError('component is not "sum", "all" or int')

        if component == 'sum':
            ps = _get_cdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return ps.sum(axis=0)
        elif component == 'all':
            ps = _get_cdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return ps
        else:
            model_params = zip(self.dists, self.params, self.params_fix, self.weights)
            dist, param, param_fix, weight = list(model_params)[component]
            ps = weight * dist.cdf(x, **param_fix, **param)
            return ps

    def pdf(self, x, component='sum'):
        """Return pdf evaluated at x.

        Parameters
        ----------
        x: 1-D ndarray
            Values at which to evaluate components of mixture model.
        component: 'sum', 'all', or int
            If 'sum', the pdfs are summed across components. If 'all', the pdf
            of each component is returned as an ndarray with shape (len(x),
            len(self.dists)). If component is an int, the pdf of the
            corresponding component is returned.

        Returns
        -------
        ps: ndarray
            pdf evaluated at data.
        """
        # Check arguments
        if (component not in ['sum', 'all']) or isinstance(int, component):
            raise ValueError('component is not "sum", "all" or int')

        if component is 'sum':
            ps = _get_pdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return ps.sum(axis=0)
        elif component == 'all':
            ps = _get_pdfstack(x, self.dists, self.params, self.params_fix, self.weights)
            return ps
        else:
            model_params = zip(self.dists, self.params, self.params_fix, self.weights)
            dist, param, param_fix, weight = list(model_params)[component]
            ps = weight * dist.pdf(x, **param_fix, **param)
            return ps
