"""Class and functions for performing calculations with mixture models."""

import numpy as np
from . import estimators


def _get_loglikelihood(data, components, params, params_fix, weights):
    """Return log-likelihood of data according to mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    components: list of rv_continuous instances
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
    model_params = zip(components, params, params_fix, weights)
    for component, param, param_fix, weight in model_params:
        p += weight * component.pdf(data, **param_fix, **param)  # This step is not log b/c we are summing the contribution of each component
    return np.log(p).sum()


def _get_posterior(data, components, params, params_fix, weights):
    """Return array of posterior probabilities of data for each component of mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    components: list of rv_continuous instances
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
        model. Shape is (len(data), len(components)).
    """
    ps = _get_pdfstack(data, components, params, params_fix, weights)
    return ps / ps.sum(axis=0)  # Normalize stack to yield posterior


def _get_cdfstack(data, components, params, params_fix, weights):
    """Return array of cdfs evaluated at data for each component of mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    components: list of rv_continuous instances
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
        Shape is (len(data), len(components)).
    """
    model_params = zip(components, params, params_fix, weights)
    ps = [weight * component.cdf(data, **param, **param_fix) for component, param, param_fix, weight in model_params]
    return np.stack(ps, axis=0)


def _get_pdfstack(data, components, params, params_fix, weights):
    """Return array of pdfs evaluated at data for each component of mixture model.

    Parameters
    ----------
    data: 1-D ndarray
        Values at which to evaluate components of mixture model.
    components: list of rv_continuous instances
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
        Shape is (len(data), len(components)).
    """
    model_params = zip(components, params, params_fix, weights)
    ps = [weight * component.pdf(data, **param, **param_fix) for component, param, param_fix, weight in model_params]
    return np.stack(ps, axis=0)


class MixtureModel:
    """Class for performing calculations with mixture models.

    params and params_fix are lists of dicts where each dict contains the names
    of the parameters of the components of the mixture model with their
    associated values. "Free" variables are specified in the params dicts, and
    "fixed" variables are specified in the params_fix dicts. The fit method
    updates free parameters only, that is, any parameters given in params_fix
    are not updated.

    The default value of a parameter, if defined, is used if that parameter is
    not in the params or params_fix dict. Once fit is called, the params dicts
    are populated with any parameters not defined in the corresponding
    params_fix dict.

    If params is given, its length must match the length of components, so the
    correspondence between the two is unambiguous. Likewise, if params_fix is
    given, its length must also match the length of components.

    A RuntimeError is raised if the same parameter is defined in corresponding
    params and params_fix dicts.

    Though the components are effectively instances of rv_continuous as defined
    in the scipy stats module, this condition is not formally checked. As long
    as each component implements a pdf and cdf method, most of the defined
    methods will execute correctly. The major exception is the fit method.
    First, it requires the components have name attributes since they are used
    to select the correct estimator functions. These estimator functions also
    use the parameter names as defined in the scipy stats module to set the
    keys in each param dict. Thus, the fit method is only implemented for the
    distributions with estimators defined in estimators.py.

    Parameters
    ----------
    components: list of rv_continuous instances
        Components of mixture model.
    params: list of dicts
        Initial values for the free parameters of components of mixture model.
    params_fix: list of dicts
        Fixed parameters of components of mixture model.
    weights: list of floats
        Initial weights of components of model. Uses a uniform distribution if
        None.
    name: str
        Name of mixture model for display when printing.

    Attributes
    ----------
    components
    params
    weights
    name
    converged: bool
        If call to fit successfully converged.

    Methods
    -------
    clear
    fit
    loglikelihood
    posterior
    cdf
    pdf
    """
    def __init__(self, components, params=None, params_fix=None, weights=None, name='mixture'):
        # Check arguments
        if params is None:
            params = [{} for _ in range(len(components))]
        elif len(params) != len(components):
            raise RuntimeError('len(params) does not equal len(components)')

        if params_fix is None:
            params_fix = [{} for _ in range(len(components))]
        elif len(params_fix) != len(components):
            raise RuntimeError('len(params_fix) does not equal len(components)')

        for param, param_fix in zip(params, params_fix):
            if set(param) & set(param_fix):
                raise RuntimeError('Corresponding dicts in params and params_fix define the same parameter')

        if weights is None:
            weights = [1 / len(components) for _ in range(len(components))]
        elif len(weights) != len(components):
            raise RuntimeError('len(weights) does not equal len(components)')

        # Set instance attributes
        self.components = components
        self.params = params
        self.params_fix = params_fix
        self.weights = weights
        self.name = name
        self.converged = False

    def __repr__(self):
        pad = 13 * ' '
        components = [component.name for component in self.components]
        return (f'MixtureModel(components={components},\n'
                f'{pad}params={self.params},\n'
                f'{pad}params_fix={self.params_fix},\n'
                f'{pad}weights={self.weights},\n'
                f'{pad}name=\'{self.name}\')')

    def clear(self):
        """Reset free parameters and weights of mixture model."""
        self.params = [{} for _ in range(len(self.components))]
        self.weights = [1 / len(self.components) for _ in range(len(self.components))]
        self.converged = False

    def fit(self, data, tol=1E-3, maxiter=250, verbose=False):
        """Fit the free parameters of the mixture model with EM algorithm.

        Only the "free" parameters are fit. Any parameters in the params_fix
        dicts are not changed.

        Parameters given in params dicts are used as initial estimates.
        Otherwise initial estimates are calculated from the data.

        If the log-likelihood is ever NaN or infinite, iteration stops. No
        warning or error is raised, but the converged attribute is not set to
        True. If the iteration did not converge and the number of iterations
        is not equal to maxiter, then a numerical exception occurred.

        Parameters
        ----------
        data: 1-D ndarray
            Data to fit mixture model.
        tol: positive int or float
            Optimization stops if the difference between log-likelihoods is less
            than tol between subsequent iterations.
        maxiter: int
            Maximum number of iterations. Must be at least 1.
        verbose: bool
            Prints log-likelihood at each iteration if True.

        Returns
        -------
        i, ll: (int, float)
            The number of iterations before a stop conditions was reached, and
            the final log-likelihood.
        """
        # Check arguments
        if maxiter < 1:
            raise ValueError('max_iter must be at least 1')
        if tol <= 0:
            raise ValueError('tol must be positive')

        # Initialize params, using temporary values to preserve originals in case of error
        weights_opt = self.weights.copy()
        params_opt = []
        for component, param, param_fix in zip(self.components, self.params, self.params_fix):
            cfe = estimators.cfes[component.name]  # Get closed-form estimator
            param_init = {**cfe(data, param_fix=param_fix), **param}  # Overwrite random initials with any provided initials
            params_opt.append(param_init)

        for i in range(1, maxiter + 1):
            ll0 = _get_loglikelihood(data, self.components, params_opt, self.params_fix, weights_opt)

            # Expectation
            expts = _get_posterior(data, self.components, params_opt, self.params_fix, weights_opt)
            weights_opt = expts.sum(axis=1) / expts.sum()

            # Maximization
            for component, param_opt, param_fix, expt in zip(self.components, params_opt, self.params_fix, expts):
                mle = estimators.mles[component.name]  # Get MLE function
                opt = mle(data, param_fix=param_fix, expt=expt, initial=param_opt)  # Get updated parameters
                param_opt.update(opt)
            ll = _get_loglikelihood(data, self.components, params_opt, self.params_fix, weights_opt)

            # Print output
            if verbose:
                print(f'Step {i} / {maxiter}')
                print(f'    Log-likelihood: {ll}')
                print(f'    Delta: {ll0-ll}')
                print()

            # Test numerical exception then convergence
            if np.isnan(ll) or np.isinf(ll):
                break
            if abs(ll - ll0) < tol:
                self.converged = True
                if verbose:
                    print(f'Convergence reached with log-likelihood {ll} after {i} steps.')
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
        return _get_loglikelihood(data, self.components, self.params, self.params_fix, self.weights)

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
            model. Shape is (len(data), len(self.components)).
        """
        return _get_posterior(data, self.components, self.params, self.params_fix, self.weights)

    def cdf(self, x, component='sum'):
        """Return cdf evaluated at x.

        Parameters
        ----------
        x: 1-D ndarray
            Values at which to evaluate components of mixture model.
        component: 'sum', 'all', or int
            If 'sum', the cdfs are summed across components. If 'all', the cdf
            of each component is returned as an ndarray with shape (len(x),
            len(self.components)). If component is an int, the cdf of the
            corresponding component is returned.

        Returns
        -------
        ps: ndarray
            cdf evaluated at data.
        """
        # Check arguments
        if (component not in ['sum', 'all']) and (not isinstance(component, int)):
            raise ValueError('component is not "sum", "all", or int')

        if component == 'sum':
            ps = _get_cdfstack(x, self.components, self.params, self.params_fix, self.weights)
            return ps.sum(axis=0)
        elif component == 'all':
            ps = _get_cdfstack(x, self.components, self.params, self.params_fix, self.weights)
            return ps
        else:
            model_params = zip(self.components, self.params, self.params_fix, self.weights)
            component, param, param_fix, weight = list(model_params)[component]
            ps = weight * component.cdf(x, **param_fix, **param)
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
            len(self.components)). If component is an int, the pdf of the
            corresponding component is returned.

        Returns
        -------
        ps: ndarray
            pdf evaluated at data.
        """
        # Check arguments
        if (component not in ['sum', 'all']) and (not isinstance(component, int)):
            raise ValueError('component is not "sum", "all", or int')

        if component == 'sum':
            ps = _get_pdfstack(x, self.components, self.params, self.params_fix, self.weights)
            return ps.sum(axis=0)
        elif component == 'all':
            ps = _get_pdfstack(x, self.components, self.params, self.params_fix, self.weights)
            return ps
        else:
            model_params = zip(self.components, self.params, self.params_fix, self.weights)
            component, param, param_fix, weight = list(model_params)[component]
            ps = weight * component.pdf(x, **param_fix, **param)
            return ps
