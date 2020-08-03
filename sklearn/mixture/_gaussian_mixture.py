"""Gaussian Mixture Model."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from scipy import linalg

from ._base import BaseMixture, _check_shape
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.validation import _deprecate_positional_args


###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=covariance_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type],
                 '%s precision' % covariance_type)

    _check_precisions = {'full': _check_precisions_full,
                         'tied': _check_precision_matrix,
                         'diag': _check_precision_positivity,
                         'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[::len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk,
                                               means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {"full": _estimate_gaussian_covariances_full,
                   "tied": _estimate_gaussian_covariances_tied,
                   "diag": _estimate_gaussian_covariances_diag,
                   "spherical": _estimate_gaussian_covariances_spherical
                   }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
    @_deprecate_positional_args
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

class GuassianMixtureMS(BaseCluster):
    """
    Automatic Gaussian Mixture Model (GMM) model selection.

    Clustering algorithm using a hierarchical agglomerative clustering then Gaussian
    mixtured model (GMM) fitting. Different combinations of agglomeration, GMM, and 
    cluster numbers are used and the clustering with the best selection
    criterion (bic/aic) is chosen.

    Parameters
    ----------
    min_components : int, default=2.
        The minimum number of mixture components to consider (unless
        max_components=None, in which case this is the maximum number of
        components to consider). If max_components is not None, min_components
        must be less than or equal to max_components.
        If label_init is given, min_components must match number of unique labels
        in label_init.

    max_components : int or None, default=10.
        The maximum number of mixture components to consider. Must be greater
        than or equal to min_components.
        If label_init is given, min_components must match number of unique labels
        in label_init.

    affinity : {'euclidean','manhattan','cosine','none', 'all' (default)}, optional
        String or list/array describing the type of affinities to use in agglomeration.
        If a string, it must be one of:

        - 'euclidean'
            L2 norm
        - 'manhattan'
            L1 norm
        - 'cosine'
            cosine similarity
        - 'none'
            no agglomeration - GMM is initialized with k-means
        - 'all'
            considers all affinities in ['euclidean','manhattan','cosine','none']

        If a list/array, it must be a list/array of strings containing only
        'euclidean', 'manhattan', 'cosine', and/or 'none'.

        Note that cosine similarity can only work when all of the rows are not the zero vector.
        If the input matrix has a zero row, cosine similarity will be skipped and a warning will
        be thrown.

    linkage : {'ward','complete','average','single', 'all' (default)}, optional
        String or list/array describing the type of linkages to use in agglomeration.
        If a string, it must be one of:

        - 'ward'
            ward's clustering, can only be used with euclidean affinity
        - 'complete'
            complete linkage
        - 'average'
            average linkage
        - 'single'
            single linkage
        - 'all'
            considers all linkages in ['ward','complete','average','single']

        If a list/array, it must be a list/array of strings containing only
        'ward', 'complete', 'average', and/or 'single'.

    covariance_type : {'full', 'tied', 'diag', 'spherical', 'all' (default)} , optional
        String or list/array describing the type of covariance parameters to use.
        If a string, it must be one of:

        - 'full'
            each component has its own general covariance matrix
        - 'tied'
            all components share the same general covariance matrix
        - 'diag'
            each component has its own diagonal covariance matrix
        - 'spherical'
            each component has its own single variance
        - 'all'
            considers all covariance structures in ['spherical', 'diag', 'tied', 'full']

        If a list/array, it must be a list/array of strings containing only
        'spherical', 'tied', 'diag', and/or 'spherical'.

    random_state : int, RandomState instance or None, optional (default=None)
        There is randomness in k-means initialization of 
        :class:`sklearn.mixture.GaussianMixture`. This parameter is passed to 
        :class:`~sklearn.mixture.GaussianMixture` to control the random state.
        If int, random_state is used as the random number generator seed;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.

    label_init : array-like, shape (n_samples,), optional (default=None)
        List of labels for samples if available. Used to initialize the model.
        If provided, min_components and max_components must match the number of 
        unique labels given here.

    max_iter : int, optional (default = 100).
        The maximum number of EM iterations to perform.

    selection_criteria : str {"bic" or "aic"}, optional, (default="bic")
        select the best model based on Bayesian Information Criterion (bic) or 
        Aikake Information Criterion (aic)

    verbose : int, optional (default = 0)
        Enable verbose output. If 1 then it prints the current initialization and each 
        iteration step. If greater than 1 then it prints also the log probability and 
        the time needed for each step.

    max_agglom_size : int or None, optional (default = 2000)
        The maximum number of datapoints on which to do agglomerative clustering as the 
        initialization to GMM. If the number of datapoints is larger than this value, 
        a random subset of the data is used for agglomerative initialization. If None,
        all data is used for agglomerative clustering for initialization.

    n_jobs : int or None, optional (default = None)
        The number of jobs to use for the computation. This works by computing each of
        the initialization runs in parallel. None means 1 unless in a 
        ``joblib.parallel_backend context``. -1 means using all processors. 
        See https://scikit-learn.org/stable/glossary.html#term-n-jobs for more details.

    Attributes
    ----------
    results_ : pandas.DataFrame
        Contains exhaustive information about all the clustering runs.
        Columns are:

        'model' : GaussianMixture object
            GMM clustering fit to the data
        'bic/aic' : float
            Bayesian Information Criterion
        'ari' : float or nan
            Adjusted Rand Index between GMM classification, and true classification,
            nan if y is not given
        'n_components' : int
            number of clusters
        'affinity' : {'euclidean','manhattan','cosine','none'}
            affinity used in Agglomerative Clustering
        'linkage' : {'ward','complete','average','single'}
            linkage used in Agglomerative Clustering
        'covariance_type' : {'full', 'tied', 'diag', 'spherical'}
            covariance type used in GMM
        'reg_covar' : float
            regularization used in GMM

    criter_ : the best (lowest) Bayesian Information Criterion

    n_components_ : int
        number of clusters in the model with the best bic/aic

    covariance_type_ : str
        covariance type in the model with the best bic/aic

    affinity_ : str
        affinity used in the model with the best bic/aic

    linkage_ : str
        linkage used in the model with the best bic/aic

    reg_covar_ : float
        regularization used in the model with the best bic/aic

    ari_ : float
        ARI from the model with the best bic/aic, nan if no y is given

    model_ : :class:`sklearn.mixture.GaussianMixture`
        object with the best bic/aic

    See Also
    --------
    graspy.cluster.GaussianCluster
    graspy.cluster.KMeansCluster

    Notes
    -----
    This algorithm was strongly inspired by mclust, a clustering package in R

    References
    ----------
    .. [1] Jeffrey D. Banfield and Adrian E. Raftery. Model-based gaussian and
       non-gaussian clustering. Biometrics, 49:803–821, 1993.

    .. [2] Abhijit Dasgupta and Adrian E. Raftery. Detecting features in spatial point
       processes with clutter via model-based clustering. Journal of the American
       Statistical Association, 93(441):294–302, 1998.
    """

    def __init__(
        self,
        min_components=2,
        max_components=10,
        affinity="all",
        linkage="all",
        covariance_type="all",
        random_state=None,
        label_init=None,
        max_iter=100,
        verbose=0,
        selection_criteria="bic",
        max_agglom_size=2000,
        n_jobs=None,
    ):
        if isinstance(min_components, int):
            if min_components <= 0:
                msg = "min_components must be >= 1."
                raise ValueError(msg)
        else:
            msg = "min_components must be an integer, not {}.".format(
                type(min_components)
            )
            raise TypeError(msg)

        if isinstance(max_components, int):
            if max_components <= 0:
                msg = "max_components must be >= 1 or None."
                raise ValueError(msg)
            elif min_components > max_components:
                msg = "min_components must be less than or equal to max_components."
                raise ValueError(msg)
        elif max_components is not None:
            msg = "max_components must be an integer or None, not {}.".format(
                type(max_components)
            )
            raise TypeError(msg)

        if isinstance(affinity, (np.ndarray, list)):
            affinity = np.unique(affinity)
        elif isinstance(affinity, str):
            if affinity == "all":
                affinity = ["euclidean", "manhattan", "cosine", "none"]
            else:
                affinity = [affinity]
        else:
            msg = "affinity must be a numpy array, a list, or "
            msg += "string, not {}".format(type(affinity))
            raise TypeError(msg)

        for aff in affinity:
            if aff not in ["euclidean", "manhattan", "cosine", "none"]:
                msg = (
                    "affinity must be one of "
                    + '["euclidean","manhattan","cosine","none"]'
                )
                msg += " not {}".format(aff)
                raise ValueError(msg)

        if ("ward" in linkage) and not ("euclidean" in affinity):
            msg = (
                'if "ward" is a linkage option, '
                + '"euclidean" must be an affinity option'
            )
            raise ValueError(msg)

        if isinstance(linkage, (np.ndarray, list)):
            linkage = np.unique(linkage)
        elif isinstance(linkage, str):
            if linkage == "all":
                linkage = ["ward", "complete", "average", "single"]
            else:
                linkage = [linkage]
        else:
            msg = "linkage must be a numpy array, a list, or "
            msg += "string, not {}".format(type(linkage))
            raise TypeError(msg)

        for link in linkage:
            if link not in ["ward", "complete", "average", "single"]:
                msg = (
                    "linkage must be one of "
                    + '["ward", "complete", "average", "single"]'
                )
                msg += " not {}".format(link)
                raise ValueError(msg)

        if isinstance(covariance_type, (np.ndarray, list)):
            covariance_type = np.unique(covariance_type)
        elif isinstance(covariance_type, str):
            if covariance_type == "all":
                covariance_type = ["spherical", "diag", "tied", "full"]
            else:
                covariance_type = [covariance_type]
        else:
            msg = "covariance_type must be a numpy array, a list, or "
            msg += "string, not {}".format(type(covariance_type))
            raise TypeError(msg)

        for cov in covariance_type:
            if cov not in ["spherical", "diag", "tied", "full"]:
                msg = (
                    "covariance structure must be one of "
                    + '["spherical", "diag", "tied", "full"]'
                )
                msg += " not {}".format(cov)
                raise ValueError(msg)

        new_covariance_type = []
        for cov in ["spherical", "diag", "tied", "full"]:
            if cov in covariance_type:
                new_covariance_type.append(cov)

        if isinstance(label_init, list):
            label_init = np.array(label_init)
        elif isinstance(label_init, np.ndarray):
            if label_init.ndim > 2 or (
                label_init.ndim == 2 and 1 not in label_init.shape
            ):
                msg = "label_init must be a one dimension array."
                raise TypeError(msg)
        elif label_init is not None:
            msg = "label_init must be a 1-D numpy array, a list, or None,"
            msg += "not {}".format(type(label_init))
            raise TypeError(msg)

        if selection_criteria not in ["aic", "bic"]:
            msg = "selection_criteria must be one of " + '["aic, "bic"]'
            msg += " not {}".format(selection_criteria)
            raise ValueError(msg)

        # Adjust elements in label_init to range(n_components of label_init)
        if label_init is not None:
            uni_label_init = np.unique(label_init)
            n_components_init = np.size(uni_label_init)

            if (
                min_components != n_components_init
                or max_components != n_components_init
            ):
                msg = "min_components and max_components must equal "
                msg += " the number of init labels: {}".format(n_components_init)
                raise ValueError(msg)

            labels_init = np.copy(label_init)
            for i in range(n_components_init):
                labels_init[np.argwhere(label_init == uni_label_init[i])] = i
            labels_init = labels_init.astype(int)
        else:
            labels_init = None

        if not isinstance(max_agglom_size, int) and max_agglom_size is not None:
            raise TypeError("`max_agglom_size` must be an int or None")
        if max_agglom_size is not None and max_agglom_size < 2:
            raise ValueError("Must use at least 2 points for `max_agglom_size`")

        self.min_components = min_components
        self.max_components = max_components
        self.affinity = affinity
        self.linkage = linkage
        self.covariance_type = new_covariance_type
        self.random_state = random_state
        self.label_init = labels_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.selection_criteria = selection_criteria
        self.max_agglom_size = max_agglom_size
        self.n_jobs = n_jobs

    # ignoring warning here because if convergence is not reached, the regularization
    # is automatically increased
    @ignore_warnings(category=ConvergenceWarning)
    def _fit_cluster(self, X, y, params):
        label_init = self.label_init
        if label_init is not None:
            onehot = _labels_to_onehot(label_init)
            weights_init, means_init, precisions_init = _onehot_to_initial_params(
                X, onehot, params[1]["covariance_type"]
            )
            gm_params = params[1]
            gm_params["weights_init"] = weights_init
            gm_params["means_init"] = means_init
            gm_params["precisions_init"] = precisions_init
        elif params[0]["affinity"] != "none":
            agg = AgglomerativeClustering(**params[0])
            n = X.shape[0]

            if self.max_agglom_size is None or n <= self.max_agglom_size:
                X_subset = X
            else:  # if dataset is huge, agglomerate a subset
                subset_idxs = np.random.choice(np.arange(0, n), self.max_agglom_size)
                X_subset = X[subset_idxs, :]
            agg_clustering = agg.fit_predict(X_subset)
            onehot = _labels_to_onehot(agg_clustering)
            weights_init, means_init, precisions_init = _onehot_to_initial_params(
                X_subset, onehot, params[1]["covariance_type"]
            )
            gm_params = params[1]
            gm_params["weights_init"] = weights_init
            gm_params["means_init"] = means_init
            gm_params["precisions_init"] = precisions_init
        else:
            gm_params = params[1]
            gm_params["init_params"] = "kmeans"
        gm_params["reg_covar"] = 0
        gm_params["max_iter"] = self.max_iter

        criter = np.inf  # if none of the iterations converge, bic/aic is set to inf
        # below is the regularization scheme
        while gm_params["reg_covar"] <= 1 and criter == np.inf:
            model = GaussianMixture(**gm_params)
            try:
                model.fit(X)
                predictions = model.predict(X)
                counts = [
                    sum(predictions == i) for i in range(gm_params["n_components"])
                ]
                # singleton clusters not allowed
                assert not any([count <= 1 for count in counts])

            except ValueError:
                gm_params["reg_covar"] = _increase_reg(gm_params["reg_covar"])
                continue
            except AssertionError:
                gm_params["reg_covar"] = _increase_reg(gm_params["reg_covar"])
                continue
            # if the code gets here, then the model has been fit with no errors or
            # singleton clusters
            if self.selection_criteria == "bic":
                criter = model.bic(X)
            else:
                criter = model.aic(X)
            break

        if y is not None:
            self.predictions = model.predict(X)
            ari = adjusted_rand_score(y, self.predictions)
        else:
            ari = float("nan")
        results = {
            "model": model,
            "bic/aic": criter,
            "ari": ari,
            "n_components": gm_params["n_components"],
            "affinity": params[0]["affinity"],
            "linkage": params[0]["linkage"],
            "covariance_type": gm_params["covariance_type"],
            "reg_covar": gm_params["reg_covar"],
        }
        return results

    def fit(self, X, y=None):
        """
        Fits gaussian mixture model to the data.
        Initialize with agglomerative clustering then
        estimate model parameters with EM algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_samples,), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        # Deal with number of clusters
        if self.max_components is None:
            lower_ncomponents = 1
            upper_ncomponents = self.min_components
        else:
            lower_ncomponents = self.min_components
            upper_ncomponents = self.max_components

        if upper_ncomponents > X.shape[0]:
            if self.max_components is None:
                msg = "if max_components is None then min_components must be >= "
                msg += "n_samples, but min_components = {}, n_samples = {}".format(
                    upper_ncomponents, X.shape[0]
                )
            else:
                msg = "max_components must be <= n_samples, but max_components = "
                msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)
        elif lower_ncomponents > X.shape[0]:
            msg = "min_components must be <= n_samples, but min_components = "
            msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)
        # check if X contains the 0 vector
        if np.any(~X.any(axis=1)) and ("cosine" in self.affinity):
            if isinstance(self.affinity, np.ndarray):
                self.affinity = np.delete(
                    self.affinity, np.argwhere(self.affinity == "cosine")
                )
            if isinstance(self.affinity, list):
                self.affinity.remove("cosine")
            warnings.warn("X contains a zero vector, will not run cosine affinity.")

        label_init = self.label_init
        if label_init is not None:
            if label_init.size != X.shape[0]:
                msg = "n_samples must be the same as the length of label_init"
                raise ValueError(msg)

        param_grid = dict(
            affinity=self.affinity,
            linkage=self.linkage,
            covariance_type=self.covariance_type,
            n_components=range(lower_ncomponents, upper_ncomponents + 1),
            random_state=[self.random_state],
        )

        param_grid = list(ParameterGrid(param_grid))
        param_grid = _process_paramgrid(param_grid)

        def _fit_for_data(p):
            return self._fit_cluster(X, y, p)

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_for_data)(p) for p in param_grid
        )

        results = pd.DataFrame(results)

        self.results_ = results
        # Get the best cov type and its index within the dataframe
        best_idx = results["bic/aic"].idxmin()

        self.criter_ = results.loc[best_idx, "bic/aic"]
        self.n_components_ = results.loc[best_idx, "n_components"]
        self.covariance_type_ = results.loc[best_idx, "covariance_type"]
        self.affinity_ = results.loc[best_idx, "affinity"]
        self.linkage_ = results.loc[best_idx, "linkage"]
        self.reg_covar_ = results.loc[best_idx, "reg_covar"]
        self.ari_ = results.loc[best_idx, "ari"]
        self.model_ = results.loc[best_idx, "model"]

        return self


def _increase_reg(reg):
    """
    Increase regularization factor by factor of 10.

    Parameters
    ----------
    reg: float
        Current regularization factor.

    Returns
    -------
    reg : float
        Increased regularization
    """
    if reg == 0:
        reg = 1e-06
    else:
        reg = reg * 10
    return reg


def _onehot_to_initial_params(X, onehot, cov_type):
    """
    Computes cluster weights, cluster means and cluster precisions from
    a given clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.
    onehot : ndarray, shape (n_samples, n_clusters)
        Each row has a 1 indicating cluster membership, other entries are 0.
    cov_type : {'full', 'tied', 'diag', 'spherical'}
        Covariance type for Gaussian mixture model
    """
    n = X.shape[0]
    weights, means, covariances = _estimate_gaussian_parameters(
        X, onehot, 1e-06, cov_type
    )
    weights /= n

    precisions_cholesky_ = _compute_precision_cholesky(covariances, cov_type)

    if cov_type == "tied":
        c = precisions_cholesky_
        precisions = np.dot(c, c.T)
    elif cov_type == "diag":
        precisions = precisions_cholesky_
    else:
        precisions = [np.dot(c, c.T) for c in precisions_cholesky_]

    return weights, means, precisions


def _labels_to_onehot(labels):
    """
    Converts labels to one-hot format.

    Parameters
    ----------
    labels : ndarray, shape (n_samples,)
        Cluster labels

    Returns
    -------
    onehot : ndarray, shape (n_samples, n_clusters)
        Each row has a single one indicating cluster membership.
        All other entries are zero.
    """
    n = len(labels)
    k = max(labels) + 1
    onehot = np.zeros([n, k])
    onehot[np.arange(n), labels] = 1
    return onehot


def _process_paramgrid(paramgrid):
    """
        Removes combinations of affinity and linkage that are not possible.

        Parameters
        ----------
        paramgrid : list of dicts
            Each dict has the keys 'affinity', 'covariance_type', 'linkage',
            'n_components', and 'random_state'

        Returns
        -------
        paramgrid_processed : list pairs of dicts
            For each pair, the first dict are the options for AgglomerativeClustering.
            The second dict include the options for GaussianMixture.
        """
    paramgrid_processed = []

    for params in paramgrid:
        if (
            params["affinity"] == "none"
            and params["linkage"] != paramgrid[0]["linkage"]
        ):
            continue
        elif (
            params["linkage"] == "ward"
            and params["affinity"] != "euclidean"
            and params["affinity"] != "none"
        ):
            continue
        else:

            gm_keys = ["covariance_type", "n_components", "random_state"]
            gm_params = {key: params[key] for key in gm_keys}

            ag_keys = ["affinity", "linkage"]
            ag_params = {key: params[key] for key in ag_keys}
            ag_params["n_clusters"] = params["n_components"]

            paramgrid_processed.append([ag_params, gm_params])

    return paramgrid_processed
