"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from ._gaussian_mixture import GaussianMixture, GaussianMixtureCV
from ._bayesian_mixture import BayesianGaussianMixture


__all__ = ['GaussianMixture',
			'GaussianMixtureCV',
           'BayesianGaussianMixture']
