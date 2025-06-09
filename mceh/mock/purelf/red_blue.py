from ...model import all_redblue as fitting
from ... import find_luminosity as fl
from ... import utility as ut
import numpy as np


def generate_mock_mag(phi, alpha, ms, method='inverse transform sampling'):
    """
    Generate magnitude of a mock galaxy cluster following the LF.

    Args:
        phi(float): The normalization of the luminosity function.
        alpha(float): The slope of the luminosity function.
        ms(float): The characteristic magnitude of the luminosity function.
        method(str): The method to use for generating the mock magnitude.
            'inverse transform sampling' or 'rejection sampling'.
            Defaults to 'inverse transform sampling'.
    Returns:
        np.ndarray: The generated magnitudes.
    """
    # Generate a mock luminosity function using the red-blue method
    n = round(fitting.schechter_bins(phi, alpha, ms, fitting.BINS).sum())
    x_min, x_max = min(fitting.BINS), max(fitting.BINS)

    @np.vectorize
    def cdf(mag, x_min=x_min):
        returnme = fitting.schechter_bins(phi, alpha, ms,
                                          np.array([x_min, mag]))[0]

        return returnme

    def pdf(mag):
        return fitting.schechter(mag, ms, phi, alpha) / cdf(x_max, x_min=x_min)

    if method == 'inverse transform sampling':
        x = ut.inverse_transform_sampling(n, x_min, x_max, cdf)
    elif method == 'rejection sampling':
        x = ut.rejection_sampling(n, x_min, x_max, pdf)
    return x


