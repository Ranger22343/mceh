"""
Last changed date: 2024 Oct 4
"""

from astropy.io import fits
import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import interpolate
import astropy.coordinates as coord
import random
import astropy.units as u
from astropy.table import QTable
import tqdm
from scipy import special


def ordinal(num):
    """Turn an integer into an ordinal string.
    
    Args:
        num (int): The number you want to turn into an ordinal string.

    Returns:
        str: The resulting ordinal string.
    """
    num_str = str(int(num))
    if num_str[-1] == '1':
        returnme = num_str + 'st'
    elif num_str[-1] == '2':
        returnme = num_str + 'nd'
    elif num_str[-1] == '3':
        returnme = num_str + 'rd'
    else:
        returnme = num_str + 'th'
    if len(num_str) > 1 and num_str[-2] == '1':
        returnme = num_str + 'th'
    return returnme


def plot_step(sampler,
              show_index,
              label=['A', 'B', r'$\alpha$', r'$\Delta m$'],
              title=None):
    """Plot the steps of the MCMC
    
    Args:
        sampler (EnsembleSampler): The MCMC sampler.
        show_index (int list): A list of paramter indicies you want to plot.
        label (str list): A list of parameter names shown on the plot.
        title (str): Title of the plot.
    """
    samples = sampler.get_chain()
    fig, axes = plt.subplots(len(show_index), figsize=(10, 7), sharex=True)
    ndim = len(show_index)
    for i in range(len(show_index)):
        ax = axes[i]
        ax.plot(samples[:, :, show_index[i]], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        if label is not None:
            ax.set_ylabel(label[i], fontsize=16)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    if title is not None:
        plt.suptitle(title)
    return fig, axes


def printt(*args):
    # Print with time at the front. Also, flush = True.
    t = time.localtime(time.time())
    tstr = f"[{time.strftime('%m/%d %H:%M:%S',t)}]"
    print(tstr, *args, flush=True)


def pickle_load(file_path):
    # Load a pickle file.
    with open(file_path, 'rb') as f:
        returnme = pickle.load(f)
    return returnme


def pickle_dump(data, file_path):
    # Save a pickle file.
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def change_bins(y, old_bin, new_bin):
    # Make sure that the bin width of the old/new one is the same.
    old_bin_c = (old_bin[:-1] + old_bin[1:]) / 2
    new_bin_c = (new_bin[:-1] + new_bin[1:]) / 2
    y_func = interpolate.interp1d(old_bin_c, y)
    return y_func(new_bin_c)


def cut_range(num, bin_num):
    """Evenly distribute range(`num`) in given bins.

    range(`num`) will be created and numbers in it will be evenly distributed
    in `bin_num` bins and these bins will be returned. 

    Args:
        num (int): Length of the numbers (start at 0).
        bin_num (int): Number of bins to ditribute the numbers.
    Returns:
        list: Every element represents the numbers in this bin. 
            Ex: [[0, 1, 2], [3, 4, 5], [6, 7]]
    """
    returnme = [[] for i in range(bin_num)]
    width = num // bin_num
    redundant_num = num - bin_num * width
    finish_num = 0
    for i in range(bin_num):
        if redundant_num != 0:
            returnme[i] = list(range(finish_num, finish_num + width + 1))
            redundant_num -= 1
            finish_num += width + 1
        else:
            returnme[i] = list(range(finish_num, finish_num + width))
            finish_num += width
    return returnme


def area(r):
    # The on-sky circular area by a given radius.
    return 2 * np.pi * (1 - np.cos(r)) * u.rad**2


def is_used2bins(original_bins, is_used):
    # Give the right bins by `is_used` list offered by the `get_sampler()`.
    is_bins = np.append(is_used, False)
    for i in range(len(is_used)):
        if is_used[i] == True:
            is_bins[i] = True
            is_bins[i + 1] = True
    return original_bins[is_bins]


def sem(error: float) -> float:
    weights = 1 / error**2
    return np.sqrt(1 / np.sum(weights))


def is_between(x, x_min, x_max):
    """Check if x is between x_min and x_max.

    Args:
        x (float): The value to check.
        x_min (float): The minimum value.
        x_max (float): The maximum value.

    Returns:
        bool: True if x is between x_min and x_max, False otherwise.
    """
    return (x >= x_min) & (x <= x_max)


def rejection_sampling(n, x_min, x_max, pdf, n_samples=None):
    """Rejection sampling from a given PDF.

    Args:
        n (int): The number of samples to generate.
        x_min (float): The minimum value of the range.
        x_max (float): The maximum value of the range.
        pdf (function): The probability density function to sample from.
                        It must be normalized.
        n_samples (int): The number of samples to draw.

    Returns:
        np.ndarray: The generated samples.
    """
    samples = []
    if n_samples is None:
        n_samples = n // 10
    while len(samples) < n:
        x = np.random.uniform(x_min, x_max, n_samples)
        y = np.random.uniform(0, 1, n_samples)
        mask = y < pdf(x)
        samples.extend(x[mask])
    return np.array(samples[:n])


def inverse_transform_sampling(n, x_min, x_max, cdf,
                               y_min=None,
                               y_max=None,
                               inv_cdf=None,
                               bin_num=10000):
    """Inverse transform sampling from a given CDF.

    Args:
        n (int): The number of samples to generate.
        x_min (float): The minimum value of the range.
        x_max (float): The maximum value of the range.
        cdf (function): The cumulative distribution function to sample from.
                        It can be normalized or not.
        y_min (float, optional): The minimum value of the CDF. Defaults to None.
        y_max (float, optional): The maximum value of the CDF. Defaults to None.
        inv_cdf (function, optional): The inverse CDF. Defaults to None.
        bin_num (int, optional): Number of bins for interpolation. 
                                 Defaults to 10000.
    Returns:
        np.ndarray: The generated samples.
    """
    if inv_cdf is None:
        bins = np.linspace(x_min, x_max, bin_num)
        inv_cdf = interpolate.interp1d(cdf(bins), bins,
                                       bounds_error=False,
                                       fill_value=(x_min, x_max))
    if y_min is None:
        y_min = cdf(x_min)
    if y_max is None:
        y_max = cdf(x_max)
    y = np.random.uniform(y_min, y_max, n)
    x = inv_cdf(y)
    return x


def schechter(m: float, m_s: float, phi_s: float, alpha: float):
    return 0.4 * np.log(10) * phi_s * (10**(0.4 * (m_s - m)))**(
        alpha + 1) * np.exp(-10**(0.4 * (m_s - m)))


def gammainc(s: float, x: float, eps=1e-6) -> float:
    if abs(s) < eps:
        number = 0
        s = eps
    elif s > 0:
        number = 0
    else:
        if abs(s - round(s)) < eps:
            s = round(s) + eps
        number = int(-s) + 1
        s = s + number
    # ex -1 - 1e-8 -> - 1e-6 int=-1
    # ez -1 + 1e-8 -> 1e-6 int=0
    returnme = special.gammainc(s, x) * special.gamma(s)
    for i in range(number):
        returnme = (returnme + x**(s - 1) * np.exp(-x)) / (s - 1)
        s -= 1
    return returnme


def schechter_bins(phi_s, alpha, m_s, bins):
    s = alpha + 1
    x = 10**(0.4 * (m_s - bins))
    bound_values = phi_s * gammainc(s, x)
    returnme = bound_values[:-1] - bound_values[1:]
    return returnme


def group_indicies_by_bins(arr, bins, no_empty=True):
    """Group indices of `arr` by the bins.

    Args:
        arr (list or np.ndarray): The array to group.
        bins (np.ndarray): The bins to group by.
        no_empty (bool, optional): If True, empty groups will be removed. 
                                   Defaults to True.

    Returns:
        list: A list of lists, where each inner list contains the indices 
              of `arr` that fall within the corresponding bin.
    """
    returnme = []
    for i in range(len(bins) - 1):
        mask = (arr >= bins[i]) & (arr < bins[i + 1])
        if i == len(bins) - 2:
            mask = (arr > bins[i]) & (arr <= bins[i + 1])
        if no_empty and not np.any(mask):
            continue
        returnme.append(np.where(mask)[0].tolist())
    return returnme
    