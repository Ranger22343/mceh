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


def sem(error):
    weights = 1 / error**2
    return np.sqrt(1 / np.sum(weights))