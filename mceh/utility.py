"""
Last changed date: 2024 Oct 4
"""

from astropy.io import fits
import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
from . import fitting
from scipy import interpolate
import astropy.coordinates as coord
import random
import astropy.units as u
from astropy.table import QTable
import tqdm


def init(*args):
    """Load frequent-used data
    
    Load frequent-used data containing eFEDS('efeds'), HSC('hsc') and 
    random('rd').

    Args:
        *args: The data you want to load. Options are 'efeds', 'hsc' and 'rd'.
    
    Returns:
        The corresponding data.
    """
    return_dict = {}
    for arg in args:
        if arg == 'efeds':
            efeds = QTable.read('data/modified_efeds_ver8.fits')
            return_dict['efeds'] = efeds
        if arg == 'hsc':
            hsc = QTable.read(
                'data/modified_hsc_ver1.fits'
                )
            return_dict['hsc'] = hsc
        if arg == 'rd':
            rd = QTable.read('data/modified_random_ver1.fits')
            return_dict['rd'] = rd
    returnme = [return_dict[arg] for arg in args]
    if len(returnme) == 1:
        returnme = returnme[0]
    return returnme


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


def mcmc_lf_all(flat_chain, log_mass, z, ms_model, area, bins, zero_index,
                progress=True, mode='phi_model_schechter'):
    """Obtain the luminosity functions of every steps from a flat MCMC chain

    The LF will be generated corresponding to the flat chain applying on each
    cluster. The order will be [CH0CL0, CH0CL1, CH0CL2, ..., CH1CL0, ...] where
    CHMCLN means the LF from Mth chain applying on the Nth clusters.

    Args:
        flat_chain (array-like): The flat chain of the MCMC.
        log_mass (array-like): The logrithmal mass of the clusters. Note that
            the units are not contained.
        z (array-like): The redshift of the clusters.
        ms_model (array-like): The model of the characteristic magnitude.
        area (array-like): The on-sky area of the cluters (assuming 0% masking
            fraction).
        bins (array-like): The (full) bins used to fit.
        zero_index (array-like): The indicies deleted from the full bins. It
            exists because some observations are always zero so it is not needed
            to fit it.
        progress (bool): Whether the progress bar is shown.
    
    Returns:
        ndarray: An array of LF corresponding to the chain. The order is
            [chain 0 cluster 0, chain 0 cluster 1, chain 0 cluster 2,...,
             chain 1 cluster 0, chain 1 cluster 1, ...].
    """
    cnum = len(log_mass)
    if mode == 'phi_model_schechter':
        A_chain = flat_chain[:, 0]
        B_chain = flat_chain[:, 1]
        alpha_chain = flat_chain[:, 2]
        dm_chain = flat_chain[:, 3]
        bkg_chain = flat_chain[:, 4:]
        phi_chain = np.array([fitting.phi_model(log_mass, A, B)
                            for A, B in zip(A_chain, B_chain)]).flatten()
        ms_chain = np.array([ms_model + dm for dm in dm_chain]).flatten()
        new_alpha_chain = np.array([np.full(cnum, alpha) 
                                    for alpha in alpha_chain]).flatten()
    elif mode == 'zm_model_schechter':
        A_chain = flat_chain[:, 0]
        B_chain = flat_chain[:, 1]
        alpha0_chain = flat_chain[:, 2]
        dm_chain = flat_chain[:, 3]
        C_chain = flat_chain[:, 4]
        D_chain = flat_chain[:, 5]
        E_chain = flat_chain[:, 6]
        bkg_chain = flat_chain[:, 7:]
        phi_chain = np.array([fitting.phi_model_mz(log_mass, z, A, B, C)
                            for A, B, C in zip(A_chain, B_chain, C_chain)]
                            ).flatten()
        ms_chain = np.array([ms_model + dm for dm in dm_chain]).flatten()
        new_alpha_chain = np.array([
            fitting.alpha_model_mz(log_mass, z, alpha0, D, E) 
            for alpha0, D, E in zip(alpha0_chain, D_chain, E_chain)
            ]).flatten()
    sf_value = []
    # [c0AB0, c1AB0, c2AB0, ..., c0AB1, c1AB1, ..., ...]
    if progress == True:
        rg = tqdm.tqdm(range(len(new_alpha_chain)))
    else:
        rg = range(len(new_alpha_chain))
    for i in rg:
        sf_value.append(fitting.schechter_bins(ms_chain[i],
                                               phi_chain[i],
                                               new_alpha_chain[i],
                                               bins=bins))
    for i in zero_index:
        bkg_chain = np.insert(bkg_chain, i, 0, axis=1)
    bkg_value = np.array([bkg_chain[i] * area[j] for i in range(len(flat_chain)) 
                          for j in range(len(area))])
    lf_value = sf_value + bkg_value
    return lf_value


def mcmc_lf_percentile(flat_chain,
                       log_mass,
                       z,
                       ms_model,
                       area,
                       percentile,
                       bins,
                       zero_index=[]):
    """The percentiles of the LF corresponding to the chain.
    
    Args:
        flat_chain (array-like): The flat chain of the MCMC.
        log_mass (array-like): The logrithmal mass of the clusters. Note that
            the units are not contained.
        z (array-like): The redshift of the clusters.
        ms_model (array-like): The model of the characteristic magnitude.
        area (array-like): The on-sky area of the cluters (assuming 0% masking
            fraction).
        percentile (array-like): The percentiles (%) that should be returned.
        bins (array-like): The (full) bins used to fit.
        zero_index (array-like): The indicies deleted from the full bins. It
            exists because some observations are always zero so it is not needed
            to fit it.
    
    Returns:
        ndarray: The LF values corresponds to the percentiles of the chain.
            The first element corresponds to the first values in `percentile`,
            etc.

    """
    lf_value = mcmc_lf_all(flat_chain, log_mass, z, ms_model, area, bins, 
                           zero_index)
    returnme = np.percentile(lf_value, percentile, axis=0)
    return returnme


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