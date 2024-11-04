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


def init(*args):
    returnme = []
    for arg in args:
        if arg == 'efeds':
            efeds = QTable.read('data/modified_efeds_ver6.fits')
            returnme.append(efeds)
        if arg == 'hsc':
            hsc = QTable.read(
                'data/modified_hsc_ver1.fits'
                )
            returnme.append(hsc)
        if arg == 'rd':
            rd = QTable.read('data/modified_random_ver1.fits')
            returnme.append(rd)
        if arg == 'bkg_all':
            with open('data/bkg_result2.pickle', 'rb') as f:
                bkg_all = pickle.load(f)
            returnme.append(bkg_all)
    if len(returnme) == 1:
        return returnme[0]
    return returnme


def ordinal(num):
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
    samples = sampler.get_chain()
    fig, axes = plt.subplots(len(show_index), figsize=(10, 7), sharex=True)
    ndim = len(show_index)
    for i in range(len(show_index)):
        ax = axes[i]
        ax.plot(samples[:, :, show_index[i]], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        if label is not None:
            ax.set_ylabel(label[i])
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
    with open(file_path, 'rb') as f:
        returnme = pickle.load(f)
    return returnme


def pickle_dump(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def bound_to_bin(bound):
    returnme = bound[:-1]
    return returnme


def mcmc_lf_percentile(flat_chain,
                       log_mass,
                       ms_model,
                       area,
                       percentile,
                       bins,
                       zero_index=[]):
    # percentile of the Schethcer functions.
    # for single cluster
    A_chain = flat_chain[:, 0]
    B_chain = flat_chain[:, 1]
    alpha_chain = flat_chain[:, 2]
    dm_chain = flat_chain[:, 3]
    bkg_chain = flat_chain[:, 4:]
    phi_chain = fitting.phi_model(log_mass, A_chain, B_chain)
    ms_chain = ms_model + dm_chain
    # xx_value = [set1, set2, ...]
    sf_value = np.array([
        fitting.schechter_bins(ms_chain[i],
                               phi_chain[i],
                               alpha_chain[i],
                               bins=bins) for i in range(len(flat_chain))
    ])
    for i in zero_index:
        bkg_chain = np.insert(bkg_chain, i, 0, axis=1)
    if bins.tolist() != fitting.bins.tolist():
        bkg_chain = [
            change_bins(bkg_chain[i], fitting.bins, bins)
            for i in range(len(bkg_chain))
        ]
    bkg_value = [bkg_chain[i] * area for i in range(len(flat_chain))]
    lf_value = sf_value + bkg_value
    returnme = np.percentile(lf_value, percentile, axis=0)
    return returnme


def mcmc_lf_all(flat_chain, log_mass, ms_model, area, bins, zero_index=[]):
    # percentile of the Schethcer functions.
    # for single cluster
    A_chain = flat_chain[:, 0]
    B_chain = flat_chain[:, 1]
    alpha_chain = flat_chain[:, 2]
    dm_chain = flat_chain[:, 3]
    bkg_chain = flat_chain[:, 4:]
    phi_chain = fitting.phi_model(log_mass, A_chain, B_chain)
    ms_chain = ms_model + dm_chain
    # xx_value = [set1, set2, ...]
    sf_value = np.array([
        fitting.schechter_bins(ms_chain[i],
                               phi_chain[i],
                               alpha_chain[i],
                               bins=bins) for i in range(len(flat_chain))
    ])
    for i in zero_index:
        bkg_chain = np.insert(bkg_chain, i, 0, axis=1)
    if bins.tolist() != fitting.bins.tolist():
        bkg_chain = [
            change_bins(bkg_chain[i], fitting.bins, bins)
            for i in range(len(bkg_chain))
        ]
    bkg_value = [bkg_chain[i] * area for i in range(len(flat_chain))]
    lf_value = sf_value + bkg_value
    return lf_value


def obs_alllf_error_bound(obs_alllf):
    # every bin has its own upper/lower error bounds
    # for single cluster
    error = obs_alllf**0.5
    upper, lower = obs_alllf + error, obs_alllf - error
    return upper, lower


def change_bins(y, old_bin, new_bin):
    # Make sure that the bin width of the old/new one is the same.
    old_bin_c = (old_bin[:-1] + old_bin[1:]) / 2
    new_bin_c = (new_bin[:-1] + new_bin[1:]) / 2
    y_func = interpolate.interp1d(old_bin_c, y)
    return y_func(new_bin_c)


def cut_range(num, bin_num):
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