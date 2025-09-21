import numpy as np
import emcee
from scipy import integrate
import multiprocessing
import functools
from scipy import interpolate
import astropy.units as u
import os
from astropy.table import QTable
from astropy.io import fits
from scipy import stats
from scipy import special
from .. import utility as ut

"""
Quick note: This model fits every target clusters in one set of parameters. 
Red-blue separation is used.
"""
os.environ["OMP_NUM_THREADS"] = "1"
multiprocessing.set_start_method('fork', force=True)
BINS = np.arange(10, 30.1, 0.2)  # Boundaries of bins
MBINS = BINS[:-1] / 2 + BINS[1:] / 2
DIFF_BINS = np.arange(-2, 2.1, 0.2)
DIFF_MBINS = (DIFF_BINS[:-1] + DIFF_BINS[1:]) / 2
ZBINS = np.hstack([np.linspace(0.10, 0.35, 7)[:-1],    # g-r v.s. r or g-i v.s. i
                   np.linspace(0.35, 0.75, 6)[:-1],    # r-i v.s. i or r-z v.s. z
                   np.linspace(0.75, 1.31, 4),         # i-z v.s. z
                   #0.10, 0.225, 0.35,                 # g-r v.s. r or g-i v.s. i
                   #0.480, 0.60, 0.75,                 # r-i v.s. i or r-z v.s. z
                   #1.000, 1.31,                       # i-z v.s. z
                   ])
# Since the last group divided by ZBINS has only 1 cluster, I removed the second
# last value so it combine to the second last group.
ZBINS2 = np.delete(ZBINS, -2)
ARG_NUM = 3
LABELS = [r'$\phi_0$', r'$\beta_\phi$', r'$\alpha_0$',
          r'$\Delta m_0$', r'$\gamma_\phi$', r'$\beta_\alpha$',
          r'$\gamma_\alpha$', r'$\beta_{m}$', r'$\gamma_{m}$']
BAND_NAME = np.array(['g', 'r', 'i', 'z', 'y'])
PIV_LOG_MASS = 14  # 10^14 M_sun/h


def init(*args):
    """Load frequent-used data
    
    Load frequent-used data containing eFEDS('efeds'), HSC('hsc') and 
    random('rd') and so on.

    Args:
        *args: The data you want to load. Options are 'efeds', 'hsc' and 'rd'.
    
    Returns:
        The corresponding data.
    """
    return_dict = {}
    for arg in args:
        if arg == 'efeds':
            efeds = QTable.read('data/modified_efeds_ver10.fits')
            return_dict['efeds'] = efeds
        if arg == 'hsc':
            hsc = QTable.read('data/modified_hsc_ver5.fits')
            return_dict['hsc'] = hsc
        if arg == 'rd':
            rd = QTable.read('data/modified_random_ver3.fits')
            return_dict['rd'] = rd
        if arg == 'zmbins':
            zmbins = ut.pickle_load('data/zmbins20241108.pickle')
            return_dict['zmbins'] = zmbins
        if arg == 'sr_efeds':
            efeds = QTable.read('data/modified_efeds_ver10.fits')
            efeds = efeds[efeds['low_cont_flag']
                          & (efeds['unmasked_fraction'] > 0.6)]
            return_dict['sr_efeds'] = efeds
        if arg == 'rd_result':
            rd_result = ut.pickle_load('data/20250813bkg_result.pickle')
            return_dict['rd_result'] = rd_result
        if arg == 'rs_rd_result':
            rs_rd_result = ut.pickle_load('result/20250319redblue_bkg.pickle')
            return_dict['rs_rd_result'] = rs_rd_result
        if arg == 'rs_data':
            rs_data = fits.getdata('data/rs.fits', ext=-1)
            return_dict['rs_data'] = rs_data
    returnme = [return_dict[arg] for arg in args]
    if len(returnme) == 1:
        returnme = returnme[0]
    return returnme


# lambda function cannnot go across cpu, so I created it.
def nan_func(*args, **kwargs):
    return np.nan


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


def param_to_lf(phi_per_mass, alpha, dm, cms, clog_mass, all_bin_pair):
    schechter_sum = np.zeros(len(all_bin_pair[0]))
    cnum = len(cms)
    for i in range(cnum):
        ms = cms[i]
        log_mass = clog_mass[i]
        phi = phi_per_mass * 10**(log_mass - PIV_LOG_MASS)
        this_schechter = np.array([
            schechter_bins(phi,
                           alpha,
                           ms + dm,
                           bins=[all_bin_pair[i][j][0], all_bin_pair[i][j][1]])
            for j in range(len(all_bin_pair[i]))
        ]).flatten()
        schechter_sum += this_schechter
    return schechter_sum


def log_likelihood(p0, cms, clog_mass, sum_mean, sum_std, all_bin_pair):
    phi_per_mass, alpha, dm = p0
    schechter_sum = param_to_lf(phi_per_mass, alpha, dm, cms, clog_mass,
                                all_bin_pair)
    log_gauss = -(schechter_sum - sum_mean)**2 / (2 * sum_std**2)
    log_gauss[np.isnan(log_gauss)] = 0
    iambadvalues = (schechter_sum <= 0.0) & (np.isclose(sum_mean, 0.0))
    log_gauss[iambadvalues] = 0.0
    returnme = np.sum(log_gauss)
    if np.isnan(returnme).any() == True:
        print('p0 =', p0)
        print('schechter_sum =', schechter_sum)
        print('sum_mean, sum_std =', sum_mean, sum_std)
        return -np.inf
    return returnme


def log_prior(p0, cms, clog_mass):
    (phi_per_mass, alpha, dm) = p0
    phi = phi_per_mass * clog_mass
    ms = cms + dm
    if not ((0 < phi) & (phi < 1000)).all():
        return -np.inf
    if not ((15 < ms) & (ms < 25)).all():
        return -np.inf
    if not -5 < alpha <= 5:
        return -np.inf
    return 0.


def log_prob(p0, cms, clog_mass, sum_mean, sum_std, bin_pair):
    ll = log_likelihood(p0, cms, clog_mass, sum_mean, sum_std, bin_pair)
    lp = log_prior(p0, cms, clog_mass)
    return ll + lp


def find_ini_args(nwalkers, ndim):
    """Generate the initial values of MCMC
    
    Args:
        bkg_mean_d (ndarray): 1D array. Mean background density.
        nwalkers (int): Number of walkers.
        ndim (int): Number of MCMC parameters.
    
    Returns:
        ndarray: Initial values of MCMC parameters.
    """
    p0_number = [20., -1., 0.] # phi, alpha, dm
    p_var = [10., 0.2, 0.2]
    total_var = p_var
    randomization = np.random.rand(nwalkers, ndim) * 2
    randomization -= 1
    # Determine the scale of `p0_number`.
    p0 = np.transpose([np.full(nwalkers, i) for i in p0_number])
    p0 += total_var * randomization
    return p0


def find_mcmc_values(sampler, percent=(50, ), return_flat=False, quiet=False):
    """Find the proper result values from the sampler.
    
    It will flatten the sampler and return the percentile of it.

    Args:
        sampler (emcee.EnsembleSampler): the sampler.
        percent (array-like): A list contains the percentile you want the output 
            gives.
        return_flat (bool): return the flatten sampler if True.
        quiet (bool): Whether the `AutocorrError` should be applied. Default is 
            True.

    Returns:
        ndarray: (parNum) array.The percentile of the results from the sampler.
            If return_flat == True, there will be flatten sampler at the second 
            postion.
    """
    values = []
    tau = max(sampler.get_autocorr_time(quiet=quiet))
    flat_samples = sampler.get_chain(discard=int(3 * tau),
                                     thin=int(tau / 2),
                                     flat=True)
    ndim = np.shape(flat_samples)[1]
    for i in range(ndim):
        values.append(np.percentile(flat_samples[:, i], percent))
    values = np.array(values)
    #if len(percent) == 1:
    #    values = values.flatten()
    if return_flat == True:
        return values, flat_samples
    return values


def phi_model_mz(log_M, z, A, B, C):
    M_piv = 10**14  #10^14 M_sun/h
    z_piv = 0.4
    M = 10**np.array(log_M)
    return A * (M / M_piv)**B * ((1 + z) / (1 + z_piv))**C


def alpha_model_mz(log_M, z, alpha0, D, E):
    M_piv = 10**14  #10^14 M_sun/h
    z_piv = 0.4
    M = 10**np.array(log_M)
    return alpha0 * (M / M_piv)**D * ((1 + z) / (1 + z_piv))**E


def dm_model_mz(log_M, z, dm0, F, G):
    M_piv = 10**14  #10^14 M_sun/h
    z_piv = 0.4
    M = 10**np.array(log_M)
    return dm0 * (M / M_piv)**F * ((1 + z) / (1 + z_piv))**G



def get_used_bkg_bins(bkg_bins, all_obs_bins):
    cnum = len(all_obs_bins)
    bkg_mbins = (bkg_bins[:-1] + bkg_bins[1:]) / 2
    all_obs_mbins = []
    for i in range(cnum):
        this_bins = all_obs_bins[i]
        all_obs_mbins.append((this_bins[:-1] + this_bins[1:]) / 2)
    min_mag = np.min(all_obs_mbins)
    max_mag = np.max(all_obs_mbins)
    is_used = (min_mag <= bkg_mbins) & (bkg_mbins <= max_mag)
    # is_used(edge) = [(False)False, (False)False, (True)True, (True)True,
    #                  (True)False(False)]
    is_edge_used = np.full(len(bkg_bins), False)
    for i in range(len(is_used)):
        if is_used[i] == True:
            is_edge_used[i] = True
            is_edge_used[i + 1] = True
    used_bkg_bins = bkg_bins[is_edge_used]
    return is_used, used_bkg_bins




def get_sampler(all_obs_bins,
                ms_model,
                log_mass,
                sum_mean,
                sum_std,
                nwalkers='auto',
                step=10000,
                p0=None,
                cpu_num=1,
                progress=False,
                **kwargs):
    """The resulting [sampler, state, is_used] of the MCMC.
    
    This function runs the MCMC and gives the result.`sampler` and `state` can 
    be referred from `emcee`. `is_used` is a Boolean array which represents 
    whether the bin is used in each band.

    Args:
        all_obs_bins (ndarray): (cluNum, binBoundNum) array. The boundaries of
            the bins of the observaiontal LF for each cluster.
        ms_model (array-like): The model values of characteristic magnitude of
            each cluster.
        log_mass (array-like): The logarithm of the mass of the clusters.
        sum_mean (ndarray): The summation of the pure LF (applying masking correction)
        sum_std (ndarray): The uncertainty of `sum_mean`
        nwalkers (int or str): Number of walkers. If 'auto', this number will
            be automatically chosen (2.5 * number of parameters).
        step (int): The number of steps the MCMC will go through.
        p0 (ndarray or None): The initial parameter values for the MCMC. If
            `None`, it will be automaticlly generated (see `find_init_args()`).
        cpu_num (int): The number of cpu used in the MCMC.
        progress (bool): Whether the progress is shown.

    Returns:
        tuple: (sampler, state). They can be referred from `emcee`.
    """
    # log_prob(p0, cms, clog_mass, sum_mean, sum_std, bin_pair)
    # Goal: Find parameters needed for function `stacked_log_prob`.
    # Parameters: obs, ms_model, area, unmasked_fraction, obs_bin_pair,
    #             common_bkg_mean_d, common_bkg_std_d, bkg_mid_bins and
    #             obs_mid_bins.
    cnum = len(ms_model)
    # Some bins of the bkg are not used so they are removed for the fitting.
    # But for now, the background is just the mean bkg of all clusters.
    # So the remove is no longer needed.
    # is_used, used_bkg_bins = get_used_bkg_bins(bkg_bins, all_obs_bins)
    obs_bin_pair = [[[all_obs_bins[i][j], all_obs_bins[i][j + 1]]
                     for j in range(len(all_obs_bins[i]) - 1)]
                    for i in range(cnum)]
    obs_mid_bins = np.mean(obs_bin_pair, axis=2)
    ndim = ARG_NUM
    print('ndim =', ndim)
    if nwalkers == 'auto':
        nwalkers = int(ndim * 2.5)
    if p0 is None:
        p0 = find_ini_args(nwalkers, ndim)
    partial_log_prob = functools.partial(log_prob,
                                         cms=ms_model,
                                         clog_mass=log_mass,
                                         sum_mean=sum_mean,
                                         sum_std=sum_std,
                                         bin_pair=obs_bin_pair)
    print('np.shape(p0) =', np.shape(p0))
    print('ndim=', ndim)
    if cpu_num != 1:
        with multiprocessing.Pool(cpu_num) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            partial_log_prob,
                                            pool=pool)
            state = sampler.run_mcmc(p0, step, progress=progress)

    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, partial_log_prob)
        state = sampler.run_mcmc(p0, step, progress=progress)
    return sampler, state


def fit_band(z_list):
    # Find the proper band for fitting for a given redshift.
    z_list = np.atleast_1d(z_list)
    returnme = []
    for z in z_list:
        if 0.0 < z and z < 0.35:
            returnme.append('r')
        elif 0.35 <= z and z < 0.75:
            returnme.append('i')
        elif 0.75 <= z and z < 1.12:
            returnme.append('z')
        else:
            returnme.append('y')
    return returnme


def index2fl(hsc_index, hsc, band_name, bins):
    # Return the histogram of the magnitude of the given index.
    mag = hsc[band_name][hsc_index]
    return np.histogram(mag, bins)[0]


def get_obslf_cmag_cname(efeds_index, efeds, hsc, bins):
    """Get the obs LF, cmag and fitting band of the given index.
    
    This function return the observational LF, chachateristic magnitude and 
    fitting band of the given indicies of eFEDS.

    Args:
        efeds_index (ndarray): The index of the eFEDS.
        efeds (QTable): The eFEDS data.
        hsc (QTable): The HSC data.
        bins (array-like): The bins used in the LF.
    
    Returns:
        tuple: (all_lf, cmag, band). `all_lf` is the observational LF of the 
            given index. `cmag` is the characteristic magnitude of the given 
            index. `band` is the fitting band of the given index.
    """
    z = efeds['Z_BEST_COMB'][efeds_index]
    all_cmag = efeds['g_cmag', 'r_cmag', 'i_cmag', 'z_cmag',
                     'y_cmag'][efeds_index]
    all_hsc_index = efeds['galaxy_index'][efeds_index]
    cluster_num = len(z)
    band = fit_band(z)
    cmag_band = [i + '_cmag' for i in band]
    hsc_band = [i + 'mag_cmodel' for i in band]
    all_lf = [
        index2fl(all_hsc_index[i], hsc, hsc_band[i], bins)
        for i in range(cluster_num)
    ]
    cmag = [all_cmag[cmag_band[i]][i] for i in range(cluster_num)]
    return all_lf, cmag, band


def zmbins_efeds_index(zbins_i, mbins_i, zbins, mbins, efeds):
    """Get the indicies of eFEDS in the given redshift and mass bins

    When input the indicies of redshift and mass bins(`zbins_i` and `mbins_i`),
    this function will return the indicies of eFEDS whose redshift and mass are
    within the bins.
    
    Args:
        zbins_i (int): The index of the redshift bin.
        mbins_i (int): The index of the mass bin.
        zbins (array-like): The redshift bins.
        mbins (array-like): The mass bins.
        efeds (QTable): The eFEDS data.
    
    Returns:
        tuple: (efeds_index, z_range, m_range). `efeds_index` is the indicies of
            eFEDS in the given redshift and mass bins. `z_range` and `m_range`
            are the redshift and mass range of the given bins.
    """
    log_m = efeds['median_500c_lcdm'].value
    z = efeds['Z_BEST_COMB']
    low_z, up_z = zbins[zbins_i], zbins[zbins_i + 1]
    between_z = (z >= low_z) & (z < up_z)
    low_m, up_m = mbins[zbins_i][mbins_i], mbins[zbins_i][mbins_i + 1]
    between_m = ((log_m >= low_m) & (log_m < up_m))
    is_between = between_z & between_m
    return np.where(is_between)[0], [low_z, up_z], [low_m, up_m]


def proper_mag_bins(cmag, low_diff, up_diff, bin_width):
    """Generate the proper bins for the given characteristic magnitude.
    
    This function generates an array from `cmag - low_diff` to `cmag + up_diff` 
    with width `bin_width`(the boundaries are contained).

    Args:
        cmag (float): The characteristic magnitude.
        low_diff (float): The lower difference of the bins.
        up_diff (float): The upper difference of the bins.
        bin_width (float): The width of the bins.
    
    Returns:
        ndarray: The boundaries of the bins.
    """
    if not (np.isclose(((low_diff + up_diff) % bin_width), 0) or np.isclose(
        ((low_diff + up_diff) % bin_width), bin_width)):
        raise ValueError('Cannot generate new bins with the bin width same as '
                         'input bins perfectly. Please try to adjust diff or '
                         'bins')
    mag_min = cmag - low_diff
    mag_max = cmag + up_diff
    bin_num = int((low_diff + up_diff) / bin_width)
    new_bins = np.linspace(mag_min, mag_max, bin_num + 1)
    return new_bins


def easy_mcmc(efeds_index, efeds, hsc, rs_rd_result, rs_data, mode='red'):
    """Make a dict for the MCMC fitting
    
    This function generates a dict containing the data needed for the MCMC.
    For example, you can directly run get_sampler(**easy_mcmc(...)).
    Args:
        efeds_index (ndarray): The index of the eFEDS.
        efeds (QTable): The eFEDS data.
        hsc (QTable): The HSC data.
        rs_rd_result (dict): Data of the background results. See `init()`.
        rs_data (Table): Data of RS model from I-NON Chiu See `init()`.
    Returns:
        dict: The dict containing the data needed for the MCMC.
    """
    # Goal: Offer the parameters needed for `get_sampler`. They are obs_alllf,
    #       all_obs_bins, common_bkg_mean_d, common_bkg_std_d, bkg_bins,
    #       ms_model, area and unmasked_fraction.
    # Initial setup
    z = efeds[efeds_index]['Z_BEST_COMB'].value
    band = np.array(fit_band(z))
    band[band == 'y'] = 'z'  # y band is not used in the fitting
    band_index = band_name2index(band[0])
    cnum = len(efeds_index)
    unmasked_fraction = efeds[efeds_index]['unmasked_fraction'].value
    hsc_index = efeds['galaxy_index'][efeds_index]
    log_mass = efeds[efeds_index]['median_500c_lcdm'].value
    area = efeds[efeds_index]['area'].to(u.arcmin**2).value
    if len(np.atleast_1d(band)) == 1:
        band = np.full(len(efeds_index), band)
    ms_model = np.array(
        [efeds[band[i] + '_cmag'][efeds_index[i]] for i in range(cnum)])

    # Fitting bins
    all_obs_bins = [
        proper_mag_bins(ms_model[i], 2, 2, 0.2) for i in range(cnum)
    ]

    # Make obs LF
    isred = [is_red(hsc_index[i], hsc, z[i], rs_data) for i in range(cnum)]
    if mode == 'red':
        this_hsc_index = [hsc_index[i][isred[i]] for i in range(cnum)]
        all_bkg_mean_d = rs_rd_result['mean_red_bkg_d'][efeds_index].to(
            u.arcmin**-2).value
        all_bkg_std_d = rs_rd_result['std_red_bkg_d'][efeds_index].to(
            u.arcmin**-2).value

    obs_alllf = np.array([
        index2fl(this_hsc_index[i], hsc, band[i] + 'mag_cmodel', all_obs_bins[i])
        for i in range(cnum)
    ])
    obs_alllf_corrected = obs_alllf / unmasked_fraction[:, np.newaxis]
    obs_alllf_corrected_sum = np.sum(obs_alllf_corrected, axis=0)
    obs_alllf_corrected_sum_std = np.sum(obs_alllf /
                                     unmasked_fraction[:, np.newaxis]**2,
                                     axis=0)**0.5
    # Make bkg LF
    sum_bkg_mean = np.sum(all_bkg_mean_d * area[:, np.newaxis], axis=0)
    sum_bkg_std = np.sum(all_bkg_std_d**2 * area[:, np.newaxis]**2,
                         axis=0)**0.5
    sum_mean = obs_alllf_corrected_sum - sum_bkg_mean
    sum_std = (obs_alllf_corrected_sum_std**2 + sum_bkg_std**2)**0.5

    returnme = {
        'obs_alllf': obs_alllf,
        'obs_alllf_corrected': obs_alllf_corrected,
        'all_obs_bins': all_obs_bins,
        'all_bkg_mean_d': all_bkg_mean_d,
        'all_bkg_std_d': all_bkg_std_d,
        'sum_mean': sum_mean,
        'sum_std': sum_std,
        'band': band,
        'ms_model': ms_model,
        'log_mass': log_mass,
        'area': area,
        'z': z,
        'index': efeds_index,
        'unmasked_fraction': unmasked_fraction,
        'obs_alllf_corrected_sum': obs_alllf_corrected_sum,
        'obs_alllf_corrected_sum_std': obs_alllf_corrected_sum_std,
        'sum_bkg_mean': sum_bkg_mean,
        'sum_bkg_std': sum_bkg_std
    }
    return returnme


def value2bkg_d(value, is_used, bins=BINS):
    """Generate the background density based on the result of the MCMC.
    
    This function extracts the background density from, for example, 50%
    percentile of the MCMC result, and a Boolean array indicates which bins
    are used.

    Args:
        value (array-like): An 1D array that represents the result values of 
            MCMC.
        is_used (array-like): A (band, bins) Boolean array that indicates which 
            background bins are used.
        bins (array-like): The bins used in the background LF.
    Returns:
        tuple: (bkg_d, bkg_bins). `bkg_d` is the background density in each band.
            `bkg_bins` is the bins used in the background LF.
    """
    band_num = 5
    value_ = np.array(value)
    bkg_value = value_[ARG_NUM:]
    bkg_d = []
    bkg_bins = []
    bkg_num = np.sum(is_used, axis=1)
    for i in range(band_num):
        this_bins = np.full(len(bins), False)
        for j in range(len(this_bins) - 1):
            if is_used[i][j] == True:
                this_bins[j] = True
                this_bins[j + 1] = True
        num_sum = sum(bkg_num[:i])
        bkg_d.append(bkg_value[num_sum:num_sum + bkg_num[i]])
        bkg_bins.append(bins[this_bins])
    return bkg_d, bkg_bins


def value2bkg_d_funclist(value, is_used, bins=BINS):
    band_num = 5
    bkg_d, bkg_bins = value2bkg_d(value, is_used, bins)
    bkg_mbins = [(bkg_bins[i][:-1] + bkg_bins[i][1:]) / 2
                 for i in range(band_num)]
    bkg_func = []
    for i in range(band_num):
        if len(bkg_d[i]) == 0:
            bkg_func.append(lambda *args, **kwargs: np.nan)
            continue
        bkg_func.append(interpolate.interp1d(bkg_mbins[i], bkg_d[i]))
    return bkg_func


def value2purelf(value, log_M, z, ms, bins=BINS):
    value_ = np.array(value)
    # args = [A, B, alpha0, dm0, C, D, E, F, G] -> 9 args
    A, B, C = value_[[0, 1, 4]]
    alpha0, D, E = value_[[2, 5, 6]]
    dm0, F, G = value_[[3, 7, 8]]
    phi = phi_model_mz(log_M, z, A, B, C)
    alpha = alpha_model_mz(log_M, z, alpha0, D, E)
    dm = dm_model_mz(log_M, z, dm0, F, G)
    purelf = schechter_bins(ms + dm, phi, alpha, bins)
    return purelf


def get_color(index, hsc, z=None):
    hsc = hsc[index]
    if z is None:
        z = hsc['photoz_best']
    else:
        z = np.full(len(index), z)
    band = fit_band(z)
    band_name = np.array(['g', 'r', 'i', 'z', 'y'])
    band_num = np.array([np.where(band_name == b)[0][0] for b in band])
    bluer_band = band_name[band_num - 1]
    mag_err = np.array(
        [np.array(hsc[band[i] + 'mag_err'][i]) for i in range(len(hsc))])
    bluer_mag_err = np.array(
        [np.array(hsc[bluer_band[i] + 'mag_err'][i]) for i in range(len(hsc))])
    band = np.array([b + 'mag_cmodel' for b in band])
    bluer_band = np.array([b + 'mag_cmodel' for b in bluer_band])
    color = []
    color_err = (mag_err**2 + bluer_mag_err**2)**0.5
    for i in range(len(hsc)):
        color.append(np.array(hsc[bluer_band[i]][i] - hsc[band[i]][i]))
    color = np.array(color)
    return color, color_err


def get_fitmag(index, hsc, z=None):
    hsc = hsc[index]
    if z is None:
        z = hsc['photoz_best']
    else:
        z = np.full(len(index), z)
    band = fit_band(z)
    fitmag_err = np.array(
        [np.array(hsc[band[i] + 'mag_err'][i]) for i in range(len(hsc))])
    band = np.array([b + 'mag_cmodel' for b in band])
    fitmag = np.array([np.array(hsc[band[i]][i]) for i in range(len(hsc))])
    return fitmag, fitmag_err


def get_rsfunc(z, rs_data):
    rs_yint_fixslope = interpolate.interp1d(rs_data['mean_zcl'],
                                            rs_data['rs_yint_fixslope'],
                                            fill_value='extrapolate')
    rs_slope_fixslope = rs_data['rs_slope_fixslope'][0]  # It's fixed
    colorwidth_slope_fixslope = interpolate.interp1d(
        rs_data['mean_zcl'],
        rs_data["colorwidth_slope_fixslope"],
        fill_value='extrapolate')
    colorwidth_yint_fixslope = interpolate.interp1d(
        rs_data['mean_zcl'],
        rs_data['colorwidth_yint_fixslope'],
        fill_value='extrapolate')
    yint = rs_yint_fixslope(z)
    slope = rs_slope_fixslope
    cw_yint = colorwidth_yint_fixslope(z)
    cw_slope = colorwidth_slope_fixslope(z)
    cw_err_func = interpolate.interp1d(rs_data['mean_zcl'],
                                       np.nanmin(rs_data['colorwidth_mea'],
                                                 axis=1),
                                       fill_value='extrapolate')

    def rsfunc(mag):
        return slope * (mag - 20.0) + yint

    def cw_rsfunc(mag):
        returnme = np.atleast_1d(cw_slope * (mag - 20.0) + cw_yint)
        cw_err = np.atleast_1d(cw_err_func(z))
        returnme[returnme < cw_err] = cw_err
        return returnme

    return rsfunc, cw_rsfunc

# blue_mean = red_mean - red_std * 3.5, blue_std = red_std * 1.4
def get_bsfunc(z, rs_data):
    rs_func, cw_rsfunc = get_rsfunc(z, rs_data)
    mean_diff = -2.7 # loc_blue - loc_red, must be negative
    std_fraction = 1. # scale_blue / scale_red, must be positive
    def bsfunc(mag):
        return rs_func(mag) + mean_diff * cw_rsfunc(mag)
    def cw_bsfunc(mag):
        return cw_rsfunc(mag) * std_fraction
    return bsfunc, cw_bsfunc


def band_name2index(name):
    """Convert the band name to a number

    g/r/i/z/y corresponds to 0/1/2/3/4.

    Args:
        name (array-like): A list of band name. It should only contains 'g',
            'r', 'i', 'z' and 'y' elements.
    
    Returns:
        (ndarray or int): The corresponding numbers. If len(name) == 1, only
            an integer is returned instead of an array.
    """
    name = np.atleast_1d(name)
    returnme = []
    for i in range(len(name)):
        returnme.append(np.where(BAND_NAME == name[i])[0][0])
    if len(name) == 1:
        returnme = returnme[0]
    return returnme


def red_prob(index, hsc, zcl, rs_data, mode='zcl_reference'):
    if mode == 'zgl_reference':
        color, color_err = get_color(index, hsc)
        fitmag, fitmag_err = get_fitmag(index, hsc)
    elif mode == 'zcl_reference':
        color, color_err = get_color(index, hsc, zcl)
        fitmag, fitmag_err = get_fitmag(index, hsc, zcl)
    rsfunc, cw_rsfunc = get_rsfunc(zcl, rs_data)
    middle = rsfunc(fitmag)  # Middle of color
    width = cw_rsfunc(fitmag)
    cdf = stats.norm.cdf(color, loc=middle, scale=width)
    return cdf


def blue_prob(index, hsc, zcl, rs_data, mode='zcl_reference'):
    if mode == 'zgl_reference':
        color, color_err = get_color(index, hsc)
        fitmag, fitmag_err = get_fitmag(index, hsc)
    elif mode == 'zcl_reference':
        color, color_err = get_color(index, hsc, zcl)
        fitmag, fitmag_err = get_fitmag(index, hsc, zcl)
    bsfunc, cw_bsfunc = get_bsfunc(zcl, rs_data)
    middle = bsfunc(fitmag)
    width = cw_bsfunc(fitmag)
    cdf = stats.norm.cdf(color, loc=middle, scale=width)
    return 1 - cdf


def is_red(index, hsc, zcl, rs_data):
    red_p = red_prob(index, hsc, zcl, rs_data)
    blue_p = blue_prob(index, hsc, zcl, rs_data)
    total_p = red_p + blue_p
    final_red_p = red_p / total_p
    return hsc['random'][index] <= final_red_p


def get_redblue_lf(hsc_index, hsc, zcl, rs_data, bins):
    """Get the red LF and blue LF of the input HSC indicies
    
    This function first distinguishes which galaxy (i.e. `hsc[hsc_index]`) is
    red or blue, and get the LF of red/blue galaxy groups in fitting band
    (see `fit_band`).
    
    Args:
        hsc_index (list): List of HSC index.
        hsc (QTable): HSC data.
        zcl (float): Redshift of interest (ex. redshift of the galaxy cluster) 
            for finding the fitting band.
        rs_data (fits): Data of red sequence. See `init`.
        bins (ndarray): Bins for the LF.

    Returns:
        tuple: The first element is the red LF and the second is the blue LF.
    """
    band = fit_band([zcl])[0]
    isred = is_red(hsc_index, hsc, zcl, rs_data)
    red_hsc = hsc[hsc_index[isred]]
    blue_hsc = hsc[hsc_index[~isred]]
    red_mag = red_hsc[band + 'mag_cmodel']
    blue_mag = blue_hsc[band + 'mag_cmodel']
    red_lf = np.histogram(red_mag, bins)[0]
    blue_lf = np.histogram(blue_mag, bins)[0]
    return red_lf, blue_lf


def sr_efeds_index(efeds):
    # Return the indicies where efeds clusters are science ready
    return np.where(efeds['low_cont_flag'] & (efeds['unmasked_fraction'] > 0.6))[0]

def is_sr_efeds(efeds):
    # Return whether the efeds clusters are science ready
    return efeds['low_cont_flag'] & (efeds['unmasked_fraction'] > 0.6)


def index_group(efeds, mode):
    # Easily group the efeds needed for analysis
    efeds_group = [
        np.where((ZBINS2[i] <= efeds['Z_BEST_COMB'])
                 & (efeds['Z_BEST_COMB'] < ZBINS2[i + 1]))[0]
        for i in range(len(ZBINS2) - 1)
    ]
    # Make sure they are science-ready
    for i in range(len(efeds_group)):
        these_efeds = efeds[efeds_group[i]]
        is_sr = these_efeds['low_cont_flag'] & (
            these_efeds['unmasked_fraction'] > 0.6)
        efeds_group[i] = efeds_group[i][is_sr]

    if mode == 'z':
        return efeds_group
    if mode == 'z_mass':
        new_efeds_group = []
        # The last group contains only 6 cluster so we don't divide the last
        # group based on the mass
        for i in range(len(efeds_group) - 1):
            index = efeds_group[i]
            this_efeds = efeds[index]
            all_mass = this_efeds['median_500c_lcdm']
            median_mass = np.median(all_mass)
            low_index = index[all_mass <= median_mass]
            high_index = index[all_mass > median_mass]
            new_efeds_group.append(low_index)
            new_efeds_group.append(high_index)
        new_efeds_group.append(efeds_group[-1])
        return new_efeds_group
