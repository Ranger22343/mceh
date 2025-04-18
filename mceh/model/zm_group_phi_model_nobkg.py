import numpy as np
import emcee
from scipy.integrate import quad
from astropy.io import fits
from .. import find_luminosity as fl
import matplotlib.pyplot as plt
import corner
from scipy.optimize import curve_fit
import copy
import multiprocessing
from functools import partial
from scipy import interpolate
import astropy.units as u
from scipy import special
from astropy.table import QTable
from .. import utility as ut

multiprocessing.set_start_method('fork', force=True)
BINS = np.arange(10, 30.1, 0.2)
MBINS = BINS[:-1] / 2 + BINS[1:] / 2
LABELS = ['A', 'B', r'$\alpha$', r'$\Delta m$']
DIFF_BINS = np.arange(-2, 2.1, 0.2)
DIFF_MBINS = DIFF_BINS[:-1] / 2 + DIFF_BINS[1:] / 2


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
        if arg == 'zmbins':
            zmbins = ut.pickle_load('data/zmbins20241108.pickle')
            return_dict['zmbins'] = zmbins
        if arg == 'sr_efeds':
            efeds = QTable.read('data/modified_efeds_ver8.fits')
            efeds = efeds[efeds['low_cont_flag'] 
                          & (efeds['unmasked_fraction'] > 0.6)]
            return_dict['sr_efeds'] = efeds
        if arg == 'rd_result':
            rd_result = ut.pickle_load('data/bkg_lf20241111.pickle')
            return_dict['rd_result'] = rd_result
    returnme = [return_dict[arg] for arg in args]
    if len(returnme) == 1:
        returnme = returnme[0]
    return returnme


def isbin_by_isbound(is_bound_bins):
    """A boolean array corresponds the histogram from its bin.

    Since the length of the bounds of the bins is one element larger than the
    length of the histogram, it corresponds the boolean array indicates which
    bins should be taken to the histogram.
    ex:
    >>> hist = [10, 20, 30, 40, 50]
    >>> bins = [1,2,3,4,5,6]
    If you want to cut the bins to be [2,3,4]:
    >>> is_bins = [False, True, True, True, False, False]
    The corresponding `hist` is [20, 30] since [2, 3, 4] are their boundaries.
    Then is_bin_by_isbound(is_bins) gives:
    >>> isbin_by_isbound(is_bins)
    [False, True, True, False, False]
    It corresponds to [20, 30].
    
    You can use `isbin_by_isbound` to generate the `is_bound_bins`.

    Args:
        is_bound_bins (array-like): A boolean array indicates which bin you want 
            to take.
    
    Returns:
        array: A boolean array indicates the histogram being cut.

    """
    return_bins = copy.deepcopy(is_bound_bins)
    #change the last "True" in bins to False
    return_bins[np.where(is_bound_bins)[0][-1]] = False

    return_bins = np.delete(return_bins, -1)  #delete the last element
    return return_bins


def isbin_by_bound(bin, bound, mode='close'):
    """A boolean array based on the input `bin` and the `bound` wanted.
    
    When `mode` is 'close', it will return a boolean array which indicates
    which upper bound of `bin` is the closest to bound[1] and the lower
    bound is the closest to bound[0]; the qualified bound will be set as True. 
    The other parts between these two bin bounds are also True.
    
    More more information. Please see the Returns part.

    Args:
        bin: the bin you want to calibrate.
        bound: an 2D array-like object which first element is the lower bound 
            and the second element is the upper bound.
        mode:
            'close': the upper bound and the lower bound of the output array 
                will be determined by the elements in `bin` which are the 
                closest to `bound`. Every element of output corresponds to `bin` 
                within the bounds will be True and the otehrs are False.
                
                If there are multiple elements meet this condition, the lower 
                bound will choose the first qualified one, while the upper bound 
                will choose the last qualified one. 
                ex:
                in: bin = [1,3,5,7,9] bound = [2.6,6.8]
                out: [False, True, True, True, False] 

                in: bin = [1,3,5,7,9] bound = [2,6]
                out: [True, True, True, True False]
                        
            'contain': the upper bound and the lower bound of the output array 
                will be determined by the elements in `bin` which cover `bound`.
    
    Returns:
        boolean array: Based on the mode.
    """
    bin = np.array(bin)
    if np.shape(bound) == (2, ):
        bound = (bound, )
    returnme = []
    if mode == 'close':
        for b in bound:
            l_bound_index = np.where(abs(bin - b[0]) == min(abs(bin -
                                                                b[0])))[0][0]
            h_bund_index = np.where(abs(bin - b[1]) == min(abs(bin -
                                                               b[1])))[0][-1]
            result = np.full(len(bin), False)
            result[l_bound_index:h_bund_index + 1] = True
            returnme = result

    elif mode == 'contain':
        #TODO(hylin): I will finish it if I need to.
        pass
    return returnme


def schechter(m, m_s, phi_s, alpha):
    return 0.4 * np.log(10) * phi_s * (10**(0.4 * (m_s - m)))**(
        alpha + 1) * np.exp(-10**(0.4 * (m_s - m)))


def gammainc(s, x, eps=1e-6):
    # positive/negative
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
    # ex -1 - 1e-8 -> - 1e-8 int=-1
    # ez -1 + 1e-8 -> 1e-8 int=0
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


def log_likelihood(p0, obs, bin_pair, unmasked_fraction, ms_model, log_mass):
    A, B, alpha, dm = p0
    S = np.array([
        schechter_bins(phi_model(log_mass, A, B),
                       alpha,
                       ms_model + dm,
                       bins=[bin_pair[i][0], bin_pair[i][1]])
        for i in range(len(bin_pair))
    ]).flatten()
    M = S * unmasked_fraction
    D = obs
    if (M < 0).any():
        return -np.inf
    iambadvalues = (M <= 0.0) & (np.isclose(D, 0.0))  # np.close?
    returnme_array = -M + D * np.log(M)  # will have warnings.
    returnme_array[iambadvalues] = 0.0
    returnme = np.sum(returnme_array)
    if np.isnan(returnme).any() == True:
        print('S =', S)
        print('D =', D)
        print('p0 =', p0)
        return -np.inf
    return returnme


def log_gauss_value(mean, std, int_range):

    def gauss_d(x):
        return np.exp(-(x - mean)**2 / (2 * std**2))

    log_gauss = np.log10(quad(gauss_d, int_range[0], int_range[1]))
    return log_gauss


def log_prior(p0, ms_model, log_mass):
    (A, B, alpha, dm) = p0
    m_s = ms_model + dm
    phi_s = phi_model(log_mass, A, B)
    if not 0 < phi_s < 1000:
        return -np.inf
    if not 10 < m_s < 30:
        return -np.inf
    if not -5 < alpha <= 5:
        return -np.inf
    if not A >= 0:
        return -np.inf
    if not -5 <= B <=5:
        return -np.inf
    return 0


def log_prob(p,
             obs,
             bkg,
             rd_bkg_mean,
             rd_bkg_std,
             bins=np.linspace(14, 24, 41)):
    lp = log_prior(p, bkg, rd_bkg_mean, rd_bkg_std)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(p, obs, bkg, bins)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def find_half(cluster, check=False):
    '''
    This function will return the fainter and brighter part of the luminosity function and the background is subtracted.
    '''
    real = cluster.observation - cluster.bkg
    total = sum(real)
    for i in range(len(real)):
        if sum(real[:i + 1]) >= total / 2:
            median_bin_index = i + 1
            break
    low_hist = real[:median_bin_index]
    high_hist = real[median_bin_index - 1:]
    low_bin = cluster.bins[:median_bin_index + 1]
    high_bin = cluster.bins[median_bin_index - 1:]
    median_value = (cluster.bins[median_bin_index - 1] +
                    cluster.bins[median_bin_index]) / 2
    if check == True:
        plt.stairs(cluster.observation - cluster.bkg,
                   cluster.bins,
                   label='obs-bkg',
                   color='silver',
                   fill=True)
        plt.stairs(low_hist,
                   low_bin,
                   label='faint-end',
                   color='red',
                   hatch='//')
        plt.stairs(high_hist,
                   high_bin,
                   label='bright-end',
                   color='blue',
                   hatch='\\')
        plt.axvline(x=median_value, color='black')
        plt.legend()
    return [low_hist, low_bin, high_hist, high_bin, median_bin_index]


def find_ini_args(nwalkers, ndim):
    # A, B, alpha, dm 30 1 -1 0
    # rd_bkg_mean = [[b11, b12, b13...], [b21, b22, b23] ...],
    # where bij is the jth value of the ith cluster.
    p0_number = [30, 1, -1, 0]
    p_var = [15, 0.2, 1, 0.2]  # Variations of A, B, alpha and dm
    total_var = np.full((nwalkers, ndim), p_var)
    randomization = np.random.rand(nwalkers, ndim) * 2
    randomization -= 1
    # Determine the scale of `p0_number`.
    # (nwalkers, ndim)
    p0 = np.transpose([np.full(nwalkers, i) for i in p0_number])
    p0 = p0 + np.array(total_var) * randomization
    return p0


def find_mcmc_values(sampler, percent=(50, ), return_flat=False, quiet=False):
    """Find the proper result values from the sampler.
    
    It will flatten the sampler and return the percentile of it.

    Args
    ----
    sampler: the sampler.
    percent: an list contains the percentile you want the output gives.
    return_flat: return the flatten sampler if True.
    quiet: Whether the `AutocorrError` should be applied. Default is True.

    Returns
    -------
    The percentile of the results from the sampler.
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


# group clusters by redshift & mass
def make_group(efeds, z_bound, m_bound):
    # [[z1m1, z1m2, z1m3, ...], [z2m1, z2m2, z2m3, ...], ...]
    z_ilist = []
    index_list = [[] for i in range(len(z_bound) - 1)]
    z = efeds['z']
    m = efeds['median_500c_lcdm']
    for i in range(len(z_bound) - 1):
        index_z = np.where((z > z_bound[i]) & (z < z_bound[i + 1]))[0]
        for j in range(len(m_bound) - 1):
            index_m = np.where((m[index_z] > m_bound[j])
                               & (m[index_z] < m_bound[j + 1]))[0]
            index_m = index_z[index_m]
            index_list[i].append(index_m)
    return index_list


def phi_model(log_M, A, B):
    """The predicted phi in the Schechter function.
    
    The predicted phi in the Schechter function. phi = A * (M / M_piv) ** B.

    Args:
        log_M (array-like): log(Mass of the clusters/(M_sun/h)).
        A (float): A parameter of the model.
        B (float): A parameter of the model.
    
    Retruns:
        float: The predicted phi.
    """
    M_piv = 10**14  #10^14 M_sun/h
    M = 10**np.array(log_M)
    return A * (M / M_piv)**B


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


def bkg_density(z, z_list, rd_bkg_mean, rd_bkg_std):
    mean4z = np.array([
        np.interp(z, z_list, rd_bkg_mean[:, i])
        for i in range(len(rd_bkg_mean))
    ])
    std4z = np.array([
        np.interp(z, z_list, rd_bkg_std[:, i]) for i in range(len(rd_bkg_mean))
    ])
    return [mean4z, std4z]


# def log_prob(p, obs, bkg, rd_bkg_mean, rd_bkg_std, bins = np.linspace(14,24,41), double = False):
def stacked_log_prob(p,
                     obs,
                     ms_model,
                     log_mass,
                     area,
                     all_bin_pair,
                     z,
                     unmasked_fraction):
    """The sum of the logarithmic probability of the group of clusters. 

    Args:
        p: (4+number_of_bins,) array-like
            Parameters (A, B, alpha, dm) and the common background.
        obs: (number_of_clusters, number_of_bins) array-like
            The observation LF of the clusters.
        ms_model: (number_of_clusters,) array-like
            The characeristic magnitude of the clusters.
        log_mass: (number_of_clusters,) array-like
            log(mass_of_the_clusters/(M_sun/h)).
        bkg: (number_of_clusters, number_of_bins) array-like
            The background LF of the clusters.
        all_bin_pair: (number_of_clusters, number_of_bin_pairs) array-like
            The upper and lower limit of every bin for every cluster.
            For example, for a cluster whose bins are [14, 14.25, 14.5, 14.75],
            its bin pair is [[14, 14.25], [14.25, 14.5], [14.5, 14.75]].
        z (array-like): Redshift of the clusters.
    
    Returns:
        float
        The sum of the logarithmic probability of the group of clusters.
    """
    ms_model = np.atleast_1d(ms_model)
    area = np.atleast_1d(area)
    log_mass = np.atleast_1d(log_mass)
    obs = np.atleast_2d(obs)
    cluster_num = len(ms_model)
    (A, B, alpha, dm) = p[:4]
    if len(np.unique([len(obs), len(ms_model), len(area)])) != 1:
        raise ValueError('obs, ms_model, area, common_bkg_mean_d and'
                         'common_bkg_std_d must have the same length')
    # all_mid_bins = np.mean(all_bin_pair, axis=2)
    lp_list = [
        log_prior((A, B, alpha, dm), ms_model[i], log_mass[i]) 
        for i in range(cluster_num)
    ]  # its p = (m_s, phi_s, alpha)
    if not np.isfinite(sum(lp_list)):
        return -np.inf
    ll_list = [
        log_likelihood((A, B, alpha, dm), obs[i], 
                       all_bin_pair[i], unmasked_fraction[i],
                       ms_model[i], log_mass[i]) 
                       for i in range(cluster_num)
    ]
    returnme = sum(lp_list) + sum(ll_list)
    if np.isnan(returnme).any() == True:
        return -np.inf
    return sum(lp_list) + sum(ll_list)


def bkg_info(z, z_list, all_rd_bkg_mean, all_rd_bkg_std, area=None):
    """Get the random background mean and std of LF based on the redshift.

    The mean and the std of the random background LF is obtained by the
    interpolation of the data input.

    Args:
        z: (M,) array-like
            The redshift whose background data is what you want to obtain.
        area: (M,) array-like
            The area of the clusters corresponds to `z`. If None, return the
            density (N/area).
        z_list: (N, ) array-like
            The redshift interval bounds which you obatined `all_rd_bkg_mean` 
            and `all_rd_bkg_std`.
        all_rd_bkg_mean: (N-1, P) array-like
            The values of mean background LF.
        all_rd_bkg_std: (N-1, P) array-like
            The values of std of the background LF.
    
    Returns:
        (2, M, P) array
        The first element is the mean LF of the background for each redshift,
        while the second one is for std. For each element, the first index
        indicates the result LF for each redshift while the second index is the
        bin. If `area` = None, the return LF will be in unit of number density
        (N/area).
        i.e. array([[z1b1, z1b2, ...], [z2b1, z2b2, ...], ...]) for both mean 
        and std, which zibj means the ith z and the jth bin.
    """
    # all_rd_bkg_mean = [[c1b1, c1b2...], [c2b1, c2b2...]...]
    z = np.array(z)
    bins_num = len(all_rd_bkg_mean[0])
    all_rd_bkg_mean_t = np.transpose(all_rd_bkg_mean)
    all_rd_bkg_std_t = np.transpose(all_rd_bkg_std)
    bin_value_mean = []  # [[b1c1, b1c2, b1c3...], [b2c1, b2c2, b2c3...]...]
    bin_value_std = []
    for i in range(bins_num):
        bin_value_mean.append(np.interp(z, z_list, all_rd_bkg_mean_t[i]))
        bin_value_std.append(np.interp(z, z_list, all_rd_bkg_std_t[i]))
    mean = np.transpose(bin_value_mean)
    std = np.transpose(bin_value_std)
    if area is not None:
        for i in range(len(area)):
            mean[i] *= area[i]
            std[i] *= area[i]
    return np.array([mean, std])


def cut_mag(cmag, lf, diff, bins=np.linspace(14, 24, 41)):
    """Interpolate the lf to the given magnitude difference.
    
    (low_diff, high_dff) = diff. First, generate new bins [cmag - low_diff, 
    cmag - low_diff + bin_width, cmag - low_diff + 2 * bin_width, ..., 
    cmag + high_diff] with bin_width the width of `bins`. Then, interploate `lf`
    to the new bins. It returns the new lf and new bins.

    Args:
        cmag (float): The characteristic magnitude of the cluster.
        lf (array): The luminosity function of the cluster.
        diff ((2,) array-like): [low_diff, up_diff] which are the lower/upper 
            difference to the cmag for the new boundary.
        bins (array): The magnitude bins of `lf`.
    
    Returns:
        (list): The first element is the resulting lf while the second one is
            the new bins.
    """
    if len(np.unique(np.diff(bins))) != 1:
        raise ValueError('The width of the bins is not same.')
    bin_diff = bins[1] - bins[0]
    low_diff, up_diff = diff
    mid_bins = (bins[:-1] + bins[1:]) / 2
    mag_min = cmag - low_diff
    mag_max = cmag + up_diff
    if mag_min < min(bins) or mag_max > max(bins):
        raise ValueError('WARNING: The minimal or maximal magnitude is not '
                         'contained within'
                         f'[cmag-{low_diff}, cmag+{up_diff}].')
    lf_func = interpolate.interp1d(mid_bins, lf)
    if (low_diff + up_diff) % bin_diff != 0:
        raise ValueError('Cannot generate new bins with the bin width same as '
                         'input bins perfectly. Please try to adjust diff or '
                         'bins')
    bin_num = int((low_diff + up_diff) / bin_diff)
    new_bins = np.linspace(mag_min, mag_max, bin_num + 1)
    mid_new_bins = (new_bins[1:] + new_bins[:-1]) / 2
    new_lf = lf_func(mid_new_bins)
    return new_lf, new_bins


def get_sampler(obs_alllf,
                every_obs_bins,
                ms_model,
                log_mass,
                area,
                z,
                unmasked_fraction,
                nwalkers='auto',
                step=10000,
                p0=None,
                cpu_num=1,
                progress=False,
                check=False,
                **kwargs):

    cluster_num = len(obs_alllf)

    # Check validity of the data
    all_bin_pair = [[[every_obs_bins[i][j], every_obs_bins[i][j + 1]]
                     for j in range(len(every_obs_bins[i]) - 1)]
                    for i in range(cluster_num)]
    ndim = 4  # A, B, alpha, dm
    print('ndim =', ndim)
    if nwalkers == 'auto':
        nwalkers = int(ndim * 2.5)
    if p0 is None:
        p0 = find_ini_args(nwalkers, ndim)
    partial_log_prob = partial(stacked_log_prob,
                               obs=obs_alllf,
                               ms_model=ms_model,
                               log_mass=log_mass,
                               area=area,
                               z=z,
                               unmasked_fraction=unmasked_fraction,
                               all_bin_pair=all_bin_pair)
    if check == True:
        return partial_log_prob
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
    return [sampler, state]


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


def get_common_bkg_d(galaxy_index, efeds, bkg_all):
    efeds_ = efeds[galaxy_index]

    bkg_mean_d, bkg_std_d = bkg_info(np.atleast_1d(efeds_['z']),
                                     bkg_all['z_list'], bkg_all['mean'],
                                     bkg_all['std'])
    common_bkg_mean_d = np.mean(np.atleast_2d(bkg_mean_d), axis=0)
    common_bkg_std_d = (
        (np.sum(np.atleast_2d(bkg_std_d)**2, axis=0))**0.5 / len(bkg_std_d)
    )  # mean after ^ 0.5
    return common_bkg_mean_d, common_bkg_std_d


def index2fl(hsc_index, hsc, band_name, bins):
    mag = hsc[band_name][hsc_index]
    return np.histogram(mag, bins)[0]


def get_obslf_cmag_cname(efeds_index, efeds, hsc, bins):
    #z, all_cmag, all_hsc_index
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
    """Return the index of the eFEDS satisfying the mass/redshit conditions
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


def pre_mcmc_dict(efeds_index, efeds, hsc, band, unmasked_fraction=None): # band = g r i z y
    new_efeds = efeds[efeds['low_cont_flag']
                      & (efeds['unmasked_fraction'] > 0.6)]
    cnum = len(efeds_index)
    band = np.atleast_1d(band)
    if len(band) == 1:
        band = np.full(cnum, band)
    ms_model = np.array([efeds[band[i] + '_cmag'][efeds_index[i]] 
                         for i in range(cnum)])
    every_obs_bins = [
        proper_mag_bins(ms_model[i], 2, 2, 0.2)
        for i in range(len(efeds_index))
    ]
    obs_alllf = [
        index2fl(efeds['galaxy_index'][efeds_index[i]], hsc,
                 band[i] + 'mag_cmodel', every_obs_bins[i])
        for i in range(len(efeds_index))
    ]
    bkg_bins = BINS
    log_mass = new_efeds[efeds_index]['median_500c_lcdm'].value
    area = new_efeds[efeds_index]['area'].to(u.arcmin**2).value
    z = new_efeds[efeds_index]['Z_BEST_COMB'].value
    if unmasked_fraction == None:
        unmasked_fraction = np.full(cnum, 1)
    returnme = {
        'obs_alllf': obs_alllf,
        'every_obs_bins': every_obs_bins,
        'bkg_bins': bkg_bins,
        'ms_model': ms_model,
        'log_mass': log_mass,
        'area': area,
        #'zbound': z_bound,
        #'mbound': m_bound,
        'index': efeds_index,
        'z': z,
        'unmasked_fraction': unmasked_fraction
    }
    return returnme
        

def easy_mcmc(efeds, hsc, zmbins, unmasked_fraction=None, mode='group'):
    new_efeds = efeds[efeds['low_cont_flag']
                      & (efeds['unmasked_fraction'] > 0.6)]
    zbins, mbins = zmbins['zbins'], zmbins['mbins']
    zm_index = [] # z, mass
    for i in range(len(zmbins['zbins']) - 1):
        for j in range(len(zmbins['mbins'][i]) - 1):
            zm_index.append((i, j))
    if mode == 'group':
        returnme = []
        for i in range(len(zm_index)):
            z_index = zm_index[i][0]
            m_index = zm_index[i][1]
            efeds_index, z_bound, m_bound = zmbins_efeds_index(z_index, m_index, zbins,
                                                            mbins, new_efeds)
            band = fit_band(new_efeds['Z_BEST_COMB'][efeds_index])
            this_dict = pre_mcmc_dict(efeds_index, new_efeds, hsc, band, 
                                      unmasked_fraction=unmasked_fraction)
            this_dict['zbound'] = z_bound
            this_dict['mbound'] = m_bound
            this_dict['zindex'] = z_index
            this_dict['mindex'] = m_index
            returnme.append(this_dict)
    elif mode == 'overall':
        efeds_index = list(range(len(efeds)))
        band = fit_band(new_efeds['Z_BEST_COMB'][efeds_index])
        returnme = pre_mcmc_dict(efeds_index, efeds, hsc, band, 
                                 unmasked_fraction=unmasked_fraction)
    return returnme


def efeds2fake(efeds, fake_hsc, mode='real'):
    if mode == 'real':
        for i in range(len(efeds)):
            efeds['galaxy_index'][i] = np.where(fake_hsc['host_index'] == i)[0]
            efeds['unmasked_fraction'][i] = 1
    return efeds