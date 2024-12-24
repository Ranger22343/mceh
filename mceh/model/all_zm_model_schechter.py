import numpy as np
import emcee
from scipy import integrate
import multiprocessing
import functools
from scipy import interpolate
import astropy.units as u
import os

os.environ["OMP_NUM_THREADS"] = "1"
multiprocessing.set_start_method('fork', force=True)
BINS = np.arange(10, 30.1, 0.2)  # Boundaries of bins
MBINS = BINS[:-1] / 2 + BINS[1:] / 2
DIFF_BINS = np.arange(-2, 2.1, 0.2)
DIFF_MBINS = (DIFF_BINS[:-1] + DIFF_BINS[1:]) / 2
# args = [A, B, alpha0, dm0, C, D, E, F, G] -> 9 args
# phi = A * fB(M) * fC(z)
# alpha = alpha0 * fD(M) * fE(z)
# dm = dm0 * fF(M) * fG(z)
ARG_NUM = 9
LABELS = [r'$\phi_0$', r'$\beta_\phi$', r'$\alpha_0$', 
          r'$\Delta m_0$', r'$\gamma_\phi$', r'$\beta_\alpha$', 
          r'$\gamma_\alpha$', r'$\beta_{m}$', r'$\gamma_{m}$']

# lambda function cannnot go across cpu, so I created it.
def nan_func(*args, **kwargs):
    return np.nan


def schechter(m, m_s, phi_s, alpha):
    return 0.4 * np.log(10) * phi_s * (10**(0.4 * (m_s - m)))**(
        alpha + 1) * np.exp(-10**(0.4 * (m_s - m)))


def schechter_bins(m_s, phi_s, alpha, bins=BINS):
    #the number in each bin estimated by Schechter function
    result = []
    for i in range(len(bins) - 1):
        result.append(
            integrate.quad(schechter,
                           bins[i],
                           bins[i + 1],
                           args=(m_s, phi_s, alpha))[0])
    result = np.array(result)
    return result


def log_likelihood(p0, obs, bkg, bin_pair, unmasked_fraction):
    m_s, phi_s, alpha = p0
    #print('m_s, phi_s, alpha =', p0)
    if (bkg < 0).any():
        return -np.inf
    S = np.array([
        schechter_bins(m_s,
                       phi_s,
                       alpha,
                       bins=[bin_pair[i][0], bin_pair[i][1]])
        for i in range(len(bin_pair))
    ]).flatten()
    M = (S + bkg) * unmasked_fraction
    D = obs
    if (M < 0).any():
        return -np.inf
    iambadvalues = (M <= 0.0) & (np.isclose(D, 0.0))
    returnme_array = -M + D * np.log(M)  # will have warnings.
    returnme_array[iambadvalues] = 0.0
    returnme = np.sum(returnme_array)
    if np.isnan(returnme).any() == True:
        print('S =', S)
        print('bkg =', bkg)
        print('D =', D)
        return -np.inf
    return returnme


def log_prior(p0, bkg, rd_bkg_mean, rd_bkg_std):
    (m_s, phi_s, alpha) = p0
    if not 0 < phi_s < 1000:
        return -np.inf
    if not 15 < m_s < 25:
        return -np.inf
    if not -5 < alpha <= 5:
        return -np.inf
    bkg = np.array(bkg)
    if (bkg < 0).any():
        return -np.inf
    rd_bkg_mean = np.array(rd_bkg_mean)
    rd_bkg_std = np.array(rd_bkg_std)
    log_bkg_gauss = -(bkg - rd_bkg_mean)**2 / (2 * rd_bkg_std**2)
    log_bkg_gauss[np.isnan(log_bkg_gauss)] = 0
    return np.sum(log_bkg_gauss)


def find_ini_args(bkg_mean_d, nwalkers, ndim):
    """Generate the initial values of MCMC
    
    Args:
        bkg_mean_d (ndarray): 1D array. Mean background density.
        nwalkers (int): Number of walkers.
        ndim (int): Number of MCMC parameters.
    
    Returns:
        ndarray: Initial values of MCMC parameters.
    """
    # rd_bkg_mean = [[b11, b12, b13...], [b21, b22, b23] ...],
    # where bij is the jth value of the ith cluster.
    # args = [A, B, alpha0, dm0, C, D, E, F, G]
    p0_number = [30, 1, -1, 0, 0, 0, 0, 0, 0]
    p_var = [15, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    total_var = np.append(p_var, bkg_mean_d / 2)
    p0_number = np.append(p0_number, bkg_mean_d)
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


def stacked_log_prob(
        p,
        obs,
        ms_model,
        log_mass,
        area,
        unmasked_fraction,
        obs_bin_pair,
        band_index,  # 0=g, 1=r, 2=i, 3=z, 4=y
        bkg_d_mean_func_list,
        bkg_d_std_func_list,
        obs_mid_bins,
        common_mid_bins,
        z,
        bkg_bins_num):
    """The sum of the logarithmic probability of the group of clusters. 

    Args:
        p (array-like): (parNum,) array. The parameter values of the MCMC.
        obs (array-like): The observation LF of the clusters.
        ms_model (array-like): The characeristic magnitude of the clusters.
        log_mass (array-like): log(mass_of_the_clusters/(M_sun/h)).
        area (ndarray): Area (dimensionless) of the clusters.
        unmaksed_fraction (array-like): The unmasked fraction of the clusters.
        obs_bin_pair (ndarray): (cluNum, 2) array. The lower/upper bounds of
            the cluster bins.
        band_index (array-like): The band each cluster uses. 1/2/3/4/5 stands
            for g/r/i/z/y.
        bkg_d_mean_func_list (array-like): A list of the mean background density
            (N/area.value) functions.
        bkg_d_std_func_list (array-like): A list of the std of background 
            density (N/area.value) functions. 
        obs_mid_bins (array-like): (cluNum, binNum) array. The center of each 
            bin for each cluster.
        common_mid_bins (array-like): (band, binNum) array. The center of bins 
            used in fitting for each band.
        z (array-like): Redshift of the clusters.
        bkg_bins_num (array-like): The number of bins in each band.
    
    Returns:
        float: The sum of the logarithmic probability of `p`.
    """
    cluster_num = len(ms_model)
    band_num = 5
    (A, B, alpha0, dm0, C, D, E, F, G) = p[:ARG_NUM]
    common_bkg_d_raw = np.array(
        p[ARG_NUM:]
    )
    common_bkg_d = []
    n = 0
    for i in range(5):
        common_bkg_d.append(common_bkg_d_raw[n:n + bkg_bins_num[i]])
        n += bkg_bins_num[i]
    phi = phi_model_mz(log_mass, z, A, B, C)
    alpha = alpha_model_mz(log_mass, z, alpha0, D, E)
    dm = dm_model_mz(log_mass, z, dm0, F, G)
    if len(np.unique([len(obs), len(ms_model), len(area)])) != 1:
        raise ValueError('obs, ms_model, area, common_bkg_mean_d and'
                         'common_bkg_std_d must have the same length')
    #common_mid_bins = np.mean(common_bin_pair, axis=2)
    # Goal: Generate bkg for each cluster
    all_bkg_mean_d = []
    all_bkg_std_d = []
    all_bkg_d = []
    bkg_d_func_list = []
    for i in range(band_num):
        if len(common_bkg_d[i]) == 0:
            bkg_d_func_list.append(nan_func)
        else:
            bkg_d_func_list.append(
                interpolate.interp1d(common_mid_bins[i], common_bkg_d[i]))
    for i in range(cluster_num):
        band = band_index[i]
        this_bkg_mean_d_func = bkg_d_mean_func_list[band]
        this_bkg_std_d_func = bkg_d_std_func_list[band]
        this_bkg_d_func = bkg_d_func_list[band]
        all_bkg_mean_d.append(this_bkg_mean_d_func(obs_mid_bins[i]))
        all_bkg_std_d.append(this_bkg_std_d_func(obs_mid_bins[i]))
        all_bkg_d.append(this_bkg_d_func(obs_mid_bins[i]))
    # all_bkg_d = [[c1b1, c1b2, ...], [c2b1, c2b2, ...], ...]
    all_bkg_mean = np.multiply(all_bkg_mean_d, area[:, np.newaxis])
    all_bkg_std = np.multiply(all_bkg_std_d, area[:, np.newaxis])
    all_bkg = np.multiply(all_bkg_d, area[:, np.newaxis])
    lp_list = [
        log_prior((ms_model[i] + dm[i], phi[i], alpha[i]), all_bkg[i],
                  all_bkg_mean[i], all_bkg_std[i]) for i in range(cluster_num)
    ]  # its p = (m_s, phi_s, alpha)
    ll_list = [
        log_likelihood((ms_model[i] + dm[i], phi[i], alpha[i]), obs[i],
                       all_bkg[i], obs_bin_pair[i], unmasked_fraction[i])
        for i in range(cluster_num)
    ]
    returnme = sum(lp_list) + sum(ll_list)
    if np.isnan(returnme).any() == True:
        return -np.inf
    return sum(lp_list) + sum(ll_list)


def get_sampler(
        obs_alllf,
        every_obs_bins,
        common_bkg_mean_d,  # shape = (grizy, bins)
        common_bkg_std_d,
        bkg_bins,
        band,
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
        **kwargs):
    """The resulting [sampler, state, is_used] of the MCMC.
    
    This function runs the MCMC and gives the result.`sampler` and `state` can 
    be referred from `emcee`. `is_used` is a Boolean array which represents 
    whether the bin is used in each band.

    Args:
        obs_alllf (ndarray): (cluNum, binNum) array. The observational LF
            (without background subtraction) of each cluster.
        every_obs_bins (ndarray): (cluNum, binBoundNum) array. The boundaries of
            the bins of the observaiontal LF for each cluster.
        common_bkg_mean_d (list): (band, binNum) list. The common mean
            background density (N/area) for each band.
        common_bkg_std_d (list): (band, binNum) list. The common std of
            background density (N/area) for each band.
        bkg_bins (list): (band, binBoundNum) list. The bin boundaries of the
            background LF for each band.
        band (ndarray): The band used for each cluster. 'g'/'r'/'i'/'z'/'y'
            represents g/r/i/z/y band.
        ms_model (array-like): The model values of characteristic magnitude of
            each cluster.
        log_mass (array-like): log(mass_of_the_clusters/(M_sun/h)) for each 
            cluster.
        area (ndarray): Area (dimensionless) of the clusters.
        z (array-like): Redshift of the clusters.
        unmasked_fraction (array-like): The unmasked fraction of the clusters.
        nwalkers (int or str): Number of walkers. If 'auto', this number will
            be automatically chosen (2.5 * number of parameters).
        step (int): The number of steps the MCMC will go through.
        p0 (ndarray or None): The initial parameter values for the MCMC. If
            `None`, it will be automaticlly generated (see `find_init_args()`).
        cpu_num (int): The number of cpu used in the MCMC.
        progress (bool): Whether the progress is shown.

    Returns:
        list: [sampler, state, is_used]. The first/second elements can be
            referred from `emcee`. `is_used` is a (band, binNum) Boolean array 
            which represents whether the bin is used in each band.
    """
    cnum = len(obs_alllf)
    band_num = 5
    is_used = np.full((5, len(BINS) - 1), False)
    band_name = np.array(['g', 'r', 'i', 'z', 'y'])
    bkg_mean_d_4fit = []
    bkg_std_d_4fit = []
    common_bins_dim = 0
    common_bin_pair = []
    for i in range(len(band_name)):
        cindex = np.where(band == band_name[i])[0]
        if len(cindex) == 0:
            bkg_mean_d_4fit.append([])
            bkg_std_d_4fit.append([])
            common_bin_pair.append([])
            continue
        this_all_obs_bins = []
        for index in cindex:
            this_all_obs_bins.append(every_obs_bins[index])
        this_obs_min_mag = np.min(this_all_obs_bins)
        this_obs_max_mag = np.max(this_all_obs_bins)
        lowest_used_index = np.where(bkg_bins < this_obs_min_mag)[0][-1]
        highest_used_index = np.where(bkg_bins < this_obs_max_mag)[0][-1]
        is_used[i][range(lowest_used_index, highest_used_index + 1)] = True
        this_used_bkg_index = np.where(is_used[i])[0]
        bkg_mean_d_4fit.append(common_bkg_mean_d[i][is_used[i]])
        bkg_std_d_4fit.append(common_bkg_std_d[i][is_used[i]])
        common_bins_dim += len(this_used_bkg_index)
        common_bin_pair.append([[
            bkg_bins[this_used_bkg_index[i]],
            bkg_bins[this_used_bkg_index[i] + 1]
        ] for i in range(len(this_used_bkg_index))])
    obs_bin_pair = [[[every_obs_bins[i][j], every_obs_bins[i][j + 1]]
                     for j in range(len(every_obs_bins[i]) - 1)]
                    for i in range(cnum)]
    obs_mid_bins = np.mean(obs_bin_pair, axis=2)
    ndim = common_bins_dim + ARG_NUM  # A, B, alpha0, dm, C, D, E, F, G
    print('ndim =', ndim)
    if nwalkers == 'auto':
        nwalkers = int(ndim * 2.5)
    if p0 is None:
        p0 = find_ini_args(np.hstack(bkg_mean_d_4fit), nwalkers, ndim)
    band_index = []
    for i in range(cnum):
        band_index.append(np.where(band_name == band[i])[0][0])
    common_mid_bins = []
    bkg_d_mean_func_list = []
    bkg_d_std_func_list = []
    for i in range(band_num):
        if len(common_bin_pair[i]) == 0:
            common_mid_bins.append([])
            bkg_d_mean_func_list.append(nan_func)
            bkg_d_std_func_list.append(nan_func)
        else:
            common_mid_bins.append(np.mean(common_bin_pair[i], axis=1))
            bkg_d_mean_func_list.append(
                interpolate.interp1d(common_mid_bins[i], bkg_mean_d_4fit[i]))
            bkg_d_std_func_list.append(
                interpolate.interp1d(common_mid_bins[i], bkg_std_d_4fit[i]))
    bkg_bins_num = []
    for i in range(band_num):
        bkg_bins_num.append(len(bkg_mean_d_4fit[i]))
    partial_log_prob = functools.partial(
        stacked_log_prob,
        obs=obs_alllf,
        ms_model=ms_model,
        log_mass=log_mass,
        area=area,
        unmasked_fraction=unmasked_fraction,
        obs_bin_pair=obs_bin_pair,
        band_index=band_index,
        bkg_d_mean_func_list=bkg_d_mean_func_list,
        bkg_d_std_func_list=bkg_d_std_func_list,
        obs_mid_bins=obs_mid_bins,
        common_mid_bins=common_mid_bins,
        z=z,
        bkg_bins_num=bkg_bins_num)
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
    return [sampler, state, is_used]


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


def easy_mcmc(efeds_index, efeds, hsc, rd_result):  # band = g r i z y
    new_efeds = efeds[efeds['low_cont_flag']
                      & (efeds['unmasked_fraction'] > 0.6)]
    z = new_efeds[efeds_index]['Z_BEST_COMB'].value
    band = np.array(fit_band(z))
    cnum = len(efeds_index)
    band_num = 5
    if len(np.atleast_1d(band)) == 1:
        band = np.full(len(efeds_index), band)
    ms_model = np.array(
        [efeds[band[i] + '_cmag'][efeds_index[i]] for i in range(cnum)])
    every_obs_bins = [
        proper_mag_bins(ms_model[i], 2, 2, 0.2) for i in range(cnum)
    ]
    obs_alllf = [
        index2fl(efeds['galaxy_index'][efeds_index[i]], hsc,
                 band[i] + 'mag_cmodel', every_obs_bins[i])
        for i in range(cnum)
    ]
    obs_alllf_corrected = [
        index2fl(efeds['galaxy_index'][efeds_index[i]], hsc,
                 band[i] + 'mag_cmodel', 
                 every_obs_bins[i]) 
                 / efeds['unmasked_fraction'][efeds_index[i]]
        for i in range(cnum)
    ]
    common_bkg_mean_d = rd_result['mean_lf_d'].to(u.arcmin**-2).value
    common_bkg_std_d = rd_result['std_lf_d'].to(u.arcmin**-2).value
    bkg_bins = BINS
    log_mass = new_efeds[efeds_index]['median_500c_lcdm'].value
    area = new_efeds[efeds_index]['area'].to(u.arcmin**2).value
    unmasked_fraction = new_efeds[efeds_index]['unmasked_fraction'].value
    returnme = {
        'obs_alllf': obs_alllf,
        'obs_alllf_corrected': obs_alllf_corrected,
        'every_obs_bins': every_obs_bins,
        'common_bkg_mean_d': common_bkg_mean_d,
        'common_bkg_std_d': common_bkg_std_d,
        'bkg_bins': bkg_bins,
        'band': band,
        'ms_model': ms_model,
        'log_mass': log_mass,
        'area': area,
        'z': z,
        'index': efeds_index,
        'unmasked_fraction': unmasked_fraction
    }
    return returnme


def value2bkg_d(value, is_used, bins=BINS):
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
