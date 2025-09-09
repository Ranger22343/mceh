import mceh
from astropy.table import QTable
import mceh.model.group_mass_phi_guassian as fitting
import numpy as np
from astropy.io import fits
import tqdm

mceh.printt('Start')
cpu_num = 8
rd_result = mceh.pickle_load('data/20250813bkg_result.pickle')
hsc = QTable.read('data/modified_hsc_ver5.fits')
efeds = QTable.read('data/modified_efeds_ver10.fits')

efeds_group = [np.where((fitting.ZBINS[i] <= efeds['Z_BEST_COMB'])
                        & (efeds['Z_BEST_COMB'] < fitting.ZBINS[i + 1]))[0]
                        for i in range(len(fitting.ZBINS) - 1)]
for i in range(len(efeds_group)):
    these_efeds = efeds[efeds_group[i]]
    is_sr = these_efeds['low_cont_flag'] & (these_efeds['unmasked_fraction']
                                            > 0.6)
    efeds_group[i] = efeds_group[i][is_sr]

efeds_group[-2] = np.append(efeds_group[-2], efeds_group[-1][0]) # Last group only contains 1 cluster so I move it to the second one
efeds_group.pop()

all_mcmc_dict = [
    fitting.easy_mcmc(efeds_group[i],
                      efeds,
                      hsc,
                      rd_result,
                      ) for i in range(len(efeds_group))
]

all_result = []
for i in range(len(efeds_group)):
    result = {}
    sampler, state = fitting.get_sampler(**all_mcmc_dict[i], cpu_num=cpu_num, nwalkers=32)
    result['mcmc_dict'] = all_mcmc_dict[i]
    result['sampler'] = sampler
    result['state'] = state
    result['efeds_index'] = efeds_group[i]
    all_result.append(result)
    mceh.printt(f'Finished {i + 1}/{len(efeds_group)}')

mceh.printt('All samplers and states have been obtained')
mceh.pickle_dump(all_result, 'result/20250903mass_phi_guassian_fit.pickle')
mceh.printt('All finished')