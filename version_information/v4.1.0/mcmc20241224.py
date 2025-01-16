import mceh.utility as ut
import numpy as np
from mceh import fitting
from scipy.interpolate import interp1d

ut.printt('mcmc20241224.py')
fitting.print = ut.printt
ut.printt('Start data loading')
efeds, hsc = ut.init('efeds', 'hsc')
zmbins = ut.pickle_load('data/zmbins20241108.pickle')
rd_result = ut.pickle_load('data/bkg_lf20241111.pickle')

efeds = efeds[(efeds['low_cont_flag']) & (efeds['unmasked_fraction'] > 0.6)]

zlen = len(zmbins['mbins'])
ut.printt('Start MCMC')
for i in range(zlen):
    mlen = len(zmbins['mbins'][i]) - 1
    for j in range(mlen):
        mcdict = fitting.easy_mcmc(i, j, efeds, hsc, rd_result, zmbins)
        sampler, state, is_used = fitting.get_sampler(cpu_num = 32, **mcdict)
        result = {'mcmc_dict': mcdict, 'sampler': sampler,
                  'state': state, 'is_used': is_used}
        ut.pickle_dump(result, f'result/mcmc20241224_{i}_{j}.pickle')
        ut.printt(f'Finish z:{i + 1}/{zlen}, m:{j + 1}/{mlen}')
ut.printt('All finished')