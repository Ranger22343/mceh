"""
Last changed date: 2024 Oct 7

The main code to do the MCMC.
"""
from . import utility as ut
from . import fitting

import numpy as np

print('Call manual() to get the manual.')
def manual():
    print(
    "\nTo process the MCMC. Follow these steps:\n"
    "1. Import must-need data:\n"
    ">>> efeds, bkg_all = ut.init('efeds', 'bkg_all')\n\n"

    "2. Group the galaxies:\n"
    ">>> efeds_group = fitting.make_group(efeds, bkg_all['z_list'], [13, 14, 15])\n"
    "bkg_all['z_list'] is the boundary of redshift selection while [13, 14, 15] is for the mass."
    "efeds_group[i][j] means the indicies of galaxies in ith redshift group and jth mass group.\n\n"

    "3. Get the information of background LF density by inputing the indices of galaxy:\n"
    "If you want to choose the galaxies in ith redshift group and jth mass group:\n"
    ">>> common_bkg_mean_d, common_bkg_std_d = get_common_bkg_d(efeds_group[i][j], efeds, bkg_all)\n\n"

    "4. Start the MCMC:\n"
    ">>> sampler, state, zero_index = fitting.get_sampler(\n"
    "    efeds_group[i][j], common_bkg_mean_d, common_bkg_std_d,\n" 
    "    efeds, nwalkers = 150, step = 15000, cpu_num = 8\n"
    "    )\n"
    "The nwalkers, step, cpu_num should be self-explanatory. sampler is the sampler for emcee, "
    "state is the sample of the last step and zero_index is a list of deleted indices of the magnitude bin"
    "due to the zero-detected background LF."
    )

def get_common_bkg_d(galaxy_index, efeds, bkg_all):
    efeds_ = efeds[galaxy_index]
    
    bkg_mean_d, bkg_std_d = fitting.bkg_info(np.atleast_1d(efeds_['z']), bkg_all['z_list'], 
                                             bkg_all['mean'], bkg_all['std'])
    common_bkg_mean_d = np.mean(np.atleast_2d(bkg_mean_d), axis = 0)
    common_bkg_std_d = ((np.sum(np.atleast_2d(bkg_std_d) ** 2, axis = 0)) ** 0.5 
                        / len(bkg_std_d)) # mean after ^ 0.5
    return common_bkg_mean_d, common_bkg_std_d

