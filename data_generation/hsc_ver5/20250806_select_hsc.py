import astropy.io.fits as fits
from astropy.table import QTable
import astropy.units as u
import numpy as np
from mceh.utility import printt
#  125 < ra < 146
printt('Start')
data = QTable.read(
    '/array/users/profoak/dataDir/hsc/s23b/spring/spring_obj.fits')
printt('check point 1')
data = data[(data['ra'] > 125) & (data['ra'] < 146)]
printt('check point 2')
data = data[data['science_flag']]
printt('check point 3')
data['ra'] *= u.deg
data['dec'] *= u.deg
data.remove_column('science_flag')
data = data[data['iclassification_extendedness'] > 0.9]
data.remove_column('iclassification_extendedness')
np.random.seed(20250627)
data['random'] = np.random.rand(len(data))
data.write('/home/hylin/workspace/data/modified_hsc_ver5.fits')
printt('Finished object selection')


data = QTable.read('/array/users/profoak/dataDir/hsc/s23b/spring/spring_random.fits')
data = data[(data['ra'] > 125) & (data['ra'] < 146)]
data = data[data['science_flag']]
data['ra'] *= u.deg
data['dec'] *= u.deg
data.write('/home/hylin/workspace/data/modified_random_ver3.fits')
printt('Finished random selection')
