import astropy.io.fits as fits
from astropy.table import QTable
#  125 < ra < 146
print('Start', flush=True)
data = QTable.read('/array/users/profoak/dataDir/hsc/s23b/spring/spring_obj.fits')
data = data[(data['ra'] > 125) & (data['ra'] < 146)]
data.write('/array/users/hylin/hsc_s23b_spring.fits', overwrite=True)
print('Finished object selection', flush=True)
data = QTable.read('/array/users/profoak/dataDir/hsc/s23b/spring/spring_random.fits')
data = data[(data['ra'] > 125) & (data['ra'] < 146)]
data.write('/array/users/hylin/hsc_s23b_spring_random.fits', overwrite=True)
print('Finished random selection', flush=True)