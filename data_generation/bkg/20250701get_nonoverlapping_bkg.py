import mceh.utility as ut
import mceh.find_luminosity as fl
import astropy.units as u
import numpy as np
from astropy.table import QTable

rd_num = 3000

original_efeds = QTable.read('data/modified_efeds_ver9.fits')
rd = QTable.read('data/modified_random_ver2.fits')
hsc = QTable.read('data/modified_hsc_ver4.fits')
efeds = original_efeds[original_efeds['low_cont_flag']
                       & (original_efeds['unmasked_fraction'] > 0.6)]
median_r = np.median(efeds['r500c_lcdm'].value) * efeds['r500c_lcdm'].unit
fl.print = ut.printt
rd_ra, rd_dec, rd_coverage = fl.nonoverlapping_random_point(
    rd_num, median_r,
    min(rd['ra'].value) * u.deg,
    max(rd['ra'].value) * u.deg,
    min(rd['dec'].value) * u.deg,
    max(rd['dec'].value) * u.deg,
    original_efeds['RA_CORR'],
    original_efeds['DEC_CORR'],
    original_efeds['r500c_lcdm'],
    rd_data=rd,
    coverage=60,
    progress=True
    )
ut.printt('Finish finding points')
actual_rd_num = len(rd_ra)
rd_hsc_index = []
for i in range(actual_rd_num):
    this_index = fl.index_within_angular_radius(hsc['ra'], hsc['dec'], rd_ra[i],
                                                rd_dec[i], median_r)
    rd_hsc_index.append(this_index)
    ut.printt(f'Finish {i + 1}/{actual_rd_num} index finding')

result = {'ra': rd_ra, 'dec': rd_dec, 'unmasked_fraction': rd_coverage,
          'hsc_index': rd_hsc_index, 'r': median_r}

ut.pickle_dump(result, 'result/20250701nonoverlapping_bkg.pickle')
ut.printt('All finished')
