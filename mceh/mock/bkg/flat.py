from ... import find_luminosity as fl
from astropy import table
import astropy.units as u
import numpy as np
import copy
import tqdm


def generate_mock_hsc(hsc, rd, efeds, seed):
    """Generate mock background HSC data
    
    It picks random positions on the sky (but with the same galaxiy density as 
    HSC after removing the cluster part) as the mock galaxy positions, and 
    randomly pick galaxies in HSC data as the grizy magnitude of the mock ones.

    args:
        hsc (QTable): HSC data
        rd (QTable): random data using for calculating masking
        efeds (QTable): eFEDS data
        seed (int): A random seed for generation
    
    returns:
        (QTable): the resulting mock background HSC catalog
    """
    low_ra, high_ra = min(hsc['ra'].value) * u.deg, max(
        hsc['ra'].value) * u.deg
    low_dec, high_dec = min(hsc['dec'].value) * u.deg, max(
        hsc['dec'].value) * u.deg

    # Remove the cluster parts from hsc
    clean_hsc = copy.deepcopy(hsc)
    clean_rd = copy.deepcopy(rd)

    for i in tqdm.tqdm(range(len(efeds))):
        this_efeds = efeds[i]
        hsc_in_index = fl.index_within_angular_radius(clean_hsc['ra'],
                                                      clean_hsc['dec'],
                                                      this_efeds['RA_CORR'],
                                                      this_efeds['DEC_CORR'],
                                                      this_efeds['r500c_lcdm'])
        rd_in_index = fl.index_within_angular_radius(clean_rd['RA'],
                                                     clean_rd['DEC'],
                                                     this_efeds['RA_CORR'],
                                                     this_efeds['DEC_CORR'],
                                                     this_efeds['r500c_lcdm'])
        clean_hsc.remove_rows(hsc_in_index)
        clean_rd.remove_rows(rd_in_index)

    hsc_density = len(clean_hsc) / (len(clean_rd) / 100 * u.arcmin**2)
    hsc_num = int(
        (hsc_density * calculate_area(low_ra, high_ra, low_dec, high_dec)).to(
            u.dimensionless_unscaled).value)  # Number of fake galaxies
    np.random.seed(seed=seed)
    fake_hsc = table.QTable()
    fake_hsc_ra = np.random.uniform(
        low=low_ra.value, high=high_ra.value, size=hsc_num) * u.deg
    fake_hsc['ra'] = fake_hsc_ra
    fake_hsc_dec = (np.arcsin(
        np.random.uniform(low=low_dec.to(u.rad).value,
                          high=high_dec.to(u.rad).value,
                          size=hsc_num)) * u.rad).to(u.deg)
    fake_hsc['dec'] = fake_hsc_dec
    correspond_index = np.random.randint(len(clean_hsc), size=hsc_num)
    fake_hsc['correspond_index'] = correspond_index
    fake_hsc['object_id'] = hsc['object_id'][correspond_index]
    mag_cmodel = [
        clean_hsc['gmag_cmodel'], clean_hsc['rmag_cmodel'],
        clean_hsc['imag_cmodel'], clean_hsc['zmag_cmodel'],
        clean_hsc['ymag_cmodel']
    ]
    mag_err = [
        clean_hsc['gmag_err'], clean_hsc['rmag_err'], clean_hsc['imag_err'],
        clean_hsc['zmag_err'], clean_hsc['ymag_err']
    ]
    # zmag has no masked?
    fake_mag_cmodel = []
    for b in range(5):
        this_mag = mag_cmodel[b][correspond_index]
        this_err = mag_err[b][correspond_index]
        fake_mag_cmodel.append(
            np.ma.masked_invalid(np.random.normal(this_mag, this_err)))
        fake_mag_cmodel[b].fill_value = np.nan

    mag_name = [
        'gmag_cmodel', 'rmag_cmodel', 'imag_cmodel', 'zmag_cmodel',
        'ymag_cmodel'
    ]
    for i in range(5):
        fake_hsc[mag_name[i]] = fake_mag_cmodel[i]
    return fake_hsc

def calculate_area(ra_min, ra_max, dec_min, dec_max):
    ra_width = ra_max.to(u.rad).value - ra_min.to(u.rad).value
    dec_width = (np.sin(dec_max.to(u.rad).value) -
                 np.sin(dec_min.to(u.rad).value))
    return (ra_width * dec_width * u.rad**2).to(u.deg**2)
