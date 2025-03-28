from ... import find_luminosity as fl
from astropy.table import QTable

if __name__ == '__main__':
    hsc = QTable.read('data/modified_hsc_ver3.fits')
