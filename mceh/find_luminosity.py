import random
from astropy.table import Table
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import coordinates as coord
from astropy.io import fits
from astropy.io import ascii
from astropy import units as u
from matplotlib import pyplot as plt
from itertools import accumulate
import copy
import lzma
from .rsmodel import outputModels

outputModels.print = lambda *args, **kwargs: None

def index_within_angular_radius(sat_ra, sat_dec, center_ra, center_dec,
                                angular_radius):
    """Find the index of data within a given circle on the sky.

    Return a list of object indices which locate within a circle at 
    [`center_ra`, `center_dec`] with a radius `angular_radius`.

    Args:
        sat_ra (Quantity): RA of objects whose indices are you want to find.
        sat_dec (Quantity): DEC of objects whose indices are you want to find.
        center_ra (Quantity): RA of the cirlce center.
        center_dec (Quantity): DEC of the circle center.
        angular_radius (Quantity): Radius of the circle.

    Returns:
        list: List of indicies whose corresponding positions are within the 
            cirlce.
    """
    clean_index = clean_skypoint(center_ra, center_dec, angular_radius, sat_ra, 
                                 sat_dec, return_index=True)
    clean_sat_ra, clean_sat_dec = sat_ra[clean_index], sat_dec[clean_index]
    sat_coord = coord.SkyCoord(clean_sat_ra, clean_sat_dec)
    cen_coord = coord.SkyCoord(center_ra, center_dec)
    angular_distance = cen_coord.separation(sat_coord)
    in_radius = angular_distance <= angular_radius
    returnme = clean_index[in_radius]
    return returnme


def is_within_angular_radius(sat_ra, sat_dec, center_ra, center_dec,
                             angular_radius):
    """Find the index of data within a given circle on the sky.

    Return a Boolean list states if the input object locates within a circle at 
    [`center_ra`, `center_dec`] with a radius `angular_radius`.

    Args:
        sat_ra (Quantity): RA of objects whose indices are you want to find.
        sat_dec (Quantity): DEC of objects whose indices are you want to find.
        center_ra (Quantity): RA of the circle center.
        center_dec (Quantity): DEC of the circle center.
        angular_radius (Quantity): Radius of the circle.
    
    Returns:
        list: Boolean list which states whether the corresponding objects are
            within the circle.
    """
    sat_coord = coord.SkyCoord(sat_ra, sat_dec)
    cen_coord = coord.SkyCoord(center_ra, center_dec)
    angular_distance = cen_coord.separation(sat_coord)
    in_radius = angular_distance <= angular_radius
    return in_radius


def satellites_around_centers(sat_data, cen_ra, cen_dec, cen_r):
    """Remove the data which is not in the circles with given positions/radii.
    
    The element in `sat_data` will be removed if it does not locate in the
    circles at [`cen_ra`, `cen_dec`] with angular radii `cen_r` on the sky.
    Note that there must be keys 'RA' and 'DEC' in `sat_data` with units.

    Args:
        sat_data (QTable): The table with elements which you want to delete if
            not within the circles. It must contains the keys `RA` and `DEC`.
        cen_ra (Quantity): RA of the circles.
        cen_dec (Quantity): DEC of the circles.
        cen_r (Quantity): Angular radii of the cirlces.

    Returns:
        QTable: The table after the deletion.
    """
    qualified_index = []
    cen_num = len(cen_ra)
    for i in range(cen_num):
        qualified_index.extend(
            index_within_angular_radius(sat_data['RA'], sat_data['DEC'],
                                        cen_ra[i], cen_dec[i], cen_r[i]))
        print(f'finish {i+1}/{cen_num}')
    qualified_index = np.unique(qualified_index)
    sat_data = sat_data[qualified_index]
    return sat_data


def quantity_function(quantity_name, center_position, satellites_data,
                      angular_radius, bins):
    """Distribution of a given quantity of the satellites around given centers.
    
    It will find the objects in `satellites_data` which are around the 
    `center_position` within raii `angular_radius` on the sky, and uses the key
    `quantity_name` of the objects to give its hisogram with `bins`.

    Args:
        quantity_name (str): Name of the quantity which contructs the
            distribution. It should be containd in `satellites_data`.
        center_position (array Quantity): [RA, DEC] of the centers.
        satellites_data (QTable): Data of the satellites. Must contains keys
            'RA', 'DEC' and `quantity_name`.
        angular_radius (Quantity): Radii of the circles around the centers.
        bins (int or array-like): Bins of the histogram. It can be the number of
            bins or their boundaries.

    Returns:
        hist (array): Histogram of the quantity.
        bins_edge (array): Edges of the distribution bins.  
    """
    sat_ra, sat_dec = satellites_data['RA'], satellites_data['DEC']
    cen_ra, cen_dec = center_position[0], center_position[1]
    index_r = index_within_angular_radius(sat_ra, sat_dec, cen_ra, cen_dec,
                                          angular_radius)
    satellites_data = satellites_data[index_r]
    satellites_data = satellites_data[~np.isnan(satellites_data[quantity_name]
                                                )]
    satellites_data = satellites_data[~np.isinf(satellites_data[quantity_name]
                                                )]
    hist, hist_bins = np.histogram(satellites_data[quantity_name], bins=bins)
    return hist, hist_bins


def count_number(position, radius, sat_data):
    '''Count the number of objects in given circles.
    
    Count the number of objects of `sat_data` within the cirlces with `position`
    and `radius` on the sky. The position informaiton should be contained in 
    `sat_data`.

    Args:
        position (array Qunatity): [RA, DEC] of the circles.
        radius (Quantity): Radii of the circles.
        sat_data (QTable): Table of the objects to be counted. Must contains
            keys 'RA' and 'DEC'.
    
    Returns:
        int: Number of objects in `sat_data` which locates within the cirlces.

    '''
    cen_ra, cen_dec = position
    sat_ra, sat_dec = clean_skypoint(cen_ra, cen_dec, radius, sat_data['ra'], 
                                     sat_data['dec'])
    number = len(
        index_within_angular_radius(sat_ra, sat_dec, cen_ra, cen_dec, radius))
    return number


def index_between_angular_radius(data, pos, r_in, r_out):
    """Find which index has distance to center between `r_in` and `r_out`.

    Return the index whose data has distance to the center between `r_in` 
    and `r_out`.

    Args
    ----
    data: satellite data with 'ra' and 'dec' keys.
    pos: position of the center which has the form [ra, dec]. 
    r_in: inner radius.
    r_out: outer radius.
    
    Returns
    -------
    An index array whose data has distance between `r_in` and `r_out`.
    """
    sat_ra, sat_dec = data['ra'], data['dec']
    distance = coord.angular_separation(
        pos[0] * np.pi / 180, pos[1] * np.pi / 180, sat_ra * np.pi / 180,
        sat_dec * np.pi / 180) * 180 / np.pi
    index = np.where((r_in <= distance) & (distance <= r_out))[0]
    return index


def local_bkg(pos,
              r_in,
              r_out,
              bins,
              area_in,
              hsc,
              random,
              quantity='imag_cmodel',
              bkg_coverage=False):
    """The local background of the cluster.
    
    Return the local background of quantity `quantity`. If coverage == True,
    also return the covered percent (masked area / actual area).

    Args
    ----
    pos: position of the center.
    r_in: inner radius of the local background.
    r_out: outer radius of the local background.
    bins: the bin bounds which sort `quantity`.
    area_in: the area in interested.
    hsc: hsc data.
    random: random data.
    bkg_coverage: if True, return a list with [original result, covered percent].

    Returns
    -------
    An array which indicates the local background with `bins`. If
    bkg_coverage == True, returns a list whose the first element is the local
    backgroudn and the second one is the covered percentof the background.
    """
    hsc_i = index_between_angular_radius(hsc, pos, r_in, r_out)
    hsc = hsc[hsc_i]
    bkg = np.histogram(hsc[quantity], bins)[0]
    actual_bkg_area = (r_out**2 - r_in**2) * np.pi
    bkg = bkg / actual_bkg_area * area_in  # normalized background
    if bkg_coverage == True:
        cover_num = index_between_angular_radius(random, pos, r_in, r_out)
        cover_area = cover_num / 100 / 3600
        cover_percent = cover_area / actual_bkg_area
        return [bkg, cover_percent]
    return bkg


def unmasked_area(ra, dec, r, rd):
    # Calculate the unmasked area of a given circle.
    pos = [ra, dec]
    num = count_number(pos, r, rd)
    return num / 100 * u.arcmin**2


def hsc_photoz(z, hsc):  # Apply photo-z selction on hsc based on the redshift.
    return hsc[(hsc['zbound'][:, 0] < z) & (hsc['zbound'][:, 1] > z)]


def nonoverlapping_random_point(number,
                                r,
                                ra_min,
                                ra_max,
                                dec_min,
                                dec_max,
                                prevent_ra,
                                prevent_dec,
                                prevent_r,
                                try_limit=10000,
                                rd_data=None,
                                coverage=0.,
                                coverage_returned_none=False,
                                progress=False,
                                step=False):
    """Generate a given number of non-overlapping random points on the sky.
    
    `number` of points with radii `r` will be generated without touching each 
    others, avoiding circular area with positions [`prevent_ra`, 
    `prevent_dec`] and radii `prevent_r`. Note that the random RA are uniformly
    chose between [`ra_min`, `ra_max`] while the DEC are [sin(`dec_min`), 
    sin(`dec_max`)]. You can choose the percentage threshold of the unmasked
    area by setting `coverage` to a number. 

    Args:
        number (int): Number of random points to be generated.
        r (scalar Quantity): The radius of the random points.
        ra_min (scalar Quantity): The minimal RA for generating random points.
        ra_max (scalar Quantity): The maximal RA for generating random points.
        dec_min (scalar Quantity): The minimal DEC for generating random points.
        dec_max (scalar Quantity): The maximal DEC for generating random points.
        prevent_ra (Quantity): The center RA of the preventing areas.
        prevent_dec (Quantity): The center DEC of the preventing areas.
        prevent_r (Quantity): The radii of the preventing areas.
        try_limit (int): Maximal try of getting a qualified point.
        rd_data (QTable): Random data for calculating the area. Must contains
            keys 'RA' and 'DEC'.
        coverage (float): The lowest percentage of unmasked area to pass. If
            != 0, this function will also return the unmasked coverage.
        coverage_returned_none(bool): Whether the unmasked coverage of each center
            should be returned `None`. Change it to `True` will save time only 
            if `coverage` equals the default value (i.e. 0).
        progress (bool): Whether it should print the progress.
        step (bool): Whether it should print the finish of each try.
    
    Returns:
        ((`number`,) Quantity, (`number`,) Quantity, (`number`,) ndarray)
        The RA/DEC/unmasked_coverage of the random points.
    """
    pass_ra = []
    pass_dec = []
    pass_coverage = []
    for i in range(number):
        prevent_coord = coord.SkyCoord(prevent_ra, prevent_dec)
        tried = 0
        while tried < try_limit:
            if step == True:
                print(f'try {tried+1}/{try_limit}')
            # Generate a random point.
            rd_ra = random.uniform(
                ra_min.to(u.deg).value,
                ra_max.to(u.deg).value) * u.deg
            rd_dec = np.arcsin(
                random.uniform(np.sin(dec_min.to(u.rad).value),
                               np.sin(dec_max.to(
                                   u.rad).value))) / np.pi * 180 * u.deg
            rd_coord = coord.SkyCoord(rd_ra, rd_dec)

            sep = rd_coord.separation(prevent_coord)
            if np.all((sep > prevent_r + r)):
                if coverage_returned_none == False:
                    area = 2 * np.pi * (1 - np.cos(r)) * u.rad**2
                    unm_area = unmasked_area(rd_ra, rd_dec, r, rd_data)
                    cov = (unm_area / area).to(
                        u.dimensionless_unscaled).value
                    if cov < coverage / 100:
                        tried += 1
                        continue
                    pass_coverage.append(cov)
                pass_ra.append(rd_ra)
                pass_dec.append(rd_dec)
                prevent_ra = np.append(prevent_ra, rd_ra)
                prevent_dec = np.append(prevent_dec, rd_dec)
                prevent_r = np.append(prevent_r, r)
                break
            tried += 1
        if tried >= try_limit:
            print(f'WARNING: Exceed the trying limit at n = {len(pass_ra)}.'
                  ' Return the left coordinates.')
            break
        if progress == True:
            print(f'Finish {i+1}/{number}')
    pass_ra = u.quantity.Quantity(pass_ra)
    pass_dec = u.quantity.Quantity(pass_dec)
    if coverage_returned_none == True:
        pass_coverage = np.full(len(pass_ra.value), None)
    return pass_ra, pass_dec, pass_coverage


def square_bound(ra, dec, r):
    ddec = r
    if np.abs(dec + r) < np.abs(dec - r):
        r = -r
    max_dra = (np.arccos((np.cos(r) - 1) / (np.cos(dec + r)**2) + 1))
    min_ra, max_ra = ra - max_dra, ra + max_dra
    min_dec, max_dec = dec - ddec, dec + ddec
    return min_ra, max_ra, min_dec, max_dec


def clean_skypoint(ra, dec, r, data_ra, data_dec, return_index=False):
    min_ra, max_ra, min_dec, max_dec = square_bound(ra, dec, r)
    is_ra = (data_ra > min_ra) & (data_ra < max_ra)
    is_dec = (data_dec > min_dec) & (data_dec < max_dec)
    is_return = is_ra & is_dec
    if return_index:
        return np.where(is_return)[0]
    return data_ra[is_return], data_dec[is_return]


@np.vectorize
def cmag(z, band_name):  # characteristic magnitude by the model
    # band_name: ex. 'hsc_i'
    return outputModels.printRSmodel('data/bc03_rs_zfp3d0_tau_p0d4.fits',
                                     [band_name], z)[1][band_name][4]


def random_point_within_radius(number,
                               ra,
                               dec,
                               r,
                               try_limit=10000):
    """Generate a given number of random points within a given radius.
    
    `number` of points with radii `r` will be generated within the circle with
    center [`ra`, `dec`] and radius `r`. The random RA are uniformly chose 
    between [`ra_min`, `ra_max`] while the DEC are [sin(`dec_min`), 
    sin(`dec_max`).

    Args:
        number (int): Number of random points to be generated.
        ra (scalar Quantity): RA of the center of the circle.
        dec (scalar Quantity): DEC of the center of the circle.
        r (scalar Quantity): The radius of the circle.
        try_limit (int): Maximal try of getting a qualified point.
    
    Returns:
        ((`number`,) Quantity, (`number`,) Quantity)
        The RA/DEC of the random points.
    """
    pass_ra = []
    pass_dec = []
    ra_min, ra_max, dec_min, dec_max = square_bound(ra, dec, r)
    for i in range(number):
        tried = 0
        while tried < try_limit:
            # Generate a random point.
            rd_ra = random.uniform(
                ra_min.to(u.deg).value,
                ra_max.to(u.deg).value) * u.deg
            rd_dec = np.arcsin(
                random.uniform(np.sin(dec_min.to(u.rad).value),
                               np.sin(dec_max.to(
                                   u.rad).value))) / np.pi * 180 * u.deg
            rd_coord = coord.SkyCoord(rd_ra, rd_dec)
            sep = rd_coord.separation(coord.SkyCoord(ra, dec))
            if sep < r:
                pass_ra.append(rd_ra)
                pass_dec.append(rd_dec)
                break
            tried += 1
        if tried >= try_limit:
            print(f'WARNING: Exceed the trying limit at n = {len(pass_ra)}.'
                  ' Return the left coordinates.')
            break
    pass_ra = u.quantity.Quantity(pass_ra)
    pass_dec = u.quantity.Quantity(pass_dec)
    return pass_ra, pass_dec