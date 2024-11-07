#!/usr/bin/env python

# basics
import math
import os
import numpy             as np
import scipy.stats       as stats
import scipy.special     as special
import scipy.interpolate as interpolate
import scipy.optimize    as optimize
import tempfile

# astropy
import astropy.io.fits   as pyfits

# utilts
from . import utilts

# ---
# output rs model
# ---
def printRSmodel(path2tbu, filter_names, zcl, **kwargs):
    """

    Parameters
    ----------------------------------------------------
    path2tbu: str or pyfits.BinTableHDU
        abs path to the fits table created by generateRS.
        If not, it has to be a BinTableHDU.
    filter_names: list containing the names of filters
        list of the filter names, which have to be used in tbu
    zcl: float
        The redshift of the outputed model
    verbose: bool
        Whether to print results on the screen
    """
    # kwargs
    verbose             = kwargs.get(   "verbose",  True)
    # sanitize
    zcl                 = float(        zcl )
    # filter_name has to be list
    assert isinstance(filter_names, list)    or   isinstance(filter_names, np.ndarray)
    # extract the column names
    if    isinstance(path2tbu, str) and os.path.isfile(path2tbu):
        tbudata         = pyfits.getdata(path2tbu, ext = 1)
    elif  isinstance(path2tbu, pyfits.BinTableHDU):
        tbudata         = path2tbu.data
    else:
        msg = 'The path2tbu is not correct.'
        raise IOError(msg)
    col_names           = tbudata.names
    # redshift
    reds                = tbudata["redshift"]
    # Lstar_offset
    Lstar_offset_labels = np.array([
                          colname.split("_")[0]
                          for colname in col_names
                          if "_Lstar_" + filter_names[0] in colname ])                            # (noffsets,)
    Lstar_offsets       = np.array([
                          utilts.sfConversion(cc)
                          for cc in Lstar_offset_labels ]).astype(float)                          # (noffsets,)
    id_zerooffset       = np.where( Lstar_offsets == 1.0 )[0][0]
    # nz, nfilters, noffsets
    nz                  = len(reds)                                                               # (nz,)
    nbands              = len(filter_names)                                                       # (nbands,)
    noffsets            = len(Lstar_offsets)                                                      # (noffsets,)
    # magnitudes
    magnitudes          = np.nan * np.ones((nz,nbands,noffsets))                                  # (nz,nbands,noffsets)
    for nbis in range(nbands):
        for nois in range(noffsets):
            magnitudes[:,nbis,nois] = tbudata[ Lstar_offset_labels[nois] + "_Lstar_" + filter_names[nbis] ]
    # interpolation
    interpolated_magnitudes = {
            bandis: interpolate.interp1d(
                    x = reds,
                    y = magnitudes[:, nthband ,:],                                                # (nz,noffsets,)
                 axis = 0,
         bounds_error = True,
                 kind = 'linear')(zcl)                                                            # (offsets,)
            for nthband, bandis in enumerate(filter_names) }                                      # (nbands,noffsets)
    # print
    if    verbose:
        print("#", "RS model at z=%.3f" % zcl, "from %s" % path2tbu)
        for nois in range(noffsets):
            printme   = "#" + " " + "%s * Lstar" % Lstar_offset_labels[ nois ]
            for nbis in range(nbands):
                printme = printme + " " + "%.4f (%s)" % (interpolated_magnitudes[ filter_names[nbis] ][nois], filter_names[ nbis ])
            print(printme)
        print()
    # return
    return Lstar_offsets, interpolated_magnitudes
# ---
# output rs equation
# ---
def printRSEquation(path2tbu, filter_names, zcl, band123_names, **kwargs):
    """
    Construct

    color = band1 - band2 = slope * band3 + yint

    Parameters
    ----------------------------------------------------
    path2tbu: str or pyfits.BinTableHDU
        abs path to the fits table created by generateRS.
        If not, it has to be a BinTableHDU.
    filter_names: list containing the names of filters
        list of the filter names, which have to be used in tbu
    zcl: float
        The redshift of the outputed model
    band123_names: list containing 3 filters
        list of the filter names, which have to be used in tbu
    verbose: bool
        Whether to print results on the screen
    """
    # kwargs
    mag_piv             = kwargs.get("mag_piv",     20.0)
    verbose             = kwargs.get(   "verbose",  True)
    # sanitize
    zcl                 = float(        zcl )
    assert len(band123_names) == 3
    #
    Lstar_offsets, interpolated_magnitudes = printRSmodel(
                                             path2tbu = path2tbu,
                                          filter_names = filter_names,
                                                   zcl = zcl,
                                                   **kwargs)
    # bands
    band1_name, band2_name, band3_name = band123_names
    band1_at_zcl        = interpolated_magnitudes[ band1_name ]         # (offsets,)
    band2_at_zcl        = interpolated_magnitudes[ band2_name ]         # (offsets,)
    band3_at_zcl        = interpolated_magnitudes[ band3_name ]         # (offsets,)
    color_at_zcl        = band1_at_zcl - band2_at_zcl                                                                                # (offsets,)
    #interpolate.interp1d(x = reds, y = colors           , axis = 0, bounds_error = True, kind = 'linear')(zcl)
    bfit, bcov          = fitLinearRelation(mags = band3_at_zcl, colors = color_at_zcl, mag_piv = mag_piv)
    berr                = np.sqrt(np.diag(bcov))
    # ---
    if      verbose:
        print("#", "%s - %s = %.4f \pm %.4f * (%s - %.4f) + %.4f \pm %.4f" % \
              (band1_name,band2_name,bfit[0],berr[0],band3_name,mag_piv,bfit[1],berr[1])
              )
    # return
    return bfit, berr
# ---
# linear model
# ---
def __linear_relation(xdata, slope, yint, xpiv):
    return slope * (xdata - xpiv) + yint
# ---
# fitting
# ---
def fitLinearRelation(mags,colors, **kwargs):
    # kwargs
    mag_piv = kwargs.get("mag_piv", 20.0)
    # sanitize
    mags    = np.array(      mags, ndmin = 1)
    colors  = np.array(    colors, ndmin = 1)
    #
    bfit, bcov = optimize.curve_fit(
                 lambda xdata, slope, yint: __linear_relation(xdata = xdata, slope = slope, yint = yint, xpiv = mag_piv),
                 xdata = mags,
                 ydata = colors,
                 )
    # return
    return bfit, bcov
