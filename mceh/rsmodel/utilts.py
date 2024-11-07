#!/usr/bin/env python

# basics
import numpy                as np
import math
import os

# pyfits
import astropy.io.fits      as pyfits


# ---
# ensuredir
# ---
def ensuredir(path2dir):
    """
    path2dir is the directory you want to make sure whether it exists and create it recursively if not.
    It is equal to mkdir -p path2dir 2>/dev/null
    Parameters:
        -`path2dir` : the directory path name you want to make sure whether it exist
                      and create it recursively if not.
    """
    if not os.path.exists(path2dir):
        try:
            os.makedirs(path2dir)
            print("#", "Create dir:", path2dir)
        except OSError as exception:
            pass
# ---
# read pickle file
# ---
def readinpickle(path2picklefile):
    import pickle
    FF = open(path2picklefile, "rb")
    readin = pickle.load(FF)
    FF.close()
    return readin
# ---
# save fits table
# ---
def saveFitsTables(tableList, outputname, comments = None):
    """
    Inputs
    ---------------------------------------------
    tableList: list
        The list containing the binTableHDU files
    outputname: str
        The absolute path to the output files
    comments: None or a list of str
        The comments that are put into the header. No actions are done if None.

    Return
    ---------------------------------------------
    """
    # sanitize
    if not isinstance(tableList, list):
        msg = "The input tableList is not a list."
        raise IOError(msg)
    # create a primary HDU
    phdu = pyfits.PrimaryHDU()
    # add header
    if    isinstance(comments, list):
        for comment_is in comments:
            phdu.header['COMMENT'] = comment_is
    elif  comments is None:
        pass
    else:
        msg = 'comments %s is neither a list of string nor None.' % comments
        raise IOError(msg)
    # construct list
    list4fits= np.append(phdu, tableList).tolist()
    #for data in tableList:
    #    list4fits.append(pyfits.BinTableHDU(data))
    thdulist = pyfits.HDUList(list4fits)
    thdulist.writeto(outputname, overwrite=True)
    # return
    return
# ---
# string and float conversion
# ---
def sfConversion(inputis, nround = 3):
    """
    convert inputis between string or float
    """
    # sanitize
    nround          = int(nround)
    # float -> string
    if      isinstance(inputis, float):
        inputis     = float(inputis)
        if    inputis > 0.0:
            outputis= ("p" + ("%." + str(nround) + "f") % inputis).replace(".","d")
        elif  inputis < 0.0:
            outputis= ("m" + ("%." + str(nround) + "f") % abs(inputis)).replace(".","d")
        else:
            outputis= ("0" + ("%." + str(nround) + "f") % inputis).replace(".","d")
    elif    isinstance(inputis, str):
        if    inputis[0] == "p":
            outputis= 1.0 * float(inputis[1:].replace("d","."))
        elif  inputis[0] == "m":
            outputis=-1.0 * float(inputis[1:].replace("d","."))
        elif  inputis[0] == "0" and inputis[1] == "0":
            outputis= 0.0
        else:
            msg = 'the inputis %s cannot be converted into flaot.' % inputis
            raise IOError(msg)
    else:
        msg = 'inputis %s is neither float nor string.' % inputis
        raise IOError(msg)
    # return
    return outputis
