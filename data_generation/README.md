Change Log of Data
===

[Toc]

---
# Introduction
This markdown file contains the history of data change for Hung-Yu Lin's work. The python package `mceh`(created by Lin; its name is `mceh_verX` where "X" is the version) is used to generate new data. When a generating code is mentioned (named `generate_modified_xxx.ipynb` where xxx is the type of data), despite it is saved in `mceh/data_generation`, it should be ran in the same directory as `mceh` package. Therefore, the directory should looks like the following to run the generating code.
```
/mceh_verX
/data
    data_needed.fits
generate_modified_xxx.ipynb
```
The `data_needed.fits` is the data needed for generating modified data and should be easily told in the code.
# eFEDS
## Original
The original data is called `eFEDS_clusters_V3.2.fits` and `eFEDS_c001_main_ctp_clus_v2.1.fits`, which is from [the eFEDS catalogue website](https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/) and their IDs are 11 and 12 (`eFEDS EXT Clusters cat` and `eFEDS EXT CTP cat` buttons).

## eFEDS Ver.6
The data name is `modified_efeds_ver6.fits`.
The generating code is `generate_modified_efeds_ver6.ipynb`.
`modified_random_ver1.fits`, `modified_hsc_ver1.fits`, `eFEDSmasstable.fits`, `bc03_rs_zfp3d0_tau_p0d4.fits`(from I-Non) and `eFEDS_c001_main_ctp_clus_v2.1.fits` are needed.
The modifications of the previous versions are unclear, so they are not recorded in this markdown file. This new version of eFEDS file will be generated independently of the previous modified versions.
The modifications are:
- Only columns named `Name`, `ID_SRC`, `RA_CORR`, `DEC_CORR` and `Z_BEST_COMB` remain from `eFEDS_c001_main_ctp_clus_v2.1.fits`.
- Add a column named `low_cont_flag`, which is `True` when the clusters meet the following conditions (from [Chiu+23](https://arxiv.org/abs/2207.12429) section 2.1): 
1. `DET_LIKE` > 5
2. `EXT_LIKE` > 6
3. `EXT` > 0
4. `F_CONT_BEST_COMB` < 0.3
5.  0.1 < `Z_BEST_COMB` < 1.2
- Add columns `r500c_lcdm` and `median_500c_lcdm` from `eFEDSmasstable.fits`. Note that unit of `median_500c_lcdm` has changed to better suit the units of astropy.
- Add columns `unmasked_area` and `area`. The unmasked area is obtained by counting numbers (#) of random points of random catlog within the radii (`r500c_lcdm`) of the cluter centers, so it is #/100 $\text{arcmin}^2$. And `area` is the actual areas of the clusters on the sky, which are obtained by $2\pi (1-\cos^2{r})$.
- Add columns `g/r/i/z/ymag_cmodel` which are the characteristic magnitude from I-Non's model.
- Add the column `galaxy_index` which is the  indicies of galaxies (from `modified_hsc_ver1.fits`) within the radii (`r500c_lcdm`) of the cluster centers.

## eFEDS Ver.7
The data name is `modified_efeds_ver7.fits`.
The generating code is `generate_modified_efeds_ver7.ipynb`.
`modified_efeds_ver6.fits` and `bc03_rs_zfp3d0_tau_p0d4.fits` are needed.
`mceh` version is `3.0.0-alpha.2`.
- Fix cmag()
- Add the code of I-Non's characteristic magnitude model called `rsmodel`
- Add a column `unmasked_fraction`

## eFEDS Ver.8
The data name is `modified_efeds_ver8.fits`.
The generating code is `generate_modified_efeds_ver8.ipynb`.
`modified_efeds_ver7.fits` and `eFEDS_c001_main_ctp_clus_v2.1.fits` are needed.
`mceh` version is `3.0.0`.
- Add a column `SIGMA_Z_BEST_COMB`

# HSC
## Original
The original data is called `429561.fits`, which is from [the HSP-SSP data release website](https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/). Use the `CAS Search` entry and the SQL (`hsc.sql`) is saved in the same directory as this markdown file.

## HSC Ver.1
The data name is `modified_hsc_ver1.fits`.
The generating code is `generate_modified_hsc_ver1.ipynb`.
`429561.fits` is needed.
The modifications are:
- Only rows with `science_flag == True` remain.
- Only rows with `iclassification_extendedness > 0.9` remain.
- Only columns named `object_id`, `tract`, `ra`, `dec`, `gmag_cmodel`, `rmag_cmodel`, `imag_cmodel`, `zmag_cmodel` and `ymag_cmodel` remain.
- The columns `ra` and `dec` have a unit degree.

## HSC Ver.2
The data name is `modified_hsc_ver2.fits`.
The generating code is `generate_modified_hsc_ver2.ipynb`.
`429561.fits` is needed.
The modifications are:
- Only rows with `science_flag == True` remain.
- Only rows with `iclassification_extendedness > 0.9` remain.
- Only columns named `object_id`, `tract`, `ra`, `dec`, `gmag_cmodel`, `rmag_cmodel`, `imag_cmodel`, `zmag_cmodel` and `ymag_cmodel`, `gmag_err`, `rmag_err`, `imag_err`, `zmag_err`, `ymag_err`, `photoz_mean`, `photoz_mode`, `photoz_median`, `photoz_best`, `photoz_mc`, `photoz_err68_min`, `photoz_err68_max`, `photoz_err95_min` and `photoz_err95_max` remain.
- The columns `ra` and `dec` have a unit degree.

## HSC Ver.3
The data name is `modified_hsc_ver3.fits`.
The generating code is `generate_modified_hsc_ver3.ipynb`.
`modified_hsc_ver2.fits` is needed.
- Add a uniformly random number [0, 1) for each index. The column name is `random`.

## HSC Ver.4
The data name is `modified_hsc_ver4.fits`.
The generating code is in `hsc_ver4` folder.
This version comes up because HSC has updated their data (s23b).
From the original s23b data whose directory is `/array/users/profoak/dataDir/hsc/s23b/spring/spring_obj.fits`, it has the following changes:
- Only `science_flag == True` rows remain
- Only `iclassification_extendedness > 0.9` rows remain
- `science_flag` and `iclassification_extendedness` columns are removed
- A uniform random number in [0, 1) is added for every galaxy
- `ra` and `dec` columns now have a unit `degree` from `astropy.units`

# Random

## Original
The random data is `random_gama09h.fits`, which directly links to `/array/users/profoak/dataDir/hsc/pdr3/random/random_gama09h.fits` in Pikachu.

## Random Ver.1

The data name is `modified_random_ver1.fits`. 
The generating code is called `generate_modified_random_ver1.ipynb`.
`random_gama09h.fits` is needed.
The modifications are:
- Only save the `RA` and `DEC` columns.
- Add the unit `degree` to these two columns.
- Only rows with `science_flag == True` are saved.


## Random Ver.2
The data name is `modified_random_ver2.fits`.
The generating code is in `random_ver2` folder.
This version comes up because HSC has updated their data (s23b).
From the original s23b data whose directory is `/array/users/profoak/dataDir/hsc/s23b/spring/spring_random.fits`, it has the following changes:
- Only `science_flag == True` rows remain
- `science_flag` column is removed
- `ra` and `dec` columns now have a unit `degree` from `astropy.units`

# eFEDSmasstable

The data is called `eFEDSmasstable.fits` and it's from [I-Non's website](https://inonchiu.github.io/eFEDScosmology_chiu22/).

# Background

The generating codes are in `generate_background` folder. `get_bkg20241109.py` is the first code and `20241111.ipynb` is the second one and needs data generated from the first code.
`mceh` version is `3.0.0`.