--
-- PDR3 photometry / photo-z catalog
--
SELECT 
   photo.object_id,
   photo.tract,
   photo.ra,
   photo.dec, 
   --- star/galaxy separation
   photo.i_extendedness_value as iclassification_extendedness,
   --- photometry measurements
   photo.g_cmodel_mag - photo.a_g as gmag_cmodel, 
   photo.g_cmodel_magerr as gmag_err,
   ---
   photo.r_cmodel_mag - photo.a_r as rmag_cmodel,
   photo.r_cmodel_magerr as rmag_err,
   ---
   photo.i_cmodel_mag - photo.a_i as imag_cmodel,
   photo.i_cmodel_magerr as imag_err,
   ---
   photo.z_cmodel_mag - photo.a_z as zmag_cmodel,       
   photo.z_cmodel_magerr as zmag_err,
   ---
   photo.y_cmodel_mag - photo.a_y as ymag_cmodel,
   photo.y_cmodel_magerr as ymag_err,
   --- photometry flags
   (  photo.g_pixelflags_edge  
   OR photo.r_pixelflags_edge  
   OR photo.i_pixelflags_edge  
   OR photo.z_pixelflags_edge  
   OR photo.y_pixelflags_edge  

   OR photo.g_pixelflags_interpolatedcenter  
   OR photo.r_pixelflags_interpolatedcenter  
   OR photo.i_pixelflags_interpolatedcenter  
   OR photo.z_pixelflags_interpolatedcenter  
   OR photo.y_pixelflags_interpolatedcenter  

   OR photo.g_pixelflags_crcenter  
   OR photo.r_pixelflags_crcenter  
   OR photo.i_pixelflags_crcenter  
   OR photo.z_pixelflags_crcenter  
   OR photo.y_pixelflags_crcenter  

   OR photo.g_pixelflags_saturatedcenter  
   OR photo.r_pixelflags_saturatedcenter  
   OR photo.i_pixelflags_saturatedcenter  
   OR photo.z_pixelflags_saturatedcenter  
   OR photo.y_pixelflags_saturatedcenter  

   OR photo.g_pixelflags_suspectcenter  
   OR photo.r_pixelflags_suspectcenter  
   OR photo.i_pixelflags_suspectcenter  
   OR photo.z_pixelflags_suspectcenter  
   OR photo.y_pixelflags_suspectcenter  

   OR photo.g_cmodel_flag  
   OR photo.r_cmodel_flag  
   OR photo.i_cmodel_flag  
   OR photo.z_cmodel_flag  
   OR photo.y_cmodel_flag
   ) as combined_photo_flag 
  --- FDFC flag
 , (   photo.g_inputcount_value >= 2
   AND photo.r_inputcount_value >= 2
   AND photo.i_inputcount_value >= 4
   AND photo.z_inputcount_value >= 4
   AND photo.y_inputcount_value >= 4
   ) as fdfc_flag
   --- bright star masks
 , ( mask.g_mask_brightstar_halo   
  OR mask.r_mask_brightstar_halo   
  OR mask.i_mask_brightstar_halo   
  OR mask.z_mask_brightstar_halo   
  OR mask.y_mask_brightstar_halo 
  --- 
  OR mask.g_mask_brightstar_ghost  
  OR mask.r_mask_brightstar_ghost  
  OR mask.i_mask_brightstar_ghost  
  OR mask.z_mask_brightstar_ghost  
  OR mask.y_mask_brightstar_ghost 
  ---
  OR mask.g_mask_brightstar_blooming 
  OR mask.r_mask_brightstar_blooming 
  OR mask.i_mask_brightstar_blooming 
  OR mask.z_mask_brightstar_blooming 
  OR mask.y_mask_brightstar_blooming
  ) as combined_bstar_flag
  --- science ready flag 
  , (   NOT photo.g_pixelflags_edge
    AND NOT photo.r_pixelflags_edge
    AND NOT photo.i_pixelflags_edge
    AND NOT photo.z_pixelflags_edge
    AND NOT photo.y_pixelflags_edge

    AND NOT photo.g_pixelflags_interpolatedcenter
    AND NOT photo.r_pixelflags_interpolatedcenter
    AND NOT photo.i_pixelflags_interpolatedcenter
    AND NOT photo.z_pixelflags_interpolatedcenter
    AND NOT photo.y_pixelflags_interpolatedcenter

    AND NOT photo.g_pixelflags_crcenter
    AND NOT photo.r_pixelflags_crcenter
    AND NOT photo.i_pixelflags_crcenter
    AND NOT photo.z_pixelflags_crcenter
    AND NOT photo.y_pixelflags_crcenter
  
    AND NOT photo.g_pixelflags_saturatedcenter
    AND NOT photo.r_pixelflags_saturatedcenter
    AND NOT photo.i_pixelflags_saturatedcenter
    AND NOT photo.z_pixelflags_saturatedcenter
    AND NOT photo.y_pixelflags_saturatedcenter
  
    AND NOT photo.g_pixelflags_suspectcenter
    AND NOT photo.r_pixelflags_suspectcenter
    AND NOT photo.i_pixelflags_suspectcenter
    AND NOT photo.z_pixelflags_suspectcenter
    AND NOT photo.y_pixelflags_suspectcenter
  
    AND NOT photo.g_cmodel_flag
    AND NOT photo.r_cmodel_flag
    AND NOT photo.i_cmodel_flag
    AND NOT photo.z_cmodel_flag
    AND NOT photo.y_cmodel_flag
       
    AND photo.g_inputcount_value >= 2
    AND photo.r_inputcount_value >= 2
    AND photo.i_inputcount_value >= 4
    AND photo.z_inputcount_value >= 4
    AND photo.y_inputcount_value >= 4

    AND NOT mask.g_mask_brightstar_halo
    AND NOT mask.r_mask_brightstar_halo
    AND NOT mask.i_mask_brightstar_halo
    AND NOT mask.z_mask_brightstar_halo
    AND NOT mask.y_mask_brightstar_halo
 
    AND NOT mask.g_mask_brightstar_ghost
    AND NOT mask.r_mask_brightstar_ghost
    AND NOT mask.i_mask_brightstar_ghost
    AND NOT mask.z_mask_brightstar_ghost
    AND NOT mask.y_mask_brightstar_ghost

    AND NOT mask.g_mask_brightstar_blooming
    AND NOT mask.r_mask_brightstar_blooming
    AND NOT mask.i_mask_brightstar_blooming
    AND NOT mask.z_mask_brightstar_blooming
    AND NOT mask.y_mask_brightstar_blooming
    ) as science_flag
   --- photoz and stellar properties
   , redz.photoz_mean
   , redz.photoz_mode
   , redz.photoz_median
   , redz.photoz_best
   , redz.photoz_mc
   , redz.photoz_err68_min
   , redz.photoz_err68_max
   , redz.photoz_err95_min
   , redz.photoz_err95_max
   , redz.stellar_mass
   , redz.stellar_mass_err68_min
   , redz.stellar_mass_err68_max
   , redz.stellar_mass_err95_min
   , redz.stellar_mass_err95_max
   , redz.sfr
   , redz.sfr_err68_min
   , redz.sfr_err68_max
   , redz.sfr_err95_min
   , redz.sfr_err95_max
   FROM pdr3_wide.forced           as photo
   LEFT JOIN pdr3_wide.masks       as mask         USING (object_id)
   LEFT JOIN pdr3_wide.photoz_demp as redz         USING (object_id)

   WHERE  

   photo.isprimary is True
   AND boxSearch(photo.coord, 127, 153.5, -3, 6) --- This is GAMA09H
   ---AND boxSearch(photo.coord, 205, 226, -3, 6) --- This is GAMA15H
   ---AND boxSearch(photo.coord, 153.5, 200, -3, 6) --- This is WIDE12H
   ---AND boxSearch(photo.coord, 153.5, 175, -3, 6) --- This is WIDE12H 1
   ---AND boxSearch(photo.coord, 175, 200, -3, 6) --- This is WIDE12H 2
   ---AND boxSearch(photo.coord, 127.0, 225.0, -3, 6) --- This is for the common footprint between eRASS and HSC
   AND (photo.z_cmodel_mag - photo.a_z) < 25.0 --- conservative magnitude cut

   ---LIMIT 10