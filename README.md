# ReadMe #
This is the ReadMe for the GitHub repository that contains supplemental data and codes for the in review manuscript "Low variability, snowmelt runoff inhibits coupling of climate, tectonics, and topography in the Greater Caucasus" by Forte et al. submitted to Earth and Planetary Science Letters. Below are brief descriptions of categories of items included in this repository.

Note that this does not include all codes to process some raw data that we either do not have permission to redistribute (e.g., GRDC or ECAD time series) or is impractical to rehost in their entirety (TRMM or MOD10C2 time series). In these cases, we provide the processed values or products.

## Supplemental Methods and Figures
A pdf of the supplemental methods and figures referenced in the main text of the manuscript. An identical version is submitted with the in review manuscript as supplementary material.

## Lithology Compilations
A pdf describing the compilation of geologic maps to describe the lithology of each catchment. Included in this are lithologic maps and distributions for each sampled catchment.

## Excel Tables
Four excel supplemental tables referenced in the main text of the manuscript. Identical versions are submitted with the in review manuscript as supplementary material.

## Codes
We provide a series of analysis or plotting codes written in Python v.3.7 for some of the more intricate analyses described in the main text methods or supplement. We also provide some of the scripts for generating various plots in the main paper or  supplement. This is not a complete set of all functions or scripts to generate every figure in the paper, but rather as a way of interpreting the data stored in the data tables. Brief descriptions of each code are provided below.
### analyze_widths.py
Reads and plots results of measuring widths of streams for selected 10Be basins as described in the supplement
### basic_topo_plots.py
Reads and plots basics comparisons of erosion rate and channel steepness and gradient, also compares erosion rates to ksn over a range of concavities
### cluster_analyis.py
Performs and plots the results of the k-means clustering to break the 10Be basins into different hydro-climatic regions/categories
### correlation_plots.py
Plots a variety of variables against mean basin ksn, mean basin gradient, and erosion rate and calculates pearson's correlation coefficient for each pair
### grdc_classify.py
Classifies GRDC basins based on clusters defined by cluster_analyis.py, note that this is done manually
### grdc_seasonal_plots.py
Plots the various plots related to the seasonal fraction and daily means
### optimize_k_e.py
Performs the optimization to estiamte k_e for every 10Be basin
### optimize_tau_c.py
Performs the optimization to estimate tau_c for every 10Be basin (alternative to optimize_k_e.py)
### optimize_plot_results.py
Plots the results of both the optimize_k_e.py and optimize_tau_c.py routines
### power_law_fits.py
Algorithms for performing the bootstrap and monte-carlo fitting of the power law SPIM ksn - E relationship
### stim_fit.py
Calculates a best STIM fit to the data
### residuals.py 
Calcuates and plots residuals and RMSE for the various fits to the ksn - E data
### stochastic_threshold.py
Implementation of the stochastic threshold model
### tectonics_plot.py
Plots some of the plots related to the comparison of the erosion rates with tectonic indicators

## Data Tables
The "data_tables" folder includes raw data that is called in the various Python codes above and reproduce the data available in the tables provided in the supplement of the text. There are two subfolders within the "data_tables" folder
### GRDC Daily Means
The "grdc_daily_means" folder contains mean daily values, averaged over the length of the record, for the GRDC basins. I.e. there are 365 mean daily values in mm/day where the first value represent the mean daily runoff on January 1 across all years of the record and so on.
### Width Tables
The "width_tables" folder contains both raw and moving averaged tables of river width, drainage area, river length, and gradient for the 10Be basins for which we were able to extract width information.
### File List
  * chi_r2.csv - chi-elevation R^2 values for all streams and for the trunk streams for each 10Be basin
  * filtered_ksn_050.csv - ksn with a reference concavity of 0.5 for the different filter options (lithogically, low, and high elevation, see supplement for discussion). Corresponds to erosion rates for these different scenarios in separate tables, "litho_ero.csv", "low_ero.csv", and "high_ero.csv"
  * gauged_discharges.csv - GRDC discharge data reporting mean discharge (Qb), mean runoff (Rb), the discharge for the 2 year flood (Qbf), and the runoff for the 2 year flood (Rbf), discharge is in m^3/sec and runoff in m/sec
  * gc_ero_master_table.csv - majority of relevant values for 10Be basins
  * grdc_seasonal_values.csv - seasonal means for GRDC basins and annual, seasonal, and event totals and fractions
  * grdc_summary_values.csv - basic info for the GRDC basins
  * high_ero.csv - erosion rate calculations for the high elevation filtering experiment
  * interpolated_convergence.txt - estimated GC velocity, LC velocity, and GC-LC convergence for 10Be basins
  * ksn_diff_concavities.csv - mean basin ksn, standard error, and standard deviation for concavities of 0.3, 0.4, 0.45, and 0.6
  * lith_ero.csv - erosion rate calculations for the lithological filtering experiemnt
  * low_ero.csv - erosion rate calculations for the low elevation filtering experiment
  * swath_distances.txt - distances along and across swath for the 10Be basins
  * width_key.csv - list of river mouth identifying numbers compared to sample names

## Result Tables
The "result_tables" folder includes tables output from the analysis codes in the main repository, specifically the "optimize_k_e.py", "optimize_tau_c.py", "power_law_fits.py", and "stim_fit.py". The results here represent the values used in the published paper. If users wish to explore the sensitivity of the results to parameter choices, it is recommended that you store these original results elsewhere as re-running the above codes will overwrite the existing result tables.

## Rasters
We include three rasters that are generally not directly downloadable from public databases. All three of these were originally processed for use in Forte et al, 2016, Earth and Planetary Science Letters:
### MOD10C2_2002_2012_CAUCASUS_mean.tif
Mean snow cover derived from MODIS MOD10C2 data averaged over the 2002-2012 period
### MOD10C2_2002_2012_CAUCASUS_monthlySTD.tif 
Mean of the standard deviation of monthly mean snow cover derived from MODIS MOD10C2 data from 2002-2012
### TRMM3B42V7_1998_2012_CAUCASUS_mean_mm_day.tif
Mean daily rainfall from TRMM3B42V7 between 1998 and 2012 in mm/day.

## CRONUS v3 inputs
We provide raw inputs suitable for direct input to CRONUS v3 online erosion rate calculator. There are four files:
  * cronus3_high_in.txt - Input for erosion rate calculation assuming contribution only from top 50% of landscape
  * cronus3_lith_in.txt - Input for erosion rate calculation assuming contribution only from lithologies that are dominantly siliclastic or otherwise expected to be sources of quartz
  * cronus3_low_in.txt - Input for erosion rate calculation assuming contribution only from bottom 50% of landscape
  * cronus3_reg_in.txt - Input for erosion rate calculation assuming contribution form entire catchment

## Shapefiles
We provide a series of shapefiles to aid in the exploration or use of the data from this effort. Specifically we include:
### 10Be Polygons
These are outlines of the sampled catchments for the 10Be erosion rates that represent the primary data for this manuscript. The attribute table for this shapefile includes a variety of data:
  * center_x - x coordinate of center of basin
  * center_y - y coordinate of center of basin
  * drainage_ar - drainage area in km^2
  * outlet_elev - elevation of outlet (sample location) in m
  * mean_el - mean elevation of basin in m
  * max_el - maximum elevation of basin in m
  * mean_ksn - mean normalized channel steepness (ksn) with reference concavity of 0.5
  * mean_gradie - mean gradient of basin
  * se_el - standard error of mean elevation in m
  * se_ksn - standard error of mean ksn with reference concavity of 0.5
  * se_gradient - standard error of mean gradient
  * std_el - standard deviation of mean elevation in m
  * std_ksn - standard deviation of mean ksn with reference concavity of 0.5
  * std_gradien - standard deviation of mean gradient
  * mnKSN_0_3 - mean ksn with reference concavity of 0.3
  * seKSN_0_3 - standard error of ksn with reference concavity of 0.3
  * stdKSN_0_3 - standard deviation of ksn with reference concavity of 0.3
  * mnKSN_0_4 - mean ksn with reference concavity of 0.4
  * seKSN_0_4 - standard error of ksn with reference concavity of 0.4
  * stdKSN_0_4 - standard deviation of ksn with reference concavity of 0.4
  * mnKSN_0_45 - mean ksn with reference concavity of 0.45
  * seKSN_0_45 - standard error of ksn with reference concavity of 0.45
  * stdKSN_0_45 - standard deviation of ksn with reference concavity of 0.45
  * mnKSN_0_6 - mean ksn with reference concavity of 0.6
  * seKSN_0_6 - standard error of ksn with reference concavity of 0.6
  * stdKSN_0_6 - standard deviation of ksn with reference concavity of 0.6
  * hyp_int - hypsometric intergral
  * theta - best fit concavity
  * chi_r2 - R^2 value on chi-elevation relationship
  * outlet_lat - latitude of outlet (sample location)
  * outlet_lon - longitude of outlet (sample location)
  * mean_ksn_q - mean discharge weighted ksn with reference concavity of 0.5 (see Adams et al, 2020, Science Advances)
  * se_ksn_q - standard error on discharge weighted ksn with reference concavity of 0.5
  * std_ksn_q - standard deviation on discharge weighted ksn with reference concavity of 0.5
  * mean_TRMM_M - mean rainfall from TRMM3B42V7 in m/yr
  * se_TRMM_MAP - standard error of rainfall from TRMM3B42V7 in m/yr
  * std_TRMM_MA - standard deviation of rainfall from TRMM3B42V7 in m/yr
  * mean_TRMM_T - mean percent of rainfall that comes in the top 5 events per year
  * se_TRMM_TOP - standard error on percent of rainfall that comes in the top 5 events per year
  * std_TRMM_TO - standard deviation on percent of rainfall that comes in the top 5 events per year
  * mean_SNOWpe - mean percent of snow cover
  * se_SNOWperc - standard error on percent of snow cover
  * std_SNOWper - standard deviation on percent of snow cover
  * mean_SNOWst - mean standard deviation of mean monthly snow cover
  * se_SNOWstd - standard error on mean standard deviation of mean monthly snow cover
  * std_SNOWstd - standard deviation on mean standard deviation of mean monthly snow cover
  * mean_EVI_DJ - mean enhanced vegetation index (EVI) in winter (DJF)
  * se_EVI_DJF - standard error on mean enhanced vegetation index (EVI) in winter (DJF)
  * std_EVI_DJF - standard deviation on mean enhanced vegetation index (EVI) in winter (DJF)
  * mean_EVI_MA - mean enhanced vegetation index (EVI) in spring (MAM)
  * se_EVI_MAM - standard error on mean enhanced vegetation index (EVI) in spring (MAM)
  * std_EVI_MAM - standard deviation on mean enhanced vegetation index (EVI) in spring (MAM)
  * mean_EVI_JJ - mean enhanced vegetation index (EVI) in summer (JJA)
  * se_EVI_JJA - standard error on mean enhanced vegetation index (EVI) in summer (JJA)
  * std_EVI_JJA - standard deviation on mean enhanced vegetation index (EVI) in summer (JJA)
  * mean_EVI_SO - mean enhanced vegetation index (EVI) in fall (SON)
  * se_EVI_SON - standard error on mean enhanced vegetation index (EVI) in fall (SON)
  * std_EVI_SON - standard deviation on mean enhanced vegetation index (EVI) in fall (SON)
  * mean_rlf100 - mean local relief in 1000 meter radius
  * se_rlf1000 - standard error on mean local relief in 1000 meter radius
  * std_rlf1000 - standard deviation on mean local relief in 1000 meter radius
  * mean_rlf250 - mean local relief in 2500 meter radius
  * se_rlf2500 - standard error on mean local relief in 2500 meter radius
  * std_rlf2500 - standard deviation on mean local relief in 2500 meter radius
  * mean_rlf500 - mean local relief in 5000 meter radius
  * se_rlf5000 - standard error on mean local relief in 5000 meter radius
  * std_rlf5000 - standard deviation on mean local relief in 5000 meter radius 
  * sample_name - original sample name
  * region - region classification, ALZN - Alazani valley, WGC - western, CGC - central, NGC - northeastern Greater Caucasus, LC - Lesser Caucasus
  * be10_atom_p - 10Be atoms per gram
  * be10_unc - 1 sigma uncertainty on 10Be atoms per gram
  * St_E_rate_g - erosion rate in g/cm2/yr assuming St scaling scheme from CRONUS vs3
  * St_E_rate_m - erosion rate in m/Myr assuming St scaling scheme from CRONUS vs3 
  * St_Int_Unc - internal uncertainty on erosion rate in m/Myr
  * St_Ext_Unc - external uncertainty on erosion rate in m/Myr
  * Lm_E_rate_g - erosion rate in g/cm2/yr assuming Lm scaling scheme from CRONUS vs3
  * Lm_E_rate_m - erosion rate in m/Myr assuming Lm scaling scheme from CRONUS vs3 
  * Lm_Int_Unc - internal uncertainty on erosion rate in m/Myr
  * Lm_Ext_Unc - external uncertainty on erosion rate in m/Myr
  * LSDn_E_rate - erosion rate in g/cm2/yr assuming St scaling scheme from CRONUS vs3
  * LSDn_E_rate1 - erosion rate in m/Myr assuming St scaling scheme from CRONUS vs3 
  * LSDn_Int_Un - internal uncertainty on erosion rate in m/Myr
  * LSDn_Ext_Un - external uncertainty on erosion rate in m/Myr
  * mean_runoff - estimated mean runoff in mm/day
  * mean_runoff1 - estiamted  mean runoff in mm/day using conservative approach
  * corrected_m - mean rainfall from corrected TRMM in mm/day
  * runoff_rati - estimated runoff ratio
  * cr_SSN_est - estimated variability using stretched exponential from standard deviation of monthly mean snow cover
  * cr_z_est - estimated variability using stretched exponential from max elevation of basin
  * k_SSN_est - estimated variability using power law from standard deviation of monthly mean snow cover
  * k_z_est - estimated variability using power law from max elevation of basin 
  * mode_rock_t - modal rock type exposed in catchment
  * mode_rock_t1 - percentage of catchment occupied by this modal rock type
  * rock_type_p - percentage of catchment that is unmapped rocktype
  * rock_type_p1 - percentage of catchment that is limestone and minor clastics
  * rock_type_p2 - percentage of catchment that is limestone and marl     
  * rock_type_p3 - percentage of catchment that is limestone
  * rock_type_p4 - percentage of catchment that is marl
  * rock_type_p5 - percentage of catchment that is limestone and sandstone
  * rock_type_p6 - percentage of catchment that is sandstone and shale
  * rock_type_p7 - percentage of catchment that is sandstone and minor carbonate
  * rock_type_p8 - percentage of catchment that is shale
  * rock_type_p9 - percentage of catchment that is conglomerate
  * rock_type_p10 - percentage of catchment that is undifferentiated volcanics
  * rock_type_p11 - percentage of catchment that is phyllite and schist
  * rock_type_p12 - percentage of catchment that is granite
  * mksn_unmapp - mean ksn in unmapped 
  * mgrad_unmap - mean gradient in unmapped
  * mksn_limest - mean ksn in limestone and minor clastics
  * mgrad_limes - mean gradient in limestone and minor clastics
  * mksn_limest1 - mean ksn in limestone and marl
  * mgrad_limes1 - mean gradient in limestone and marl
  * mksn_limest2 - mean ksn in limestone
  * mgrad_limes2 - mean gradient in limestone
  * mksn_marl - mean ksn in marl
  * mgrad_marl - mean gradient in marl
  * mksn_limest3 - mean ksn in limestone and sandstone
  * mgrad_limes3 - mean gradient in limestone and sandstone    
  * mksn_sandst - mean ksn in sandstone and shale
  * mgrad_sands - mean gradient in sandstone and shale
  * mksn_sandst1 - mean ksn in sandstone and minor carbonate
  * mgrad_sands1 - mean gradient in sandstone and carbonate
  * mksn_shale - mean ksn in shale
  * mgrad_shale - mean gradient in shale
  * mksn_conglo - mean ksn in conglomerate
  * mgrad_congl - mean gradient in conglomerate
  * mksn_volcan - mean ksn in undifferentiated volcanics
  * mgrad_volca - mean gradient in undifferentiated volcanics
  * mksn_phylli - mean ksn in phyllite and schist
  * mgrad_phyll - mean gradient in phyliite and schist
  * mksn_granit - mean ksn in granite
  * mgrad_grani - mean gradient in granite
  
### GRDC Polygons
These are outlines of the GRDC watersheds used for runoff analysis. The attribute table for this shapefile includes a variety of data:
  * GRDC_ID - Original ID number in the GRDC database at the time of data acquisition (these are not static)
  * mean_el - mean elevation in m
  * stat_Lat - latitude of station
  * stat_Lon - longitude of station
  * DA_km2 - drainage area of watershed in km^2
  * Q_m3_s - mean discharge in Q_m3_s
  * R_mm_dy - mean runoff in mm/day
  * k_99perc - variability fit with a power law for the events above a 99% threshold
  * N_99perc - number of events above threshold
  * TRMMrain - corrected mean TRMM rainfall in watershed in mm/day
  * RunRatio - runoff ratio based on TRMM rainfall
  * Z_max_km - max elevation in km of watershed
  * Snow_std - mean standard deviation of monthly mean snow cover
  * L_Rec_dy - length of record in days
  * Rec_Strt - start date of record
  * Rec_Stop - stop date of record 
  * cluster_num - cluster number from main text

### ECAD Points
These are the locations of ECAD precipitation stations used to calibrate the TRMM rainfall data. The attribute table for this shapefile includes:
  * stat_id - Original ECAD station ID
  * elev_m - elevation of station in m
  * mm_day - mean preciptation in mm/day
  * start - start date of the record
  * xEnd - stop date of the record

### Individual Basin PolyLines
For each sampled basin for which an erosion rate is reported, we provide the stream network extracted from the STRM30 dataset. Each shapefile is named 'basin_XXXX_ksn0_5.shp', where XXXX is the sample name and the suffix indicates that it includes ksn values calculated for a reference concavity of 0.5. Each shapefile has the following attributes for each segment:
  * ksn - mean ksn value for that segment
  * uparea - maximum drainage area for that segment in m^2
  * gradient - mean gradient for that segment
  * cut_fill - mean of deviation between raw DEM and hydrologically conditioned DEM, negative numbers indicate conditioned elevations below raw elevations
  * seg_dist - legnth of segment in m, measured as stream distance


