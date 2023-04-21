#%%

# Import Packages
from pathlib import Path
from os import access
import geopandas as gpd
from pandas.core.frame import DataFrame
import rioxarray as rio
import xarray as xr
import rasterio
from shapely.geometry import mapping, Polygon
import matplotlib.pyplot as plt
import pandas as pd
from xarray.core.computation import join_dict_keys
from xarray.core.parallel import dataset_to_dataarray
from xarray.core.utils import V
import xesmf as xe
import cartopy.crs as ccrs
import numpy as np
# import pymannkendall as mk
# import xclim
# from xclim.indicators import atmos
# from xclim import indices

# to check for autocorrelation
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from pandas.plotting import autocorrelation_plot

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

# Deep Learning modules
import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)


#%%
# keep attributes of all datasets

xr.set_options(keep_attrs=True)


#%%
from dask.distributed import Client, LocalCluster


#%%
# client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
# client

# #%%
# # Dask Local Cluster
# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster)
# client

#%%
# file directory 
inDir = "/datadrive/project/climate/dl_downscaling/data_raw/gpm_imerge_raw/"
inDir2 = "/datadrive/project/climate/dl_downscaling/data_raw/era5_downscale_dataset/"
outDir = "/datadrive/project/climate/dl_downscaling/output/"


#%%
# Path to Geometry files

PATH_TO_SHAPEFILE = '/datadrive/project/climate/dl_downscaling/data_raw/nigerbasin_merged_polygon/hybas_nigerbasin_lev3.shp'

#%%
# open the shapefiles with geopandas
basin_polygon = gpd.read_file(PATH_TO_SHAPEFILE)
basin_polygon

#%%
# climatology filenames
pr = "gpm_africa_daily_transposed_20000930_20210930.nc"

# windspeed = 'windspeed_10m_day_1979_2020.nc'
tas = 'era5_tas_day1979_2021_transposed.nc'
tasmax = 'era5_tmax_day1979_2021.nc'
tasmin = 'era5_tmin_day1979_2021.nc'
huss = 'era5_specific_humidity_1000hpa_resample_2000_2020_transposed.nc'
zg500 = 'era5_geopotential_500hpa_resample_2000_2020_transposed.nc'
zg1000 = 'geopotential_1000hpa_daily_2000_2020.nc'
era5_pr = 'era5_pr_daily_transposed_dim_1979_2021.nc'
psl = 'mean_sea_level_pressure_daily_2000_2020.nc'
uas = 'uwind_10m_day_1979_2020.nc'
vas = 'vwind_10m_day_1979_2020.nc'

#%%
# Cordinates

# lokoja = da_copy.sel(lat='7.7523314', lon='6.7532070', method='nearest').resample(time="Y").mean()

# jide = da_copy.sel(lat='11.3858578', lon='4.1210672', method='nearest').resample(time="Y").mean()

# niamey = da_copy.sel(lat='13.5232201', lon='2.0795191', method='nearest').resample(time="Y").mean()


#%%
# ds = xr.tutorial.open_dataset(inDir + pr,
#                               chunks={'lat': 1000, 'lon': 1000, 'time': -1})
# ds

# #%%
# ds = xr.tutorial.open_dataset(inDir2 + tasmax,
#                               chunks={'lat': 1000, 'lon': 1000, 'time': -1})
# ds

#%%
ds = xr.open_dataset(inDir + pr, chunks={'lat': 25, 'lon': 25, 'time': -1})
ds = ds.transpose("time", "lat", "lon")
ds = ds.sel(time=slice('2001-01-01', '2020-12-31'))
ds 


#%%
ds_tas = xr.open_dataset(inDir2 + tas, chunks={'lat': 25, 'lon': 25, 'time': -1})
ds_tas = ds_tas.transpose("time", "lat", "lon")
ds_tas = ds_tas.sel(time=slice('2001-01-01', '2020-12-31'))
ds_tas = ds_tas.rename({'tas_day':'tas'})
ds_tas

#%%
ds_tasmax = xr.open_dataset(inDir2 + tasmax, chunks={'latitude': 25, 'longitude': 25, 'time': -1})
ds_tasmax = ds_tasmax.sel(time=slice('2001-01-01', '2020-12-31'))
ds_tasmax = ds_tasmax.rename({'tmax_day':'tasmax'})
ds_tasmax = ds_tasmax.rename({'longitude':'lon'})
ds_tasmax = ds_tasmax.rename({'latitude':'lat'})
ds_tasmax

#%%
ds_tasmin = xr.open_dataset(inDir2 + tasmin, chunks={'latitude': 25, 'longitude': 25, 'time': -1})
ds_tasmin = ds_tasmin.sel(time=slice('2001-01-01', '2020-12-31'))
ds_tasmin = ds_tasmin.rename({'tmin_day':'tasmin'})
ds_tasmin = ds_tasmin.rename({'longitude':'lon'})
ds_tasmin = ds_tasmin.rename({'latitude':'lat'})
ds_tasmin 

#%%
ds_huss = xr.open_dataset(inDir2 + huss, chunks={'lat': 25, 'lon': 25, 'time': -1})
ds_huss = ds_huss.transpose("time", "lat", "lon")
ds_huss = ds_huss.sel(time=slice('2001-01-01', '2020-12-31'))
# ds_huss = ds_huss.rename({'longitude':'lon'})
# ds_huss = ds_huss.rename({'latitude':'lat'})
ds_huss 

#%%
ds_zg500 = xr.open_dataset(inDir2 + zg500, chunks={'lat': 25, 'lon': 25, 'time': -1})
ds_zg500 = ds_zg500.transpose("time", "lat", "lon")
ds_zg500 = ds_zg500.sel(time=slice('2001-01-01', '2020-12-31'))
ds_zg500 = ds_zg500.rename({'z':'z500'})
# ds_zg500 = ds_zg500.rename({'longitude':'lon'})
# ds_zg500 = ds_zg500.rename({'latitude':'lat'})
ds_zg500

#%%
ds_zg1000 = xr.open_dataset(inDir2 + zg1000, chunks={'latitude': 25, 'longitude': 25, 'time': -1})
ds_zg1000 = ds_zg1000.sel(time=slice('2001-01-01', '2020-12-31'))
ds_zg1000 = ds_zg1000.rename({'z':'z1000'})
ds_zg1000 = ds_zg1000.rename({'longitude':'lon'})
ds_zg1000 = ds_zg1000.rename({'latitude':'lat'})
ds_zg1000

#%%
ds_psl = xr.open_dataset(inDir2 + psl, chunks={'latitude': 25, 'longitude': 25, 'time': -1})
ds_psl = ds_psl.sel(time=slice('2001-01-01', '2020-12-31'))
ds_psl = ds_psl.rename({'msl':'psl'})
ds_psl = ds_psl.rename({'longitude':'lon'})
ds_psl = ds_psl.rename({'latitude':'lat'})
ds_psl

#%%
ds_uas = xr.open_dataset(inDir2 + uas, chunks={'latitude': 25, 'longitude': 25, 'time': -1})
ds_uas = ds_uas.sel(time=slice('2001-01-01', '2020-12-31'))
ds_uas = ds_uas.rename({'u10':'uas'})
ds_uas = ds_uas.rename({'longitude':'lon'})
ds_uas = ds_uas.rename({'latitude':'lat'})
ds_uas

#%%
ds_vas = xr.open_dataset(inDir2 + vas, chunks={'latitude': 25, 'longitude': 25, 'time': -1})
ds_vas = ds_vas.sel(time=slice('2001-01-01', '2020-12-31'))
ds_vas = ds_vas.rename({'v10':'vas'})
ds_vas = ds_vas.rename({'longitude':'lon'})
ds_vas = ds_vas.rename({'latitude':'lat'})
ds_vas

#%%
ds_era5_pr = xr.open_dataset(inDir2 + era5_pr, chunks={'lat': 25, 'lon': 25, 'time': -1})
ds_era5_pr = ds_era5_pr.transpose("time", "lat", "lon")
ds_era5_pr = ds_era5_pr.sel(time=slice('2001-01-01', '2020-12-31'))
ds_era5_pr = ds_era5_pr.rename({'pr':'era5_pr'})
ds_era5_pr


#%%
# Standardized GPM IMERGE Precipitation by month

#%%
ds_gb_mnt = ds.groupby('time.month')
ds_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_gbmonth_mean = ds.groupby('time.month').mean('time')
ds_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_gbmonth_std = ds.groupby('time.month').std('time')
ds_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_month_anom = ds_gb_mnt - ds_gbmonth_mean
ds_month_anom

#%%
ds_anom_gb_mnt = ds_month_anom.groupby('time.month')
ds_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_standardized = ds_anom_gb_mnt / ds_gbmonth_std
ds_standardized = ds_standardized.drop('month', dim=None)
ds_standardized


#%%
# Standardized Reanalysis variables by month

# Standardize Mean Temperature

#%%
ds_tas_gb_mnt = ds_tas.groupby('time.month')
ds_tas_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_tas_gbmonth_mean = ds_tas.groupby('time.month').mean('time')
ds_tas_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_tas_gbmonth_std = ds_tas.groupby('time.month').std('time')
ds_tas_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_tas_month_anom = ds_tas_gb_mnt - ds_tas_gbmonth_mean
ds_tas_month_anom

#%%
ds_tas_anom_gb_mnt = ds_tas_month_anom.groupby('time.month')
ds_tas_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_tas_standardized = ds_tas_anom_gb_mnt / ds_tas_gbmonth_std
ds_tas_standardized = ds_tas_standardized.drop('month', dim=None)
ds_tas_standardized


#%%
# Standadized Maximum Temperature

#%%
ds_tasmax_gb_mnt = ds_tasmax.groupby('time.month')
ds_tasmax_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_tasmax_gbmonth_mean = ds_tasmax.groupby('time.month').mean('time')
ds_tasmax_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_tasmax_gbmonth_std = ds_tasmax.groupby('time.month').std('time')
ds_tasmax_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_tasmax_month_anom = ds_tasmax_gb_mnt - ds_tasmax_gbmonth_mean
ds_tasmax_month_anom

#%%
ds_tasmax_anom_gb_mnt = ds_tasmax_month_anom.groupby('time.month')
ds_tasmax_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_tasmax_standardized = ds_tasmax_anom_gb_mnt / ds_tasmax_gbmonth_std
ds_tasmax_standardized = ds_tasmax_standardized.drop('month', dim=None)
ds_tasmax_standardized


#%%
# Standardize Minimum Temperature


#%%
ds_tasmin_gb_mnt = ds_tasmin.groupby('time.month')
ds_tasmin_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_tasmin_gbmonth_mean = ds_tasmin.groupby('time.month').mean('time')
ds_tasmin_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_tasmin_gbmonth_std = ds_tasmin.groupby('time.month').std('time')
ds_tasmin_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_tasmin_month_anom = ds_tasmin_gb_mnt - ds_tasmin_gbmonth_mean
ds_tasmin_month_anom

#%%
ds_tasmin_anom_gb_mnt = ds_tasmin_month_anom.groupby('time.month')
ds_tasmin_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_tasmin_standardized = ds_tasmin_anom_gb_mnt / ds_tasmin_gbmonth_std
ds_tasmin_standardized = ds_tasmin_standardized.drop('month', dim=None)
ds_tasmin_standardized


#%%
# Standardized Specific Humidity

#%%
ds_huss_gb_mnt = ds_huss.groupby('time.month')
ds_huss_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_huss_gbmonth_mean = ds_huss.groupby('time.month').mean('time')
ds_huss_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_huss_gbmonth_std = ds_huss.groupby('time.month').std('time')
ds_huss_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_huss_month_anom = ds_huss_gb_mnt - ds_huss_gbmonth_mean
ds_huss_month_anom

#%%
ds_huss_anom_gb_mnt = ds_huss_month_anom.groupby('time.month')
ds_huss_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_huss_standardized = ds_huss_anom_gb_mnt / ds_huss_gbmonth_std
ds_huss_standardized = ds_huss_standardized.drop('month', dim=None)
ds_huss_standardized


#%%
# Standardized Geopotential at 500hpa


#%%
ds_zg500_gb_mnt = ds_zg500.groupby('time.month')
ds_zg500_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_zg500_gbmonth_mean = ds_zg500.groupby('time.month').mean('time')
ds_zg500_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_zg500_gbmonth_std = ds_zg500.groupby('time.month').std('time')
ds_zg500_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_zg500_month_anom = ds_zg500_gb_mnt - ds_zg500_gbmonth_mean
ds_zg500_month_anom

#%%
ds_zg500_anom_gb_mnt = ds_zg500_month_anom.groupby('time.month')
ds_zg500_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_zg500_standardized = ds_zg500_anom_gb_mnt / ds_zg500_gbmonth_std
ds_zg500_standardized = ds_zg500_standardized.drop('month', dim=None)
ds_zg500_standardized


#%%
# Standardized Geopotential at 1000hpa


#%%
ds_zg1000_gb_mnt = ds_zg1000.groupby('time.month')
ds_zg1000_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_zg1000_gbmonth_mean = ds_zg1000.groupby('time.month').mean('time')
ds_zg1000_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_zg1000_gbmonth_std = ds_zg1000.groupby('time.month').std('time')
ds_zg1000_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_zg1000_month_anom = ds_zg1000_gb_mnt - ds_zg1000_gbmonth_mean
ds_zg1000_month_anom

#%%
ds_zg1000_anom_gb_mnt = ds_zg1000_month_anom.groupby('time.month')
ds_zg1000_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_zg1000_standardized = ds_zg1000_anom_gb_mnt / ds_zg1000_gbmonth_std
ds_zg1000_standardized = ds_zg1000_standardized.drop('month', dim=None)
ds_zg1000_standardized


#%%
# Standardized Mean Sea Level Pressure


#%%
ds_psl_gb_mnt = ds_psl.groupby('time.month')
ds_psl_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_psl_gbmonth_mean = ds_psl.groupby('time.month').mean('time')
ds_psl_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_psl_gbmonth_std = ds_psl.groupby('time.month').std('time')
ds_psl_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_psl_month_anom = ds_psl_gb_mnt - ds_psl_gbmonth_mean
ds_psl_month_anom

#%%
ds_psl_anom_gb_mnt = ds_psl_month_anom.groupby('time.month')
ds_psl_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_psl_standardized = ds_psl_anom_gb_mnt / ds_psl_gbmonth_std
ds_psl_standardized = ds_psl_standardized.drop('month', dim=None)
ds_psl_standardized


#%%
# Standardized Eastward Wind


#%%
ds_uas_gb_mnt = ds_uas.groupby('time.month')
ds_uas_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_uas_gbmonth_mean = ds_uas.groupby('time.month').mean('time')
ds_uas_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_uas_gbmonth_std = ds_uas.groupby('time.month').std('time')
ds_uas_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_uas_month_anom = ds_uas_gb_mnt - ds_uas_gbmonth_mean
ds_uas_month_anom

#%%
ds_uas_anom_gb_mnt = ds_uas_month_anom.groupby('time.month')
ds_uas_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_uas_standardized = ds_uas_anom_gb_mnt / ds_uas_gbmonth_std
ds_uas_standardized = ds_uas_standardized.drop('month', dim=None)
ds_uas_standardized


#%%
# Standardized Northward Wind

#%%
ds_vas_gb_mnt = ds_vas.groupby('time.month')
ds_vas_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_vas_gbmonth_mean = ds_vas.groupby('time.month').mean('time')
ds_vas_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_vas_gbmonth_std = ds_vas.groupby('time.month').std('time')
ds_vas_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_vas_month_anom = ds_vas_gb_mnt - ds_vas_gbmonth_mean
ds_vas_month_anom

#%%
ds_vas_anom_gb_mnt = ds_vas_month_anom.groupby('time.month')
ds_vas_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_vas_standardized = ds_vas_anom_gb_mnt / ds_vas_gbmonth_std
ds_vas_standardized = ds_vas_standardized.drop('month', dim=None)
ds_vas_standardized


#%%
# Standardized Era5 Precipitation


#%%
ds_era5_pr_gb_mnt = ds_era5_pr.groupby('time.month')
ds_era5_pr_gb_mnt

#%%
# get the groupby month mean of years 2001 to 2020
ds_era5_pr_gbmonth_mean = ds_era5_pr.groupby('time.month').mean('time')
ds_era5_pr_gbmonth_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_era5_pr_gbmonth_std = ds_era5_pr.groupby('time.month').std('time')
ds_era5_pr_gbmonth_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_era5_pr_month_anom = ds_era5_pr_gb_mnt - ds_era5_pr_gbmonth_mean
ds_era5_pr_month_anom

#%%
ds_era5_pr_anom_gb_mnt = ds_era5_pr_month_anom.groupby('time.month')
ds_era5_pr_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_era5_pr_standardized = ds_era5_pr_anom_gb_mnt / ds_era5_pr_gbmonth_std
ds_era5_pr_standardized = ds_era5_pr_standardized.drop('month', dim=None)
ds_era5_pr_standardized


#%%
# ds_standardized
# ds_tas_standardized
# ds_tasmax_standardized
# ds_tasmin_standardized
# ds_huss_standardized
# ds_zg500_standardized
# ds_zg1000_standardized
# ds_psl_standardized
# ds_uas_standardized
# ds_vas_standardized
# ds_era5_pr_standardized


#%%
# Save ERA5 xarray scaler dataset (2001 to 2020)

ds_tas_gbmonth_mean = ds_tas_gbmonth_mean.rename({'tas':'tas_mean'})
ds_tas_gbmonth_std = ds_tas_gbmonth_std.rename({'tas':'tas_std'})

ds_tasmax_gbmonth_mean = ds_tasmax_gbmonth_mean.rename({'tasmax':'tasmax_mean'})
ds_tasmax_gbmonth_std = ds_tasmax_gbmonth_std.rename({'tasmax':'tasmax_std'})

ds_tasmin_gbmonth_mean = ds_tasmin_gbmonth_mean.rename({'tasmin':'tasmin_mean'})
ds_tasmin_gbmonth_std = ds_tasmin_gbmonth_std.rename({'tasmin':'tasmin_std'})

ds_huss_gbmonth_mean = ds_huss_gbmonth_mean.rename({'q':'q_mean'})
ds_huss_gbmonth_std = ds_huss_gbmonth_std.rename({'q':'q_std'})

ds_zg500_gbmonth_mean = ds_zg500_gbmonth_mean.rename({'z500':'z500_mean'})
ds_zg500_gbmonth_std = ds_zg500_gbmonth_std.rename({'z500':'z500_std'})

ds_zg1000_gbmonth_mean = ds_zg1000_gbmonth_mean.rename({'z1000':'z1000_mean'})
ds_zg1000_gbmonth_std = ds_zg1000_gbmonth_std.rename({'z1000':'z1000_std'})

ds_psl_gbmonth_mean = ds_psl_gbmonth_mean.rename({'psl':'psl_mean'})
ds_psl_gbmonth_std = ds_psl_gbmonth_std.rename({'psl':'psl_std'})

ds_uas_gbmonth_mean = ds_uas_gbmonth_mean.rename({'uas':'uas_mean'})
ds_uas_gbmonth_std = ds_uas_gbmonth_std.rename({'uas':'uas_std'})

ds_vas_gbmonth_mean = ds_vas_gbmonth_mean.rename({'vas':'vas_mean'})
ds_vas_gbmonth_std = ds_vas_gbmonth_std.rename({'vas':'vas_std'})

ds_era5_pr_gbmonth_mean = ds_era5_pr_gbmonth_mean.rename({'era5_pr':'era5_pr_mean'})
ds_era5_pr_gbmonth_std = ds_era5_pr_gbmonth_std.rename({'era5_pr':'era5_pr_std'})

#%%

da_tas_gbmonth_mean = ds_tas_gbmonth_mean.tas_mean.load()
da_tas_gbmonth_std = ds_tas_gbmonth_std.tas_std.load()

da_tasmax_gbmonth_mean = ds_tasmax_gbmonth_mean.tasmax_mean.load()
da_tasmax_gbmonth_std = ds_tasmax_gbmonth_std.tasmax_std.load()

da_tasmin_gbmonth_mean = ds_tasmin_gbmonth_mean.tasmin_mean.load()
da_tasmin_gbmonth_std = ds_tasmin_gbmonth_std.tasmin_std.load()

da_huss_gbmonth_mean = ds_huss_gbmonth_mean.q_mean.load()
da_huss_gbmonth_std = ds_huss_gbmonth_std.q_std.load()

da_zg500_gbmonth_mean = ds_zg500_gbmonth_mean.z500_mean.load()
da_zg500_gbmonth_std = ds_zg500_gbmonth_std.z500_std.load()

da_zg1000_gbmonth_mean = ds_zg1000_gbmonth_mean.z1000_mean.load()
da_zg1000_gbmonth_std = ds_zg1000_gbmonth_std.z1000_std.load()

da_psl_gbmonth_mean = ds_psl_gbmonth_mean.psl_mean.load()
da_psl_gbmonth_std = ds_psl_gbmonth_std.psl_std.load()

da_uas_gbmonth_mean = ds_uas_gbmonth_mean.uas_mean.load()
da_uas_gbmonth_std = ds_uas_gbmonth_std.uas_std.load()

da_vas_gbmonth_mean = ds_vas_gbmonth_mean.vas_mean.load()
da_vas_gbmonth_std = ds_vas_gbmonth_std.vas_std.load()

da_era5_pr_gbmonth_mean = ds_era5_pr_gbmonth_mean.era5_pr_mean.load()
da_era5_pr_gbmonth_std = ds_era5_pr_gbmonth_std.era5_pr_std.load()


#%%
ds_era5_gbmonth_mean_std = da_tas_gbmonth_mean.to_dataset()
ds_era5_gbmonth_mean_std['tas_std'] = da_tas_gbmonth_std

ds_era5_gbmonth_mean_std['tasmax_mean'] = da_tasmax_gbmonth_mean
ds_era5_gbmonth_mean_std['tasmax_std'] = da_tasmax_gbmonth_std

ds_era5_gbmonth_mean_std['tasmin_mean'] = da_tasmin_gbmonth_mean
ds_era5_gbmonth_mean_std['tasmin_std'] = da_tasmin_gbmonth_std

ds_era5_gbmonth_mean_std['q_mean'] = da_huss_gbmonth_mean
ds_era5_gbmonth_mean_std['q_std'] = da_huss_gbmonth_std

ds_era5_gbmonth_mean_std['z500_mean'] = da_zg500_gbmonth_mean
ds_era5_gbmonth_mean_std['z500_std'] = da_zg500_gbmonth_std

ds_era5_gbmonth_mean_std['z1000_mean'] = da_zg1000_gbmonth_mean
ds_era5_gbmonth_mean_std['z1000_std'] = da_zg1000_gbmonth_std

ds_era5_gbmonth_mean_std['psl_mean'] = da_psl_gbmonth_mean
ds_era5_gbmonth_mean_std['psl_std'] = da_psl_gbmonth_std

ds_era5_gbmonth_mean_std['uas_mean'] = da_uas_gbmonth_mean
ds_era5_gbmonth_mean_std['uas_std'] = da_uas_gbmonth_std

ds_era5_gbmonth_mean_std['vas_mean'] = da_vas_gbmonth_mean
ds_era5_gbmonth_mean_std['vas_std'] = da_vas_gbmonth_std

ds_era5_gbmonth_mean_std['era5_pr_mean'] = da_era5_pr_gbmonth_mean
ds_era5_gbmonth_mean_std['era5_pr_std'] = da_era5_pr_gbmonth_std

ds_era5_gbmonth_mean_std

#%%
# Save GPM IMERGE xarray scaler dataset (2001 to 2020)
ds_gbmonth_mean = ds_gbmonth_mean.rename({'pr':'pr_mean'})
ds_gbmonth_std = ds_gbmonth_std.rename({'pr':'pr_std'})

da_gbmonth_mean = ds_gbmonth_mean.pr_mean.load()
da_gbmonth_std = ds_gbmonth_std.pr_std.load()

ds_gpm_gbmonth_mean_std = da_gbmonth_mean.to_dataset()
ds_gpm_gbmonth_mean_std['pr_std'] = da_gbmonth_std

ds_gpm_gbmonth_mean_std


#%%
ds_era5_gbmonth_mean_std.to_netcdf(scaler_path / 'era5_dl_downscale_gbmonth_mean_std_2001_2020.nc')
ds_gpm_gbmonth_mean_std.to_netcdf(scaler_path / 'gpm_imerge_dl_downscale_gbmonth_mean_std_2001_2020.nc')


#%%

# Selected months
months = [1,2,12]

# %%
ds_standardized_months = ds_standardized.sel(
    time = np.in1d(ds_standardized['time.month'], months))

ds_standardized_months


# %%
ds_tas_standardized_months = ds_tas_standardized.sel(
    time = np.in1d(ds_tas_standardized['time.month'], months))

ds_tas_standardized_months


# %%
ds_tasmax_standardized_months = ds_tasmax_standardized.sel(
    time = np.in1d(ds_tasmax_standardized['time.month'], months))

ds_tasmax_standardized_months


# %%
ds_tasmin_standardized_months = ds_tasmin_standardized.sel(
    time = np.in1d(ds_tasmin_standardized['time.month'], months))

ds_tasmin_standardized_months


# %%
ds_huss_standardized_months = ds_huss_standardized.sel(
    time = np.in1d(ds_huss_standardized['time.month'], months))

ds_huss_standardized_months


# %%
ds_zg500_standardized_months = ds_zg500_standardized.sel(
    time = np.in1d(ds_zg500_standardized['time.month'], months))

ds_zg500_standardized_months


# %%
ds_zg1000_standardized_months = ds_zg1000_standardized.sel(
    time = np.in1d(ds_zg1000_standardized['time.month'], months))

ds_zg1000_standardized_months


# %%
ds_psl_standardized_months = ds_psl_standardized.sel(
    time = np.in1d(ds_psl_standardized['time.month'], months))

ds_psl_standardized_months


# %%
ds_uas_standardized_months = ds_uas_standardized.sel(
    time = np.in1d(ds_uas_standardized['time.month'], months))

ds_uas_standardized_months


# %%
ds_vas_standardized_months = ds_vas_standardized.sel(
    time = np.in1d(ds_vas_standardized['time.month'], months))

ds_vas_standardized_months


# %%
ds_era5_pr_standardized_months = ds_era5_pr_standardized.sel(
    time = np.in1d(ds_era5_pr_standardized['time.month'], months))

ds_era5_pr_standardized_months


#%%

# ds_standardized_months
# ds_tas_standardized_months
# ds_tasmax_standardized_months
# ds_tasmin_standardized_months
# ds_huss_standardized_months
# ds_zg500_standardized_months
# ds_zg1000_standardized_months
# ds_psl_standardized_months
# ds_uas_standardized_months
# ds_vas_standardized_months
# ds_era5_pr_standardized_months

#%%

# da = ds.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_tas = ds_tas.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_tasmax = ds_tasmax.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_tasmin = ds_tasmin.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_huss = ds_huss.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_zg500 = ds_zg500.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_zg1000 = ds_zg1000.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_psl = ds_psl.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_uas = ds_uas.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_vas = ds_vas.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da_era5_pr = ds_era5_pr.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
# da

#%%

# Step 1: convert dataarrays to numpy arrays

da_arr = ds_standardized_months.pr.load().data
da_tas_arr = ds_tas_standardized_months.tas.load().data
da_tasmax_arr = ds_tasmax_standardized_months.tasmax.load().data
da_tasmin_arr = ds_tasmin_standardized_months.tasmin.load().data
da_huss_arr = ds_huss_standardized_months.q.load().data
da_zg500_arr = ds_zg500_standardized_months.z500.load().data
da_zg1000_arr = ds_zg1000_standardized_months.z1000.load().data
da_psl_arr = ds_psl_standardized_months.psl.load().data
da_uas_arr = ds_uas_standardized_months.uas.load().data
da_vas_arr = ds_vas_standardized_months.vas.load().data

print(da_arr[0])
print(da_arr.shape)


#%%
# # Complete 12 months Time series

# # Step 1: convert dataarrays to numpy arrays

# da_arr = ds_standardized.pr.load().data
# da_tas_arr = ds_tas_standardized.tas.load().data
# da_tasmax_arr = ds_tasmax_standardized.tasmax.load().data
# da_tasmin_arr = ds_tasmin_standardized.tasmin.load().data
# da_huss_arr = ds_huss_standardized.q.load().data
# da_zg500_arr = ds_zg500_standardized.z500.load().data
# da_zg1000_arr = ds_zg1000_standardized.z1000.load().data
# da_psl_arr = ds_psl_standardized.psl.load().data
# da_uas_arr = ds_uas_standardized.uas.load().data
# da_vas_arr = ds_vas_standardized.vas.load().data

# print(da_arr[0])
# print(da_arr.shape)



#%%
print(da_tas_arr[0])
print(da_tas_arr.shape)

#%%
# Step 2: Flatten the numpy array

da_arr_flat = da_arr.flatten()
da_tas_arr_flat = da_tas_arr.flatten()
da_tasmax_arr_flat = da_tasmax_arr.flatten()
da_tasmin_arr_flat = da_tasmin_arr.flatten()
da_huss_arr_flat = da_huss_arr.flatten()
da_zg500_arr_flat = da_zg500_arr.flatten()
da_zg1000_arr_flat = da_zg1000_arr.flatten()
da_psl_arr_flat = da_psl_arr.flatten()
da_uas_arr_flat = da_uas_arr.flatten()
da_vas_arr_flat = da_vas_arr.flatten()

print(da_tas_arr_flat[0])
da_tas_arr_flat.shape


#%%
# Step 3: Stack flattened Predictors 

clim_stack = np.stack((
    da_tas_arr_flat, da_tasmax_arr_flat, da_tasmin_arr_flat, 
    da_huss_arr_flat, da_zg500_arr_flat, da_zg1000_arr_flat, 
    da_psl_arr_flat, da_uas_arr_flat, da_vas_arr_flat
    ), axis=-1).astype(np.float32)

print(clim_stack.shape)
clim_stack[0]


#%%
# Convert training data to normal distribution
# if you are using only monthly standardized data

quantile = QuantileTransformer(output_distribution='normal', random_state=seed)

#%%

quantileX = quantile

# fit data to normal distribution
quantileX.fit(clim_stack)

#%%
# transform data to normal distribution
dataX_quantile = quantileX.transform(clim_stack)

print(dataX_quantile[:5])


#%%
# Step 4: Reshape Stacked Arrays to [samples, lat, lon, channels (predictors)]

clim_stack = dataX_quantile.reshape((
    da_tas_arr.shape[0], da_tas_arr.shape[1], da_tas_arr.shape[2], 9))

print(clim_stack.shape)


#%%
da_arr_flat_reshaped = da_arr_flat.reshape(-1, 1)
print(da_arr_flat_reshaped[0:5])
print(da_arr_flat_reshaped.shape)

#%%
quantileY = quantile

quantileY.fit(da_arr_flat_reshaped)

#%%
# transform data to normal distribution
dataY_quantile = quantileY.transform(da_arr_flat_reshaped)

print(dataY_quantile[:5])
print(dataY_quantile.shape)


#%%
# Reshape 2D target back to 1D target
dataY_quantile = dataY_quantile.reshape(da_arr_flat.shape)

dataY_quantile = dataY_quantile.reshape(da_arr.shape)

print(dataY_quantile[0:5])
print(dataY_quantile.shape)

#%%

# split data [train, val, test] = [80 : 15 : 5]

# size = int(len(df_copy)*0.8)
size = int(len(da_arr)*0.8)
size_2 = int(len(da_arr)*0.95)

#%%
dataX_train = clim_stack[:size]
dataX_val = clim_stack[size:size_2]
dataX_test = clim_stack[size_2:]

print(dataX_train.shape)
print(dataX_val.shape)
print(dataX_test.shape)

#%%

dataY_train = dataY_quantile[:size]
dataY_val = dataY_quantile[size:size_2]
dataY_test = dataY_quantile[size_2:]

# dataY_train = da_arr[:size]
# dataY_val = da_arr[size:size_2]
# dataY_test = da_arr[size_2:]

print(dataY_train.shape)
print(dataY_val.shape)
print(dataY_test.shape)


#%%

# BATCH_SIZE = 32
# GPUS = ["GPU:0","GPU:1","GPU:2","GPU:3"]


# strategy = tf.distribute.MirroredStrategy( GPUS )
# print('Number of devices: %d' % strategy.num_replicas_in_sync) 

# batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

# #%%
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# #%%
# # Mult-GPU Training CNN 2D Model

# # Prepare the training and validation time-series data
# BATCH_SIZE = 32
# BUFFER_SIZE = 150

# train_data = tf.data.Dataset.from_tensor_slices((dataX_train, dataY_train))
# train_data = train_data.cache().shuffle(BATCH_SIZE).batch(BATCH_SIZE).repeat()
# val_data = tf.data.Dataset.from_tensor_slices((dataX_val, dataY_val))
# val_data = val_data.batch(BATCH_SIZE).repeat()


#%%
# CNN 2D Model

# Prepare the training and validation time-series data
BATCH_SIZE = 256
BUFFER_SIZE = 150

train_data = tf.data.Dataset.from_tensor_slices((dataX_train, dataY_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data = tf.data.Dataset.from_tensor_slices((dataX_val, dataY_val))
val_data = val_data.batch(BATCH_SIZE).repeat()


#%%

input_shape = (dataX_train.shape[1], dataX_train.shape[2], dataX_train.shape[3])
output_shape = dataY_test[0].shape
print(input_shape)
print(output_shape)

#%%

# Define the CNN model

CNN_model = Sequential()

CNN_model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))
CNN_model.add(MaxPool2D(pool_size=(2, 2)))

CNN_model.add(Conv2D(64, (3, 3), padding='same'))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))

CNN_model.add(Conv2D(64, (3, 3), padding='same'))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))
CNN_model.add(MaxPool2D(pool_size=(2, 2)))

CNN_model.add(Conv2D(128, (3, 3), padding='same'))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))
CNN_model.add(MaxPool2D(pool_size=(2, 2)))

CNN_model.add(Conv2D(256, (3, 3), padding='same'))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))
CNN_model.add(MaxPool2D(pool_size=(2, 2)))
CNN_model.add(Dropout(0.25))

CNN_model.add(Flatten())
CNN_model.add(Dense(128))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))
CNN_model.add(Dropout(0.25))

CNN_model.add(Dense(128))
CNN_model.add(BatchNormalization())
CNN_model.add(Activation('relu'))
CNN_model.add(Dropout(0.25))

CNN_model.add(Dense(101441))
CNN_model.add(Reshape(output_shape))

CNN_model.compile(optimizer='adam', loss='mse')
CNN_model.summary()


#%%
from pathlib import Path

DATA_PATH = Path("/datadrive/project/")

model_path = DATA_PATH / "model_outputs"
model_path.mkdir(exist_ok=True)


# the best weights are stored in model path
# models_path = model_path / "CNN_climate_downscale_moonsoon_5months_678910_quant_target.h5"
models_path = model_path / "CNN_climate_downscale_moonsoon_season_DJF_quant_target_quant.h5"


#%%
# Configure the model and start training with early stopping and checkpointing

EVALUATION_INTERVAL = 150
EPOCHS = 1000
history = CNN_model.fit(train_data, epochs=EPOCHS,
steps_per_epoch=EVALUATION_INTERVAL,validation_data=val_data,
validation_steps=50,verbose =1,callbacks =[EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'),
    ModelCheckpoint(models_path,monitor='val_loss', 
    save_best_only=True, mode='min', verbose=0)])


#%%

# Load the best weights into the model
Trained_model = tf.keras.models.load_model(models_path)

#%%
# Plot the loss and val_loss against the epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],color='red')
plt.title('Model loss',size=14,fontweight='bold')
plt.ylabel('Loss',size=12,fontweight='semibold')
plt.xlabel('Epoch',size=12,fontweight='semibold')
plt.legend(['train loss', 'validation loss'])
plt.figure(figsize=(16,9))
plt.show()


#%%
# save model train loss and val loss history to CSV
df_history = pd.DataFrame.from_records(
    [history.history['loss'],history.history['val_loss']], 
    index=['train_loss', 'val_loss']).transpose()

df_history.to_csv(
    model_path / "CNN_climate_downscale_moonsoon_season_DJF_quant_target_quant_history.csv", 
    index=False)

# "CNN_climate_downscale_moonsoon_5months_678910_quant_target_history.csv"

#%%
# Check the model summary
Trained_model.summary()

#%%
# Prepare test dataset
x_test_multi, y_test_multi = custom_ts_multi_data_prep(
    validateX, validateY, 0, None, hist_window, horizon)
print ('Single window of past history')
print(x_test_multi[0])
print ('\n Target horizon')
print (y_test_multi[0])
print(len(y_test_multi))

#%%
# predictions for CNN

predictions = []
for i in x_test_multi:
    # predict the daily streamflow
    i_reshape = i.reshape(1, i.shape[0], i.shape[1])
    Predicted_results = Trained_model.predict(i_reshape)
    # store the predictions
    predictions.append(Predicted_results)
print(predictions[0])

#%%
# convert prediction list to numpy array
predictions = np.array(predictions).reshape(-1, 1)
print(predictions.shape)

#%%

# Rescale the predicted values back to the original scale.

predictions_Inv_trans = scaler.inverse_transform(predictions)
print(predictions_Inv_trans[0])
print(predictions_Inv_trans.shape)
print(len(predictions_Inv_trans))

#%%
# Define the time-series evaluation function

def timeseries_evaluation_metrics_func(y_true, y_pred):
    kgeprime, r, gamma, beta = he.kgeprime(y_pred, y_true)
    kge, r, alpha, beta = he.kge(y_pred, y_true)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    print(f'NSE is : {he.nse(np.array(y_pred), np.array(y_true))}')
    print(f"KGE' is : {kgeprime}")
    print(f'r is : {r}, gamma is : {gamma}')
    print(f'KGE is : {kge}')
    print(f'alpha is : {alpha}, beta is : {beta}')
    print(f'PBIAS is : {he.pbias(np.array(y_pred), np.array(y_true))}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MARE is : {he.mare(np.array(y_pred), np.array(y_true))}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}') 
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
timeseries_evaluation_metrics_func(validate['Q'].values.reshape(-1, 1),predictions_Inv_trans)


#%%
# Plot the actual versus predicted values.
plt.plot( list(validate['Q']))
plt.plot( list(predictions_Inv_trans),color='red')
plt.title("Actual vs Predicted")
plt.ylabel("Water Level")
plt.legend(('Actual','predicted'))
plt.show()

#%%
# save model train loss and val loss history to CSV
df_history = pd.DataFrame.from_records(
    [history.history['loss'],history.history['val_loss']], 
    index=['train_loss', 'val_loss']).transpose()

df_history.to_csv(
    model_path / "CNN_climate_downscale_moonsoon_5months_678910_4pca_history.csv", 
    index=False)


#%%

from pickle import dump

# create directory for dataset transformers
DATA_PATH = Path("/datadrive/project/")


scaler_path = DATA_PATH / "scaler_dir"
scaler_path.mkdir(exist_ok=True)


#%%
# save the quantile transformer for Features
# dump(quantileX, open(scaler_path / 'quantileX_5months.pkl', 'wb'))
dump(quantileX, open(scaler_path / 'quantileX_season_DJF.pkl', 'wb'))


#%%
# save the quantile transformer for Target
# dump(quantileY, open(scaler_path / 'quantileY_5months.pkl', 'wb'))
dump(quantileY, open(scaler_path / 'quantileY_season_DJF.pkl', 'wb'))

#%%
from pickle import load

# load the quantile transformer
new_quantileY = load(open(scaler_path / 'quantileY_5months.pkl', 'rb'))

#%%
print(tf.__version__)

#%%

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

for d in devices:
    t = d.device_type
    name = d.physical_device_desc
    l = [item.split(':',1) for item in name.split(", ")]
    name_attr = dict([x for x in l if len(x)==2])
    dev = name_attr.get('name', 'Unnamed device')
    print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")



#%%
import tensorflow as tf

BATCH_SIZE = 32
GPUS = ["GPU:0","GPU:1","GPU:2","GPU:3"]


strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import time

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

EPOCHS = 5
LR = 0.001 

tf.get_logger().setLevel('ERROR')

start = time.time()
with strategy.scope():
    model = tf.keras.applications.InceptionResNetV2(weights=None, classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=EPOCHS)

elapsed = time.time()-start
print (f'Training time: {hms_string(elapsed)}')





#%%
#%%
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 32
GPUS = ["GPU:0","GPU:1","GPU:2","GPU:3"]

def process(image, label):
    image = tf.image.resize(image, [299, 299]) / 255.0
    return image, label

strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

#%%

import tensorflow as tf
import tensorflow_datasets as tfds
import time

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

EPOCHS = 5
LR = 0.001 

tf.get_logger().setLevel('ERROR')

start = time.time()
with strategy.scope():
    model = tf.keras.applications.InceptionResNetV2(weights=None, classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=EPOCHS)

elapsed = time.time()-start
print (f'Training time: {hms_string(elapsed)}')

#%%

































#%%

CNN_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',
input_shape=(
    dataX_train.shape[1], dataX_train.shape[2], dataX_train.shape[3])))
CNN_model.add(MaxPool2D(pool_size=2))
CNN_model.add(Dropout(0.2))
CNN_model.add(Flatten())
CNN_model.add(Dense(30, activation='relu'))
CNN_model.add(Dropout(0.2))
# CNN_model.add(Dense(horizon))
CNN_model.add(Reshape(output_shape))
CNN_model.compile(optimizer='adam', loss='mse')



#%%
CNN_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(input_shape)))
CNN_model.add(MaxPool2D(pool_size=2))
CNN_model.add(Dropout(0.2))
CNN_model.add(Flatten())
CNN_model.add(Dense(30, activation='relu'))
CNN_model.add(Dropout(0.2))
# CNN_model.add(Dense(horizon))
CNN_model.add(Reshape(output_shape))
CNN_model.compile(optimizer='adam', loss='mse')


#%%












CNN_model = Sequential()
CNN_model.add(Conv2D(filters=64, kernel_size=3, activation='relu',
input_shape=(
    dataX_train.shape[1], dataX_train.shape[2], dataX_train.shape[3])))
CNN_model.add(MaxPool2D(pool_size=2))
CNN_model.add(Dropout(0.2))
CNN_model.add(Flatten())
CNN_model.add(Dense(30, activation='relu'))
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(horizon))
CNN_model.compile(optimizer='adam', loss='mse')


CNN_model = Sequential()
CNN_model.add(Conv2D(filters=64, kernel_size=3, activation='relu',
input_shape=(
    dataX_train.shape[1], dataX_train.shape[2], dataX_train.shape[3])))
CNN_model.add(MaxPool2D(pool_size=2))
CNN_model.add(Dropout(0.2))
CNN_model.add(Flatten())
CNN_model.add(Dense(30, activation='relu'))
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(horizon))
CNN_model.compile(optimizer='adam', loss='mse')


model = Sequential()
CNN_model.add(Conv2D(
    filters=512, kernel_size=(3,3), padding='same', 
    activation='relu', input_shape=(113, 145, 9)))
CNN_model.add(MaxPool2D(pool_size=2))
CNN_model.add(Dropout(0.2))
CNN_model.add(Conv2D(64, (1,1), activation='relu'))
# summarize model
model.summary()



#%%
# the best weights are stored in model path
model_path = r'/home/jefire/project/water/models/jiderebode_models/CNN_Multi_Jiderebode_Q_1step_30days_gpm.h5'


#%%
# Configure the model and start training with early stopping and checkpointing

EVALUATION_INTERVAL = 150
EPOCHS = 100
history = CNN_model.fit(train_data_multi, epochs=EPOCHS,
steps_per_epoch=EVALUATION_INTERVAL,validation_data=val_data_multi,
validation_steps=50,verbose =1,callbacks =[tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'),
    tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', 
    save_best_only=True, mode='min', verbose=0)])

#%%
# Load the best weights into the model
Trained_model = tf.keras.models.load_model(models_path)

#%%
# Plot the loss and val_loss against the epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],color='red')
plt.title('Model loss',size=14,fontweight='bold')
plt.ylabel('Loss',size=12,fontweight='semibold')
plt.xlabel('Epoch',size=12,fontweight='semibold')
plt.legend(['train loss', 'validation loss'])
plt.figure(figsize=(16,9))
plt.show()

#%%
# Check the model summary
Trained_model.summary()

# Prepare test dataset
x_test_multi, y_test_multi = custom_ts_multi_data_prep(
    validateX, validateY, 0, None, hist_window, horizon)
print ('Single window of past history')
print(x_test_multi[0])
print ('\n Target horizon')
print (y_test_multi[0])
print(len(y_test_multi))

#%%
# predictions for CNN

predictions = []
for i in x_test_multi:
    # predict the daily streamflow
    i_reshape = i.reshape(1, i.shape[0], i.shape[1])
    Predicted_results = Trained_model.predict(i_reshape)
    # store the predictions
    predictions.append(Predicted_results)
print(predictions[0])

#%%
# convert prediction list to numpy array
predictions = np.array(predictions).reshape(-1, 1)
print(predictions.shape)

#%%

# Rescale the predicted values back to the original scale.

predictions_Inv_trans = scaler.inverse_transform(predictions)
print(predictions_Inv_trans[0])
print(predictions_Inv_trans.shape)
print(len(predictions_Inv_trans))

#%%
# Define the time-series evaluation function

def timeseries_evaluation_metrics_func(y_true, y_pred):
    kgeprime, r, gamma, beta = he.kgeprime(y_pred, y_true)
    kge, r, alpha, beta = he.kge(y_pred, y_true)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    print(f'NSE is : {he.nse(np.array(y_pred), np.array(y_true))}')
    print(f"KGE' is : {kgeprime}")
    print(f'r is : {r}, gamma is : {gamma}')
    print(f'KGE is : {kge}')
    print(f'alpha is : {alpha}, beta is : {beta}')
    print(f'PBIAS is : {he.pbias(np.array(y_pred), np.array(y_true))}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MARE is : {he.mare(np.array(y_pred), np.array(y_true))}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}') 
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
timeseries_evaluation_metrics_func(validate['Q'].values.reshape(-1, 1),predictions_Inv_trans)


#%%
# Plot the actual versus predicted values.
plt.plot( list(validate['Q']))
plt.plot( list(predictions_Inv_trans),color='red')
plt.title("Actual vs Predicted")
plt.ylabel("Water Level")
plt.legend(('Actual','predicted'))
plt.show()

#%%
# save GPM IMERGE train loss and val loss to CSV
df_history = pd.DataFrame.from_records([history.history['loss'],history.history['val_loss']], index=['train_loss', 'val_loss']).transpose()
df_history.to_csv('/home/jefire/project/water/models/jiderebode_models/CNN_Multi_Jiderebode_Q_1step_30days_gpm_history.csv', index=False)



































#%%
# Define a function to prepare multivariate data
# so that it is suitable for a time series.

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indicex = range(i-window, i)
        X.append(dataset[indicex])
        indicey = range(i-1, i-1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)


#%%
# let’s allow the model to see / train on the past 90 days of data 
# and try to forecast the next 1 day of results. Hence, use horizon = 1
# This is called Single step Forecast 
# This step only applies to ConcLSTM2D only!

hist_window = 90
horizon = 1

x_train_multi, y_train_multi = custom_ts_multi_data_prep(
    dataX_train, dataY_train, 0, None, hist_window, horizon)


#%%
# Prepare validation dataset
x_val_multi, y_val_multi= custom_ts_multi_data_prep(
    dataX_val, dataY_val, 0, None, hist_window, horizon)

#%%
# Prepare test dataset
x_test_multi, y_test_multi = custom_ts_multi_data_prep(
    dataX_test, dataY_test, 0, None, hist_window, horizon)


#%%
# Applying Log Transformation of Features and Target datasets

dataX_log = np.log(clim_stack)

#%%
# Step 4: Reshape Stacked Arrays to [samples, lat, lon, channels (predictors)]
dataX_log = dataX_log.reshape((7305, 113, 145, 9))
dataX_log.shape

#%%
dataY_log = np.log(da_arr.flatten())
dataY_log.shape

#%%
# Step 4: Reshape Stacked Arrays to [samples, lat, lon, channels (predictors)]
dataY_log = dataY_log.reshape((7305, 113, 145))
dataY_log.shape


#%%
# Apply inverse log transformation to bring data to original scale
# yhat = np.exp(predictions.flatten())
# yhat[0]



# %%
# Prepare CovLSTM2D training and validation time-series data
BATCH_SIZE = 256
BUFFER_SIZE = 150
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi_ConvLSTM, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi_ConvLSTM, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#%%

output_shape = dataY_test[0].shape

# Define the ConvLSTM model

ConvLSTM_model = Sequential()
ConvLSTM_model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='tanh', 
input_shape=(2, 1, x_train_multi_ConvLSTM.shape[3], x_train_multi_ConvLSTM.shape[4])))
ConvLSTM_model.add(Dropout(0.2))
ConvLSTM_model.add(Flatten())
ConvLSTM_model.add(Dense(30, activation='relu'))
ConvLSTM_model.add(Dropout(0.2))
ConvLSTM_model.add(Dense(da_arr.shape))
ConvLSTM_model.compile(optimizer='adam', loss='mse')

#%%
# the best weights are stored in model path
model_path = r'/home/jefire/project/water/models/jiderebode_models/ConvLSTM_Multi_Jiderebode_Q_1step_30days_gpm_2001_2020_tanh.h5'


#%%

#%%
# Configure the model and start training with early stopping and checkpointing

EVALUATION_INTERVAL = 150
EPOCHS = 100
history = ConvLSTM_model.fit(train_data_multi, epochs=EPOCHS,
steps_per_epoch=EVALUATION_INTERVAL,validation_data=val_data_multi,
validation_steps=50,verbose =1,callbacks =[tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'),
    tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', 
    save_best_only=True, mode='min', verbose=0)])

#%%
# Load the best weights into the model
Trained_model = tf.keras.models.load_model(model_path)

#%%
# Plot the loss and val_loss against the epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],color='red')
plt.title('Model loss',size=14,fontweight='bold')
plt.ylabel('Loss',size=12,fontweight='semibold')
plt.xlabel('Epoch',size=12,fontweight='semibold')
plt.legend(['train loss', 'validation loss'])
plt.figure(figsize=(16,9))
plt.show()

#%%
# Check the model summary
Trained_model.summary()

#%%
# assign the number of days to validate model
val_period = len(df_test) - hist_window - horizon
val_period

#%%
# hold back 5 years data from test DataFrame to be used for testing the model
# validate DataFrame
validate = df_test.iloc[:,:].tail(val_period)
validate

#%%
# apply transform to test dataset
validateX = pipe.transform(df_test.iloc[:,:-1])

print(validateX[0])
print(validateX.shape)

#%%
# Let’s standardize data
validateY = scaler.transform(df_test[['Q']])
validateY[0]

# %%

# Prepare test dataset
x_test_multi, y_test_multi = custom_ts_multi_data_prep(
    validateX, validateY, 0, None, hist_window, horizon)
print ('Single window of past history')
print(x_test_multi[0])
print ('\n Target horizon')
print (y_test_multi[0])
print(x_test_multi.shape)

#%%
# reshape for ConvLSTM2D
x_test_multi_ConvLSTM = x_test_multi.reshape((x_test_multi.shape[0], 2, 1,
x_test_multi.shape[1]//2, x_test_multi.shape[2]))
print(x_test_multi_ConvLSTM.shape)

#%%
# predictions for ConvLSTM2D

predictions = []
for i in x_test_multi_ConvLSTM:
    # predict the daily streamflow
    i_reshape = i.reshape(1, i.shape[0], i.shape[1], i.shape[2], i.shape[3])
    Predicted_results = Trained_model.predict(i_reshape)
    # store the predictions
    predictions.append(Predicted_results)
print(predictions[0])


#%%
# convert prediction list to numpy array
predictions = np.array(predictions).reshape(-1, 1)
print(predictions.shape)

#%%

# Rescale the predicted values back to the original scale.

predictions_Inv_trans = scaler.inverse_transform(predictions)
print(predictions_Inv_trans[0])
print(predictions_Inv_trans.shape)

#%%
# Define the time-series evaluation function

def timeseries_evaluation_metrics_func(y_true, y_pred):
    kgeprime, r, gamma, beta = he.kgeprime(y_pred, y_true)
    kge, r, alpha, beta = he.kge(y_pred, y_true)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    print(f'NSE is : {he.nse(np.array(y_pred), np.array(y_true))}')
    print(f"KGE' is : {kgeprime}")
    print(f'r is : {r}, gamma is : {gamma}')
    print(f'KGE is : {kge}')
    print(f'alpha is : {alpha}, beta is : {beta}')
    print(f'PBIAS is : {he.pbias(np.array(y_pred), np.array(y_true))}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MARE is : {he.mare(np.array(y_pred), np.array(y_true))}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}') 
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
timeseries_evaluation_metrics_func(validate['Q'].values.reshape(-1, 1),predictions_Inv_trans)


#%%
# Plot the actual versus predicted values.
plt.plot( list(validate['Q']))
plt.plot( list(predictions_Inv_trans),color='red')
plt.title("Observed vs Predicted",size=14,fontweight='bold')
plt.ylabel("Streamflow (m$\mathregular{^{3}}$/s)",size=12,fontweight='semibold')
plt.legend(('Observed','predicted'))
plt.figure(figsize=(16,9))
plt.show()

#%%
# save GPM IMERGE train loss and val loss to CSV
df_history = pd.DataFrame.from_records([history.history['loss'],history.history['val_loss']], index=['train_loss', 'val_loss']).transpose()
df_history.to_csv('/home/jefire/project/water/models/jiderebode_models/ConvLSTM_Multi_Jiderebode_Q_1step_30days_gpm_tanh_loss.csv', index=False)







#%%
# CNN Model

# %%
# Prepare the training and validation time-series data
BATCH_SIZE = 256
BUFFER_SIZE = 150
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#%%
# Define the CNN model

CNN_model = Sequential()
CNN_model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
input_shape=(x_train_multi.shape[1], x_train_multi.shape[2])))
CNN_model.add(MaxPool1D(pool_size=2))
CNN_model.add(Dropout(0.2))
CNN_model.add(Flatten())
CNN_model.add(Dense(30, activation='relu'))
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(horizon))
CNN_model.compile(optimizer='adam', loss='mse')

#%%
# the best weights are stored in model path
model_path = r'/home/jefire/project/water/models/jiderebode_models/CNN_Multi_Jiderebode_Q_1step_30days_gpm.h5'


#%%
# Configure the model and start training with early stopping and checkpointing

EVALUATION_INTERVAL = 150
EPOCHS = 100
history = CNN_model.fit(train_data_multi, epochs=EPOCHS,
steps_per_epoch=EVALUATION_INTERVAL,validation_data=val_data_multi,
validation_steps=50,verbose =1,callbacks =[tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'),
    tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', 
    save_best_only=True, mode='min', verbose=0)])

#%%
# Load the best weights into the model
Trained_model = tf.keras.models.load_model(model_path)

#%%
# Plot the loss and val_loss against the epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],color='red')
plt.title('Model loss',size=14,fontweight='bold')
plt.ylabel('Loss',size=12,fontweight='semibold')
plt.xlabel('Epoch',size=12,fontweight='semibold')
plt.legend(['train loss', 'validation loss'])
plt.figure(figsize=(16,9))
plt.show()

#%%
# Check the model summary
Trained_model.summary()

# Prepare test dataset
x_test_multi, y_test_multi = custom_ts_multi_data_prep(
    validateX, validateY, 0, None, hist_window, horizon)
print ('Single window of past history')
print(x_test_multi[0])
print ('\n Target horizon')
print (y_test_multi[0])
print(len(y_test_multi))

#%%
# predictions for CNN

predictions = []
for i in x_test_multi:
    # predict the daily streamflow
    i_reshape = i.reshape(1, i.shape[0], i.shape[1])
    Predicted_results = Trained_model.predict(i_reshape)
    # store the predictions
    predictions.append(Predicted_results)
print(predictions[0])

#%%
# convert prediction list to numpy array
predictions = np.array(predictions).reshape(-1, 1)
print(predictions.shape)

#%%

# Rescale the predicted values back to the original scale.

predictions_Inv_trans = scaler.inverse_transform(predictions)
print(predictions_Inv_trans[0])
print(predictions_Inv_trans.shape)
print(len(predictions_Inv_trans))

#%%
# Define the time-series evaluation function

def timeseries_evaluation_metrics_func(y_true, y_pred):
    kgeprime, r, gamma, beta = he.kgeprime(y_pred, y_true)
    kge, r, alpha, beta = he.kge(y_pred, y_true)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    print(f'NSE is : {he.nse(np.array(y_pred), np.array(y_true))}')
    print(f"KGE' is : {kgeprime}")
    print(f'r is : {r}, gamma is : {gamma}')
    print(f'KGE is : {kge}')
    print(f'alpha is : {alpha}, beta is : {beta}')
    print(f'PBIAS is : {he.pbias(np.array(y_pred), np.array(y_true))}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MARE is : {he.mare(np.array(y_pred), np.array(y_true))}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}') 
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
timeseries_evaluation_metrics_func(validate['Q'].values.reshape(-1, 1),predictions_Inv_trans)


#%%
# Plot the actual versus predicted values.
plt.plot( list(validate['Q']))
plt.plot( list(predictions_Inv_trans),color='red')
plt.title("Actual vs Predicted")
plt.ylabel("Water Level")
plt.legend(('Actual','predicted'))
plt.show()

#%%
# save GPM IMERGE train loss and val loss to CSV
df_history = pd.DataFrame.from_records([history.history['loss'],history.history['val_loss']], index=['train_loss', 'val_loss']).transpose()
df_history.to_csv('/home/jefire/project/water/models/jiderebode_models/CNN_Multi_Jiderebode_Q_1step_30days_gpm_history.csv', index=False)























#%%









ds_gb_mnt = ds.groupby('time.month')
ds_gb_mnt

#%%
ds_zg500_gb_season = ds_zg500.groupby('time.season')
ds_zg500_gb_season

#%%
# get the groupby month mean of years 2001 to 2020
ds_zg500_gbmonth_mean = ds_zg500.groupby('time.month').mean('time')
ds_zg500_gbmonth_mean


#%%
# get the groupby seasonal mean of years 2010 to 2014
ds_zg500_gbseason_mean = ds_zg500.groupby('time.season').mean('time')
ds_zg500_gbseason_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_zg500_gbmonth_std = ds_zg500.groupby('time.month').std('time')
ds_zg500_gbmonth_std

#%%
# get the groupby seasonal standard deviation of years 2010 to 2014
ds_zg500_gbseason_std = ds_zg500.groupby('time.season').std('time')
ds_zg500_gbseason_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_zg500_month_anom = ds_zg500_gb_mnt - ds_zg500_gbmonth_mean
ds_zg500_month_anom

#%%
ds_zg500_anom_gb_mnt = ds_zg500_month_anom.groupby('time.month')
ds_zg500_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_month_standardized = ds_zg500_anom_gb_mnt / ds_zg500_gbmonth_std
ds_month_standardized = ds_month_standardized.drop('month', dim=None)
ds_month_standardized










ds_zg500_gb_mnt = ds_zg500.groupby('time.month')
ds_zg500_gb_mnt

#%%
ds_zg500_gb_season = ds_zg500.groupby('time.season')
ds_zg500_gb_season

#%%
# get the groupby month mean of years 2001 to 2020
ds_zg500_gbmonth_mean = ds_zg500.groupby('time.month').mean('time')
ds_zg500_gbmonth_mean


#%%
# get the groupby seasonal mean of years 2010 to 2014
ds_zg500_gbseason_mean = ds_zg500.groupby('time.season').mean('time')
ds_zg500_gbseason_mean

#%%
# get the groupby monthly standard deviation of years 2010 to 2014
ds_zg500_gbmonth_std = ds_zg500.groupby('time.month').std('time')
ds_zg500_gbmonth_std

#%%
# get the groupby seasonal standard deviation of years 2010 to 2014
ds_zg500_gbseason_std = ds_zg500.groupby('time.season').std('time')
ds_zg500_gbseason_std

#%%
# transform data to grouped monthly anomalous with calibration of 2001 to 2014
ds_zg500_month_anom = ds_zg500_gb_mnt - ds_zg500_gbmonth_mean
ds_zg500_month_anom

#%%
ds_zg500_anom_gb_mnt = ds_zg500_month_anom.groupby('time.month')
ds_zg500_anom_gb_mnt

# %%
# standardize monthly anomalous dataarray
ds_month_standardized = ds_zg500_anom_gb_mnt / ds_zg500_gbmonth_std
ds_month_standardized = ds_month_standardized.drop('month', dim=None)
ds_month_standardized












#%%
# Cordinates

# lokoja = da_copy.sel(lat='7.7523314', lon='6.7532070', method='nearest').resample(time="Y").mean()

# jide = da_copy.sel(lat='11.3858578', lon='4.1210672', method='nearest').resample(time="Y").mean()

# niamey = da_copy.sel(lat='13.5232201', lon='2.0795191', method='nearest').resample(time="Y").mean()

# ds
# ds_tas
# ds_tasmax
# ds_tasmin 
# ds_huss
# ds_zg500
# ds_zg1000
# ds_psl
# ds_uas
# ds_vas
# ds_era5_pr

#%%
ds_uas.uas.sel(latitude='7.7523314', longitude='6.7532070', method='nearest')

#%%
# Cordinates

ds_lokoja = ds_tas.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_tasmax_lokoja = ds_tasmax.tasmax.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_tasmin_lokoja = ds_tasmin.tasmin.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_huss_lokoja = ds_huss.q.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_zg500_lokoja = ds_zg500.z500.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_zg1000_lokoja = ds_zg1000.z1000.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_psl_lokoja = ds_psl.psl.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_uas_lokoja = ds_uas.uas.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_vas_lokoja = ds_vas.vas.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_era5_pr_lokoja = ds_era5_pr.era5_pr.sel(lat='7.7523314', lon='6.7532070', method='nearest')
da_pr_lokoja = ds.pr.sel(lat='7.7523314', lon='6.7532070', method='nearest')

#%%
ds_lok = ds_lokoja.copy()
ds_lok['tasmax'] = da_tasmax_lokoja
ds_lok['tasmin'] = da_tasmin_lokoja
ds_lok['q'] = da_huss_lokoja
ds_lok['z500'] = da_zg500_lokoja
ds_lok['z1000'] = da_zg1000_lokoja
ds_lok['psl'] = da_psl_lokoja
ds_lok['uas'] = da_uas_lokoja
ds_lok['vas'] = da_vas_lokoja
# ds_lok['era5_pr'] = da_era5_pr_lokoja
ds_lok['pr'] = da_pr_lokoja
ds_lok


#%%

ds_jide = ds_tas.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_tasmax_jide = ds_tasmax.tasmax.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_tasmin_jide = ds_tasmin.tasmin.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_huss_jide = ds_huss.q.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_zg500_jide = ds_zg500.z500.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_zg1000_jide = ds_zg1000.z1000.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_psl_jide = ds_psl.psl.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_uas_jide = ds_uas.uas.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_vas_jide = ds_vas.vas.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_era5_pr_jide = ds_era5_pr.era5_pr.sel(lat='11.3858578', lon='4.1210672', method='nearest')
da_pr_jide = ds.pr.sel(lat='11.3858578', lon='4.1210672', method='nearest')

#%%
ds_jiderebode = ds_jide.copy()
ds_jiderebode['tasmax'] = da_tasmax_jide
ds_jiderebode['tasmin'] = da_tasmin_jide
ds_jiderebode['q'] = da_huss_jide
ds_jiderebode['z500'] = da_zg500_jide
ds_jiderebode['z1000'] = da_zg1000_jide
ds_jiderebode['psl'] = da_psl_jide
ds_jiderebode['uas'] = da_uas_jide
ds_jiderebode['vas'] = da_vas_jide
# ds_jiderebode['era5_pr'] = da_era5_pr_jide
ds_jiderebode['pr'] = da_pr_jide
ds_jiderebode


#%%

ds_niamey = ds_tas.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_tasmax_niamey = ds_tasmax.tasmax.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_tasmin_niamey = ds_tasmin.tasmin.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_huss_niamey = ds_huss.q.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_zg500_niamey = ds_zg500.z500.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_zg1000_niamey = ds_zg1000.z1000.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_psl_niamey = ds_psl.psl.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_uas_niamey = ds_uas.uas.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_vas_niamey = ds_vas.vas.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_era5_pr_niamey = ds_era5_pr.era5_pr.sel(lat='13.5232201', lon='2.0795191', method='nearest')
da_pr_niamey = ds.pr.sel(lat='13.5232201', lon='2.0795191', method='nearest')

#%%
ds_niam = ds_niamey.copy()
ds_niam['tasmax'] = da_tasmax_niamey
ds_niam['tasmin'] = da_tasmin_niamey
ds_niam['q'] = da_huss_niamey
ds_niam['z500'] = da_zg500_niamey
ds_niam['z1000'] = da_zg1000_niamey
ds_niam['psl'] = da_psl_niamey
ds_niam['uas'] = da_uas_niamey
ds_niam['vas'] = da_vas_niamey
# ds_niam['era5_pr'] = da_era5_pr_niamey
ds_niam['pr'] = da_pr_niamey
ds_niam


#%%
ds_niamey_month_gb = ds_niam.groupby('time.month')
ds_niamey_month_gb

#%%
ds_niam_month = ds_niamey_month_gb[10]
ds_niam_month 

#%%
ds_lok_month_gb = ds_lok.groupby('time.month')
ds_lok_month_gb

#%%
ds_lok_month = ds_lok_month_gb[9]
ds_lok_month 

#%%
ds_lok_season_gb = ds_lok.groupby('time.season')
ds_lok_season_gb

#%%
ds_lok_mam = ds_lok_season_gb['MAM']
ds_lok_mam

#%%
da_tasmax_lokoja

#%%
ds_lok_sep = ds_lok.pr.sel(month=9)
ds_lok_sep

#%%
ds_lok_month = ds_lok_gb[10]
ds_lok_month

# %%
months = [6,7,8,9] # for example
ds_lok_month = ds.pr.sel(time = np.in1d( ds.pr['time.month'], months))

#%%

df_lok = ds_lok_month.to_dataframe()
df_lok

#%%

df_lok.reset_index(inplace=True)
df_lok.rename(columns = {'time':'date'}, inplace = True)
df_lok.set_index('date', inplace=True)
df_lok


#%%
# drop columns from DataFrame that are not important
#df_nasadem_gpm_025.drop(['index','time','tas'],axis=1, inplace=True)
df_lok.drop(['lon','lat'],axis=1, inplace=True)
df_lok.head()

# %%
months = [6,7,8,9] # for example
season = ds.pr.sel(time = np.in1d( ds.pr['time.month'], months))

#%%
ds_jiderebode_gb = ds_jiderebode.groupby('time.month')

#%%
ds_jiderebode_month = ds_jiderebode_gb[10]
ds_jiderebode_month

#%%
df_jiderebode = ds_jiderebode_month.to_dataframe()
df_jiderebode

#%%
df_jiderebode.reset_index(inplace=True)
df_jiderebode.rename(columns = {'time':'date'}, inplace = True)
df_jiderebode.set_index('date', inplace=True)
df_jiderebode.drop(['lon','lat'],axis=1, inplace=True)
df_jiderebode

#%%
df_niam = ds_niam_month.to_dataframe()
df_niam


#%%
df_niam.reset_index(inplace=True)
df_niam.rename(columns = {'time':'date'}, inplace = True)
df_niam.set_index('date', inplace=True)
df_niam.drop(['lon','lat'],axis=1, inplace=True)
df_niam

#%%

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

#%%
size = int(len(df_lok)*0.8)

#%%
size = int(len(df_niam)*0.8)

#%%
df_train = df_lok.iloc[:size]
df_test = df_lok.iloc[size:]

#%%
df_train = df_jiderebode.iloc[:size]
df_test = df_jiderebode.iloc[size:]

#%%
df_train = df_niam.iloc[:size]
df_test = df_niam.iloc[size:]

#%%
X_train = df_train.drop(['pr'], axis='columns').values
y_train = df_train['pr'].values
X_test = df_test.drop(['pr'], axis='columns').values
y_test = df_test['pr'].values

#%%
X[0:4]

# %%
#Split data into features and target
df_lok_copy = df_lok.copy()
X = df_lok_copy.drop(['pr'], axis='columns')
y = df_lok_copy['pr']

# %%
X.head()

# %%
y.head()

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 70)

#%%
X_train

# %%
regression = Lasso()

# %%
regression.fit(X_train,y_train)

# %%
print(regression.coef_)

#%%

format(3.5592657e-02, '.53f')

#%%
ds_tasmax.tasmax.data.shape

#%%

ds_niam.data.shape

#%%

#%%
# compute correlation for all Rainfall and Q columns 
df_lok_corr = df_lok.corr()
df_lok_corr

#%%
# drop Q row index to max value correlation for precipitation  
df_lok_corr = df_lok_corr.drop(df_lok_corr.index[-1])
df_lok_corr

#%%
# plot a line plot of the various corellation
df_lok_corr.plot(y='pr', style='.-')

#%%
# get index of maximum correlation coefficient of Q (Discharge) and lagged precipitation
df_lok_corr[['pr']].idxmax()


#%%
df_lok_corr[['pr']].max

#%%


#%%
# compute correlation for all Rainfall and Q columns 
df_jiderebode_corr = df_jiderebode.corr()
df_jiderebode_corr

#%%
# drop Q row index to max value correlation for precipitation  
df_jiderebode_corr = df_jiderebode_corr.drop(df_jiderebode_corr.index[-1])
df_jiderebode_corr

#%%
# plot a line plot of the various corellation
df_jiderebode_corr.plot(y='pr', style='.-')

#%%
# get index of maximum correlation coefficient of Q (Discharge) and lagged precipitation
df_jiderebode_corr[['pr']].idxmax()


#%%
df_jiderebode_corr[['pr']].max



#%%
# compute correlation for all Rainfall and Q columns 
df_niam_corr = df_niam.corr()
df_niam_corr

#%%
# drop Q row index to max value correlation for precipitation  
df_niam_corr = df_niam_corr.drop(df_niam_corr.index[-1])
df_niam_corr

#%%
# plot a line plot of the various corellation
df_niam_corr.plot(y='pr', style='.-')

#%%
# get index of maximum correlation coefficient of Q (Discharge) and lagged precipitation
df_niam_corr[['pr']].idxmax()


#%%
df_niam_corr[['pr']].max
























#%%
# rename the longitude to lon
ds = ds.rename({'longitude':'lon'})
ds = ds.rename({'latitude':'lat'})
ds

#%%
# reindex latitude in increasing dimension
ds = ds.reindex(lat=list(reversed(ds.lat)))
ds

#%%
# transpose dataset to (lat,lon,time)
ds = ds.transpose("lat", "lon", "time")
ds

#%%
# re-assign Dimension name attribute
ds['tmax_day'] = ds.tmax_day.assign_attrs(DimensionNames='lat,lon,time')
ds

#%%
da = ds.tas_day
da

#%%
# ERA5 Seasonal Resample
da_copy = da.copy()
da_copy

#%%
# Slice December 2000 to November 2020
da_copy = da_copy.sel(time=slice('1979-01-01', '2020-12-31'))
da_copy

#%%

da_copy.rio.set_spatial_dims('lon', 'lat').rio.write_crs("epsg:4326", inplace=True)
da_copy


#%%
da_copy_clip = da_copy.rio.clip(basin_polygon.geometry.apply(mapping), basin_polygon.crs, drop=False)
da_copy_clip

#%%
plt.figure(figsize=(16, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
da_season_clip.sel(time='2018-08-31').plot.pcolormesh(ax=ax, vmin=-20, vmax=450);
ax.coastlines()

#%%
da_copy_clip.sel(time='2018-08-31').plot(cmap='jet')

#%%
da_copy_clip.mean(['lon','lat'], skipna=True).plot()
# plt.axhline(0, color='k', linestyle='dashed')

#%%
# get subbasin mean
da_copy_clip_basin_mean = da_copy_clip.mean(['lon','lat'], skipna=True)
da_copy_clip_basin_mean

#%%
# calculate Monthly Time series
da_copy_clip_basin_mean_month = da_copy_clip_basin_mean.resample(time="M").mean()
da_copy_clip_basin_mean_month

#%%
da_copy_clip_basin_mean_month.plot()

#%%
# calculate Annual Time series
da_copy_clip_basin_mean_year = da_copy_clip_basin_mean.resample(time="Y").mean()
da_copy_clip_basin_mean_year

#%%
da_copy_clip_basin_mean_year.plot()


#%%
# calculate Seasonal Time series
da_copy_clip_basin_mean_season = da_copy_clip_basin_mean.resample(time="Q-NOV").mean()
da_copy_clip_basin_mean_season

#%%
da_copy_clip_basin_mean_season = da_copy_clip_basin_mean_season.sel(time=slice('1979-02-28', '2020-11-30'))
da_copy_clip_basin_mean_season

#%%
da_copy_clip_basin_mean_season.plot()

#%%
da_sub = ds_tas.tas.sel(time=slice('2001-01-01', '2001-01-10'))
da_sub

#%%
da_sub.data.T.shape


#%%
da_sub.data.T[10]











# %%
