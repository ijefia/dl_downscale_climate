#%%
# Import Packages
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
# Minimum Temperature


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
# Selected months
months = [1]

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

da = ds.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_tas = ds_tas.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_tasmax = ds_tasmax.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_tasmin = ds_tasmin.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_huss = ds_huss.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_zg500 = ds_zg500.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_zg1000 = ds_zg1000.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_psl = ds_psl.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_uas = ds_uas.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_vas = ds_vas.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da_era5_pr = ds_era5_pr.isel(lat=slice(None, 3), lon=slice(None, 4)).load()
da

#%%

da_arr = da.pr.data
da_tas_arr = da_tas.tas.data
da_tasmax_arr = da_tasmax.tasmax.data
da_tasmin_arr = da_tasmin.tasmin.data
da_huss_arr = da_huss.q.data
da_zg500_arr = da_zg500.z500.data
da_zg1000_arr = da_zg1000.z1000.data
da_psl_arr = da_psl.psl.data
da_uas_arr = da_uas.uas.data
da_vas_arr = da_vas.vas.data
da_era5_pr_arr = da_era5_pr.era5_pr.data
da_arr

#%%

da_arr.shape

#%%
da_arr[0].shape

#%%
a = np.stack((da_tas_arr[0], da_zg500_arr[0], da_psl_arr[0]), axis=-1)
print(a.shape)
a

#%%
print(a.shape)

#%%
da_tas_arr[0]

#%%
b = np.dstack((da_tas_arr[:3], da_zg500_arr[:3], da_psl_arr[:3]))
print(b.shape)
b

#%%

c = np.append(da_tas_arr[:3], da_zg500_arr[:3], da_psl_arr[:3])
print(c.shape)
c

#%%
da_tas_arr.flatten().shape

#%%

(da_tas_arr.reshape(da_tas_arr.shape[0], -1).T)

#%%

(da_tas_arr.reshape(da_tas_arr.shape[0], -1).T).shape


#%%
da_tas_arr.shape

vh.flatten()

#%%

(np.array(test_one['vh']).reshape((512, 512)))

#%%
d1 = da_tas_arr.flatten()
d2 = da_zg500_arr.flatten()
d3 = da_psl_arr.flatten()
d3

#%%

e = np.hstack((d1,d2,d3))
e.shape

#%%

np.transpose(e, [1,0]).shape

#%%

e.T.shape

#%%
f = np.stack((d1, d2, d3), axis=-1)
print(f.shape)
f

#%%

g = f.reshape((31,3,4,3))
g

#%%

g.shape

#%%

g[0].shape

#%%

h = np.stack((da_tas_arr, da_zg500_arr, da_psl_arr), axis=-1)

h[0]

#%%
# Step 1: convert dataarrays to numpy arrays

da_arr = da.pr.load().data
da_tas_arr = da_tas.tas.data
da_tasmax_arr = da_tasmax.tasmax.data
da_tasmin_arr = da_tasmin.tasmin.data
da_huss_arr = da_huss.q.data
da_zg500_arr = da_zg500.z500.data
da_zg1000_arr = da_zg1000.z1000.data
da_psl_arr = da_psl.psl.data
da_uas_arr = da_uas.uas.data
da_vas_arr = da_vas.vas.data

print(da_arr[0])
da_arr.shape

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

print(da_arr_flat[0])
da_arr_flat.shape


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

clim_stack = clim_stack.reshape((31, 113, 145, 10))
clim_stack.shape


#%%
# Step 3: PCA Dimensionality Reduction
















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
