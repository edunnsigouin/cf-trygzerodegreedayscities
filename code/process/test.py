import numpy as np
import xarray as xr

"""
filename = '/nird/datapeak/NS9873K/etdu/raw/eobs/daily/31_0e/tn/tn_0.1x0.1_2024.nc'

ds_tn = xr.open_dataset(filename).sel(latitude=60.3499,longitude=5.3499,method='nearest')

print(ds_tn.latitude.values, ds_tn.longitude.values)


print(ds_tn['tn'].values)
"""


filename = '/nird/datapeak/NS9873K/etdu/processed/cf-trygzerodegreedayscities/eobs/31_0e/scandinavian_city_zero_degree_crossing_with_stats_eobs_all_gridpoint_mean_1951-1951.nc'

ds = xr.open_dataset(filename).sel(city='Bergen')

#print(ds['city_lat'].values,ds['city_orig_lat'].values)

print(ds['n_valid_days'][:,-1,0].values)


