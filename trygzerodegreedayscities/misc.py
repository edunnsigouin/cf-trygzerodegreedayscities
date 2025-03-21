"""
collection of useful miscellaneous functions                                                                                                                  
"""
import numpy  as np
import xarray as xr


def get_exps(data_flag,exp_key):
    """
    translates experiment key number from config.py to list of
    experiment file name strings
    """
    if data_flag == 'cordex_kss':
        
        from beeware.config    import data_cordex_kss_name as exps
        exps = [exps[i] for i in exp_key.tolist()]

    return exps



def get_dim(data_flag,exp_flag):
    """      
    Outputs hard-coded data dimensions (lat-lon-lev-time)      
    for a given dataset                                                                                                                   
    """
    if data_flag == 'senorge':
        from beeware import dim_senorge as dim
    elif data_flag == 'cordex_kss':
        if exp_flag == 'rcp85':
            from beeware import dim_cordex_kss_rcp85 as dim
        elif exp_flag == 'hist':
            from beeware import dim_cordex_kss_hist as dim
            
    return dim



def get_dir(data_flag):
    """
    get data directories for different data sources
    """
    
    if data_flag == 'senorge':
        from beeware.config import senorge_processed as processed
        from beeware.config import senorge_raw       as raw
        from beeware.config import senorge_fig       as fig
        from beeware.config import proj

        dirs = {"processed":processed,
                "raw":raw,
                "fig":fig,
                "proj":proj,
        }

    elif data_flag == 'cordex_kss':
        from beeware.config import cordex_kss_processed as processed
        from beeware.config import cordex_kss_raw       as raw
        from beeware.config import cordex_kss_fig       as fig
        from beeware.config import proj

        dirs = {"processed":processed,
                "raw":raw,
                "fig":fig,
                "proj":proj,
        }

    return dirs


def get_filenames_senorge(var,dir_in,years):
    """ 
    generates a list of file names
    for senorge data by year
    """
    files = [dir_in + var + '_' + str(year) + '.nc' for year in years]

    return files




def tic():
    # Homemade version of matlab tic function
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    # Homemade version of matlab tic function
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def get_season(ds,season):
    """
    Extracts times belonging to a given season
    input = xarray dataset or dataarray
    """
    months = ds['time.month']

    if season == 'NDJFM':
        index = (months >= 11) | (months <= 3)
    elif season == 'MJJAS':
        index = (months >= 5) & (months <= 9)
    elif season == 'ANNUAL':
        index = (months >= 1) & (months <= 12)
    elif season == 'DJF':
        index = (months >= 12) | (months <= 2)
    elif season == 'MAM':
        index = (months >= 3) & (months <= 5)
    elif season == 'JJA':
        index = (months >= 6) & (months <= 8)
    elif season == 'SON':
        index = (months >= 9) & (months <= 11)

    return ds.sel(time=index)


def rm_lpyr_days(ds):
    """
    removes leap-year days from daily xarray dataset or dataarray 
    """
    return ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))


def groupby_month_day_str(ds):
    """
    Workaround for groupby 'dayofyear' which groups by days starting from jan 1st (1-365 or 366) and 
    so leap year dayofyear can be different for leap-year/non-leap years.
    The solution is to groupby 'month_day_str' so that leap year day is always '02-29'.   
    Discussion and solution found here: https://github.com/pydata/xarray/issues/1844
    """
    month_day_str  = xr.DataArray(ds.indexes['time'].strftime('%m-%d'),dims=["time"],name='month_day_str')
    return ds.groupby(month_day_str)


def xy_mean(ds):
    """ 
    calculates xy mean over dims lat and lon   
    with cosine weighting in lat
    """
    weights = np.cos(np.deg2rad(ds.lat))
    ds      = ds.weighted(weights).mean(dim=('lat','lon'))

    return ds


def y_mean(ds):
    """       
    calculates y mean over dim lat
    with cosine weighting in lat
    """
    weights = np.cos(np.deg2rad(ds.lat))
    ds      = ds.weighted(weights).mean(dim='lat')

    return ds



def smooth_time_dim(da,window):
    """
    running-mean smoother centered with window  
    on time dimension                                                                                                              
    """
    pad = int((window-1)/2)
    da = da.pad(time=pad, mode='wrap')
    da = da.rolling(time=window, center=True).mean()
    da = da.dropna('time')

    return da
