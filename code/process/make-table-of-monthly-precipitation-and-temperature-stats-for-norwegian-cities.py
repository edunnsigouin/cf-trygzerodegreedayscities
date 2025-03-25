"""
Calculates a set of monthly statistics for weather variables
averaged over a set of key Norwegian cities. Outputs everything
to a CSV table for a given month.

Statistics include:
  - Percent of days where daily minimum temperature < 0
  - Minimum daily-mean temperature in month
  - Maximum daily-mean temperature in month
  - Median daily-mean temperature in month
  - Mean daily-mean temperature in month
  - Percent of days with > 0 daily-accumulated precipitation
  - Percent of days where daily-accumulated precipitation is extreme 
    (i.e., > 90th quantile based on a 20-year climatology)
"""

import numpy as np
import xarray as xr
import pandas as pd
import dask
from trygzerodegreedayscities import config, misc

# input ----------------------------------------------
month                    = 'dec'
years                    = np.arange(2022, 2025, 1)
cities                   = config.cities
path_in                  = config.dirs['senorge_raw']
path_out                 = config.dirs['data']
climatology_window_years = 20
write2file               = True
# -----------------------------------------------------


def month_str2num(month_str):
    """
    Convert a 3-letter month abbreviation into its numeric equivalent.
    E.g. 'jan' -> 1, 'feb' -> 2, etc.
    """
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    return month_map[month_str]


def calc_monthly_90th_quantile_precipitation_climatology_per_city(
    clim_year, month, city, path_in, climatology_window_years=20
):
    """
    Opens all 'rr' NetCDF files from (clim_year - climatology_window_years) 
    up to (clim_year - 1), uses a custom 'preprocess' to:

      1. Subselect a bounding box based on city and drop everything else.
      2. Average in space (collapsing to a 1D time series).
      3. Keep only data from the requested month.

    Returns the 90th quantile of precipitation over the aggregated time dimension.
    """
    # Collect the years of interest and corresponding file names
    years = np.arange(clim_year - climatology_window_years, clim_year, 1)
    filenames = [f"{path_in}rr/rr_{yr}.nc" for yr in years]
    
    # Prepare bounding box and month for subsetting
    bbox = misc.get_city_bboxes(city)
    month_num = month_str2num(month)

    def preprocess(ds):
        """
        Preprocessing applied to each file individually before combining:
          - Mask using bounding box
          - Average in space
          - Keep only the requested month
        """
        # --- bounding box ---
        mask = (
            (ds["lat"] >= bbox["lat_min"])
            & (ds["lat"] <= bbox["lat_max"])
            & (ds["lon"] >= bbox["lon_min"])
            & (ds["lon"] <= bbox["lon_max"])
        ).compute()  # Compute the mask to avoid unknown-size Dask array errors
        ds = ds.where(mask, drop=True)

        # --- average in space ---
        ds = ds.mean(dim=["Y", "X"])
        
        # --- time selection ---
        month_mask = (ds.time.dt.month == month_num).compute()
        ds = ds.sel(time=month_mask)
        
        return ds
    
    # Open multiple files with custom preprocess logic
    rr = xr.open_mfdataset(filenames, preprocess=preprocess)['rr'].compute()

    # Return the 90th quantile over time
    return rr.quantile(0.9, dim="time")


def calc_monthly_weather_statistics_per_city(
    city, month, years, path_in, climatology_window_years
):
    """
    Calculates monthly weather statistics for a single city across multiple years.
    Returns a Pandas DataFrame with columns: city, year, and each statistic.

    Statistics:
      - Percent of days where daily min temperature < 0
      - Min/Max/Median/Mean of daily-mean temperature
      - Percent of days with > 0 precip
      - Percent of days with 'extreme precip' (> 90th quantile from a 20-year climatology)
    """
    # Retrieve bounding box for the city
    bbox = misc.get_city_bboxes(city)
    month_num = month_str2num(month)

    # Accumulate a list of per-year results
    records = []

    for year in years:
        misc.tic()
        print(f"Calculating stats for {city}, {year}...")

        # --- read data for tg, tn, rr ---
        filename_tg = f"{path_in}/tg/tg_{year}.nc"
        filename_tn = f"{path_in}/tn/tn_{year}.nc"
        filename_rr = f"{path_in}/rr/rr_{year}.nc"
        tg = xr.open_dataset(filename_tg)['tg']
        tn = xr.open_dataset(filename_tn)['tn']
        rr = xr.open_dataset(filename_rr)['rr']
        
        # --- subselect month ---
        tg = tg.sel(time=tg.time.dt.month == month_num)
        tn = tn.sel(time=tn.time.dt.month == month_num)
        rr = rr.sel(time=rr.time.dt.month == month_num)
        
        # --- subselect bounding box ---
        tg = tg.where(
            (tg["lat"] >= bbox["lat_min"]) & (tg["lat"] <= bbox["lat_max"]) &
            (tg["lon"] >= bbox["lon_min"]) & (tg["lon"] <= bbox["lon_max"]),
            drop=True
        )
        tn = tn.where(
            (tn["lat"] >= bbox["lat_min"]) & (tn["lat"] <= bbox["lat_max"]) &
            (tn["lon"] >= bbox["lon_min"]) & (tn["lon"] <= bbox["lon_max"]),
            drop=True
        )
        rr = rr.where(
            (rr["lat"] >= bbox["lat_min"]) & (rr["lat"] <= bbox["lat_max"]) &
            (rr["lon"] >= bbox["lon_min"]) & (rr["lon"] <= bbox["lon_max"]),
            drop=True
        )

        # --- average in space ---
        tg = tg.mean(dim=["Y", "X"])
        tn = tn.mean(dim=["Y", "X"])
        rr = rr.mean(dim=["Y", "X"])
        
        # --- calculate basic statistics ---
        # Percent of days with daily min temperature < 0
        tn_zero_days = (tn < 0).sum().item() / tn.size * 100

        # min, max, median, mean daily-mean temperature
        tg_min = tg.min(dim='time').item()
        tg_max = tg.max(dim='time').item()
        tg_median = tg.median(dim='time').item()
        tg_mean = tg.mean(dim='time').item()

        # Percent of days with > 0 precipitation
        rr_precip_days = (rr > 0).sum().item() / rr.size * 100

        # Compute the 90th quantile from the climatology
        rr_90pct_month = calc_monthly_90th_quantile_precipitation_climatology_per_city(year, month, city, path_in, climatology_window_years)
        
        # Percent of days exceeding the 90th quantile
        rr_extreme_days = (rr > rr_90pct_month).sum().item() / rr.size * 100

        # --- add row to records ---
        row = {
            "city": city,
            "year": year,
            f"{month}_percent_days_precipitation": round(rr_precip_days, 1),
            f"{month}_percent_days_extreme_precipitation": round(rr_extreme_days, 1),
            f"{month}_percent_days_below_zero_degree": round(tn_zero_days, 1),
            f"{month}_minimum_daily_mean_temperature": round(tg_min, 1),
            f"{month}_maximum_daily_mean_temperature": round(tg_max, 1),
            f"{month}_median_daily_mean_temperature": round(tg_median, 1),
            f"{month}_mean_temperature": round(tg_mean, 1),
        }
        records.append(row)

        misc.toc()
    
    # Convert to DataFrame
    df_city = pd.DataFrame(records)
    
    return df_city


def calc_monthly_weather_statistics_for_cities(cities, month, years, path_in, climatology_window_years):
    """
    Loop over multiple cities, compute stats for each city (and each year),
    and concatenate everything into one DataFrame.
    """
    df_list = []
    for city in cities:
        df_city = calc_monthly_weather_statistics_per_city(city, month, years, path_in, climatology_window_years)
        df_list.append(df_city)

    # Concatenate all cities into a single table
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all


# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    
    # Compute stats for all specified cities
    df_stats = calc_monthly_weather_statistics_for_cities(cities, month, years, path_in, climatology_window_years)

    # Optionally write to file
    if write2file:
        month_num = month_str2num(month)
        out_name  = f"{path_out}/weather_stats_norwegian_cities_{years[-1]}-{str(month_num).zfill(2)}.csv"
        df_stats.to_csv(out_name, index=False)
        print(f"\nCSV written to {out_name}")



