"""
Extract and average minimum daily temperature over a single Norwegian city,
focusing on December–January–February (DJF) for a given year. The code:
1) Makes the choice of city optional (defaults to Oslo).
2) Only reads data for the subset of time needed (Dec of previous year + Jan/Feb of target year).
3) Uses bounding-box definitions from a separate function.
4) Plots a time series of daily min temp with counts of subzero days in Dec, Jan, and Feb.
   - The timeseries plot is now factored out into a separate function.
   - The bounding box map plot is also a separate function with optional PDF output.
   - Grid lines on the timeseries are removed.
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates

from trygzerodegreedayscities import config, misc

# input --------------------------------------------
city       = 'Alta'
year       = 2025
variable   = 'tg'
path_in    = config.dirs['senorge_raw']
path_out   = config.dirs['fig']
write2file = True
# --------------------------------------------------

def get_city_bboxes():
    """
    Returns a dictionary of city bounding boxes.
    Each entry has lat_min, lat_max, lon_min, lon_max.
    """
    return {
        "Oslo": {
            "lat_min": 59.70, "lat_max": 60.25,
            "lon_min": 10.40, "lon_max": 11.10
        },
        "Kristiansand": {
            "lat_min": 58.00, "lat_max": 58.30,
            "lon_min":  7.80, "lon_max":  8.20
        },
        "Stavanger": {
            "lat_min": 58.80, "lat_max": 59.10,
            "lon_min":  5.50, "lon_max":  6.00
        },
        "Bergen": {
            "lat_min": 60.20, "lat_max": 60.55,
            "lon_min":  5.10, "lon_max":  5.50
        },
        "Ålesund": {
            "lat_min": 62.30, "lat_max": 62.60,
            "lon_min":  5.90, "lon_max":  6.40
        },
        "Trondheim": {
            "lat_min": 63.30, "lat_max": 63.55,
            "lon_min": 10.20, "lon_max": 10.60
        },
        "Bodø": {
            "lat_min": 67.10, "lat_max": 67.40,
            "lon_min": 14.20, "lon_max": 14.60
        },
        "Tromsø": {
            "lat_min": 69.50, "lat_max": 69.80,
            "lon_min": 18.60, "lon_max": 19.20
        },
        "Lillehammer": {
            "lat_min": 61.00, "lat_max": 61.20,
            "lon_min": 10.20, "lon_max": 10.70
        },
        "Alta": {
            "lat_min": 69.80, "lat_max": 70.10,
            "lon_min": 23.00, "lon_max": 23.50
        }
    }


def plot_time_series_with_subzero(tn_mean, city, year, write2file):
    """
    Plots a time series of daily min temperatures for DJF, shading days below 0°C.
    
    Parameters
    ----------
    tn_mean : xarray.DataArray
        The spatially averaged temperature time series.
    city : str
        City name (for plot title).
    year : int
        The main DJF year (e.g., 2025 means Dec(2024) + Jan/Feb(2025)).
    pdf_out : str or None
        If provided, the path/filename to save the plot as a PDF. Otherwise, no file is saved.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Convert the DataArray to numpy for fill_between
    time_vals = tn_mean["time"].values
    temp_vals = tn_mean.values

    # Plot the daily min temperature
    ax.plot(time_vals, temp_vals, color="blue")

    # Draw a zero reference line
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # Shade the area below zero
    ax.fill_between(
        time_vals,
        temp_vals,
        0,
        where=(temp_vals < 0),
        color="blue",
        alpha=0.3,
        label="Below 0°C"
    )

    # Title and labels
    ax.set_title(f"DJF {year} {city}")
    if variable == 'tn':
        ylabel = 'daily minimum temperature (°C)'
    elif variable == 'tg':
        ylabel = 'daily mean temperature (°C)'
    ax.set_ylabel(ylabel)

    # Compute sub-zero counts for each month
    tn_dec = tn_mean.sel(time=tn_mean.time.dt.month == 12)
    tn_jan = tn_mean.sel(time=tn_mean.time.dt.month == 1)
    tn_feb = tn_mean.sel(time=tn_mean.time.dt.month == 2)

    dec_count = (tn_dec < 0).sum().item()
    jan_count = (tn_jan < 0).sum().item()
    feb_count = (tn_feb < 0).sum().item()

    # Annotate with sub-zero day counts
    info_text = (
        f"Days < 0°C:\n"
        f"Dec {year-1}: {dec_count}\n"
        f"Jan {year}: {jan_count}\n"
        f"Feb {year}: {feb_count}"
    )
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
            va='top', ha='left', bbox=dict(boxstyle="round", alpha=0.2))

    # Set major ticks at 1st and 15th of each month
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    
    # Format tick labels as abbreviated month + day (e.g., Dec 1, Jan 15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    #ax.legend(loc='best')
    plt.tight_layout()

    if write2file:
        filename_out = path_out + variable + '/' + f"{city}_DJF_{year}_{variable}.pdf"
        plt.savefig(filename_out, format='pdf')

    plt.show()


def plot_bounding_box_map(ds, city, bbox, write2file):
    """
    Plots a map with the native data grid and the bounding box for the given city,
    using Cartopy for geographic context.

    Parameters
    ----------
    ds : xarray Dataset (or DataArray)
        Contains lat/lon coordinates.
    city : str
        City name (string).
    bbox : dict
        Dictionary with lat_min, lat_max, lon_min, lon_max.
    pdf_out : str or None
        If provided, the path/filename to save the map as a PDF. Otherwise, no file is saved.
    """

    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]

    if "time" in ds.dims:
        ds_plot = ds.isel(time=0)
    else:
        ds_plot = ds

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Add some geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linestyle=':')
    ax.add_feature(cfeature.LAND,      alpha=0.3)
    ax.add_feature(cfeature.LAKES,     alpha=0.5)
    ax.add_feature(cfeature.RIVERS,    alpha=0.5)

    # Draw bounding box in red
    ax.plot(
        [lon_min, lon_max], [lat_min, lat_min],
        color="red", transform=ccrs.PlateCarree()
    )
    ax.plot(
        [lon_min, lon_max], [lat_max, lat_max],
        color="red", transform=ccrs.PlateCarree()
    )
    ax.plot(
        [lon_min, lon_min], [lat_min, lat_max],
        color="red", transform=ccrs.PlateCarree()
    )
    ax.plot(
        [lon_max, lon_max], [lat_min, lat_max],
        color="red", transform=ccrs.PlateCarree()
    )

    # Zoom the map around the bounding box
    ax.set_extent(
        [lon_min - 1, lon_max + 1, lat_min - 0.5, lat_max + 0.5],
        crs=ccrs.PlateCarree()
    )

    ax.set_title(f"{city} Area: Data Grid + Bounding Box")
    plt.tight_layout()

    if write2file:
        filename_out = path_out + 'maps/' + f"{city}_map.pdf"
        plt.savefig(filename_out, format='pdf')

    plt.show()

    
def analyze_djf_temperature(year,city,variable,path_in,path_out,write2file=False):
    """
    Analyzes DJF (December, January, February) daily minimum temperature for a given city.

    :param year: The "main" DJF year. For DJF 2025, we consider Dec 2024 + Jan/Feb 2025.
    :param city: Name of the city for analysis (default: 'Oslo').
    :param variable: Variable name in the netCDF files (e.g., 'tn').
    :param path_in: Directory path to the netCDF files.
    :param write2file: Whether to save output data to a file (the netCDF result).
    """
    if city is None:
        city = "Oslo"  # default if no city is provided

    # 1) Retrieve bounding box
    bboxes = get_city_bboxes()
    if city not in bboxes:
        raise ValueError(f"City '{city}' not found in bounding-box dictionary.")
    bbox = bboxes[city]

    # 2) Gather the necessary data for Dec(year-1), Jan/Feb(year)
    year_prev = year - 1
    dec_file = f"{path_in}/{variable}/{variable}_{year_prev}.nc"
    jf_file  = f"{path_in}/{variable}/{variable}_{year}.nc"

    ds_dec = xr.open_dataset(dec_file).sel(time=slice(f"{year_prev}-12-01", f"{year_prev}-12-31"))
    ds_jf  = xr.open_dataset(jf_file).sel(time=slice(f"{year}-01-01", f"{year}-02-28"))
    ds_djf = xr.concat([ds_dec, ds_jf], dim='time')

    # 3) Spatial subset to bounding box
    ds_subset = ds_djf.where(
        (ds_djf["lat"] >= bbox["lat_min"]) & (ds_djf["lat"] <= bbox["lat_max"]) &
        (ds_djf["lon"] >= bbox["lon_min"]) & (ds_djf["lon"] <= bbox["lon_max"]),
        drop=True
    )

    # 4) Compute the spatial mean across Y, X
    tn_mean = ds_subset[variable].mean(dim=["Y", "X"])

    # write to file
    #if write2file:
    #    filename_out = path_out + f"{city}_DJF_{year}_{variable}_mean.nc"
    #    tn_mean.to_netcdf(filename_out)

    return tn_mean,ds_djf,bbox


# -- Main script ------------------------------------------------------------
if __name__ == "__main__":

    tn_mean, ds_djf, bbox = analyze_djf_temperature(year,city,variable,path_in,path_out,write2file)

    plot_time_series_with_subzero(tn_mean, city, year, write2file)

    #plot_bounding_box_map(ds_djf, city, bbox, write2file)
