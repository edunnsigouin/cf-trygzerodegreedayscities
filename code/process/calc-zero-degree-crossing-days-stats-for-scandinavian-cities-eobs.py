"""
Create historical statistics for zero-degree crossing days around Scandinavian cities.

A zero-degree crossing day is defined as a day where:
    tn < 0  and  tx > 0
where tn and tx are daily minimum and maximum 2m temperature.

The script is organized as follows:
1) User-defined settings:
   - choose dataset (E-OBS now, ERA5 later)
   - choose years, season(s), city/cities, box sizes, and spatial method
   - choose whether to write CSV and/or NetCDF

2) Read daily tn and tx files year by year:
   - for E-OBS, files are assumed to look like:
         tn_0.1x0.1_1950.nc
         tx_0.1x0.1_1950.nc
   - files are expected in:
         config.dirs['eobs_raw'] + 'tn'
         config.dirs['eobs_raw'] + 'tx'
   - the script is written so ERA5 can be added later by editing the
     dataset configuration block only

3) For each city and each box size:
   - subset a lat/lon box around the city center
   - compute daily zero-degree crossing flags
   - assign season and "season-year"
   - important: DJF is assigned to the year of January-February
     (e.g. Dec 2003 + Jan/Feb 2004 belongs to DJF 2004)

4) Two different spatial methods are available:
   A) spatial_method = "gridpoint_mean"
      - first calculate zero-degree crossing days at each individual grid cell
      - then aggregate to seasonal counts / percentages at each grid cell
      - finally average those seasonal values across all grid cells in the box
      - interpretation:
            "average seasonal zero-crossing statistics per grid cell in the box"

   B) spatial_method = "city_mean"
      - first calculate daily box-mean tn and tx
      - then define crossing days from these box-mean daily temperatures
      - then aggregate to seasonal counts / percentages
      - interpretation:
            "seasonal zero-crossing statistics of the city-box mean temperature"

   These two methods are not equivalent:
   - "gridpoint_mean" preserves spatial variability within the city box
   - "city_mean" gives one representative daily temperature series for the box

5) For each year, city, box size, and season:
   - compute:
       a) zdc_days : number of zero-degree crossing days
       b) zdc_pct  : percentage of valid days in that season that are
                     zero-degree crossing days

6) Output:
   - xarray Dataset with dimensions:
         (year, city, box_size_index, season)
   - optional write to NetCDF and/or CSV

7) Metadata written to the NetCDF output:
   - spatial_method is stored as global metadata
   - city center latitude / longitude are stored as coordinates on the city dimension
   - box_size_index is stored as a coordinate dimension
   - the numeric half-width of each box is stored as:
         box_size_delta(box_size_index)
   - this makes the file self-describing and machine-readable

Notes:
- The code assumes the raw temperature files contain dimensions
  (time, latitude, longitude).
- Missing values are handled through xarray decoding.
- The script currently starts with Oslo, but more cities can be added
  to the CITY_COORDS dictionary.
"""

# ---------------------------------------------------------------------
# 1) imports
# ---------------------------------------------------------------------
import os
from pathlib                  import Path
import numpy                  as np
import pandas                 as pd
import xarray                 as xr
from trygzerodegreedayscities import config


# ---------------------------------------------------------------------
# 2) user defined input parameters
# ---------------------------------------------------------------------
dataset = "eobs"                # "eobs" or "era5" (era5 config placeholder included)
years = [2004, 2024]            # inclusive range: [start_year, end_year]
season = "all"                  # "djf", "mam", "jja", "son", or "all"

# Spatial method:
#   "gridpoint_mean" -> crossing condition at each grid point, then spatial mean
#   "city_mean"      -> spatial mean of tn and tx first, then crossing condition
spatial_method = "gridpoint_mean"

# Output control
write2csv = True
write2nc = True
output_dir = config.dirs["eobs_processed"]

# City definitions: start with Oslo
CITY_COORDS = {
    "Oslo":       {"lat": 59.9139, "lon": 10.7522},
    "Bergen":     {"lat": 60.3913, "lon": 5.3221},
    "Trondheim":  {"lat": 63.4305, "lon": 10.3951},
    "Copenhagen": {"lat": 55.6761, "lon": 12.5683},
    "Aarhus":     {"lat": 56.1629, "lon": 10.2039},
    "Odense":     {"lat": 55.4038, "lon": 10.4024},
    "Stockholm":  {"lat": 59.3293, "lon": 18.0686},
    "Gothenburg": {"lat": 57.7089, "lon": 11.9746},
    "Malmo":      {"lat": 55.6050, "lon": 13.0038},
}

# Box sizes around each city center.
# delta means:
#   lat in [lat0 - delta, lat0 + delta]
#   lon in [lon0 - delta, lon0 + delta]
BOX_SIZE_DELTAS = {
    "small": 0.1,
    "medium": 0.2,
    "large": 0.3,
}

# Dataset-specific configuration.
# Edit the ERA5 paths / filenames later when needed.
DATASET_CONFIG = {
    "eobs": {
        "tn_dir": os.path.join(config.dirs["eobs_raw"], "tn"),
        "tx_dir": os.path.join(config.dirs["eobs_raw"], "tx"),
        "tn_var": "tn",
        "tx_var": "tx",
        "lat_name": "latitude",
        "lon_name": "longitude",
        "file_template_tn": "tn_0.1x0.1_{year}.nc",
        "file_template_tx": "tx_0.1x0.1_{year}.nc",
    },
    "era5": {
        # Placeholder config for later use.
        # Update these paths and file templates when you switch to ERA5.
        "tn_dir": os.path.join(config.dirs.get("era5_raw", ""), "tn"),
        "tx_dir": os.path.join(config.dirs.get("era5_raw", ""), "tx"),
        "tn_var": "tn",
        "tx_var": "tx",
        "lat_name": "latitude",
        "lon_name": "longitude",
        "file_template_tn": "tn_0.25x0.25_{year}.nc",
        "file_template_tx": "tx_0.25x0.25_{year}.nc",
    },
}


# ---------------------------------------------------------------------
# 3) functions
# ---------------------------------------------------------------------
def get_year_list(years_in):
    """Return an inclusive list of integer years."""
    if len(years_in) != 2:
        raise ValueError("years must be given as [start_year, end_year].")
    y0, y1 = int(years_in[0]), int(years_in[1])
    if y1 < y0:
        raise ValueError("end_year must be >= start_year.")
    return list(range(y0, y1 + 1))


def get_season_list(season_in):
    """Return a normalized list of seasons."""
    valid = ["djf", "mam", "jja", "son"]
    s = season_in.lower()
    if s == "all":
        return valid
    if s not in valid:
        raise ValueError(f"season must be one of {valid + ['all']}")
    return [s]


def build_file_path(dataset_name, var_type, year):
    """
    Build file path for tn or tx for a given year.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier, e.g. 'eobs' or 'era5'.
    var_type : str
        Either 'tn' or 'tx'.
    year : int
        Year of file to open.
    """
    cfg = DATASET_CONFIG[dataset_name]

    if var_type == "tn":
        return os.path.join(cfg["tn_dir"], cfg["file_template_tn"].format(year=year))
    if var_type == "tx":
        return os.path.join(cfg["tx_dir"], cfg["file_template_tx"].format(year=year))

    raise ValueError("var_type must be 'tn' or 'tx'.")


def subset_latlon(ds, lat0, lon0, delta, lat_name="latitude", lon_name="longitude"):
    """
    Subset a rectangular lat/lon box around a city center.

    The selected box is:
        lat in [lat0-delta, lat0+delta]
        lon in [lon0-delta, lon0+delta]

    Works for coordinates that are either ascending or descending in latitude.
    """
    lat_min, lat_max = lat0 - delta, lat0 + delta
    lon_min, lon_max = lon0 - delta, lon0 + delta

    lat_values = ds[lat_name].values
    if lat_values[0] < lat_values[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    ds_sub = ds.sel(
        {
            lat_name: lat_slice,
            lon_name: slice(lon_min, lon_max),
        }
    )

    if ds_sub.sizes.get(lat_name, 0) == 0 or ds_sub.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            f"No grid cells found in box around lat={lat0}, lon={lon0}, delta={delta}"
        )

    return ds_sub


def open_tn_tx_for_box(dataset_name, year, lat0, lon0, delta):
    """
    Open tn and tx for one year, subset to the requested city box,
    and return a Dataset with only the required spatial domain loaded.
    """
    cfg = DATASET_CONFIG[dataset_name]

    tn_path = build_file_path(dataset_name, "tn", year)
    tx_path = build_file_path(dataset_name, "tx", year)

    if not os.path.exists(tn_path):
        raise FileNotFoundError(f"Missing file: {tn_path}")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing file: {tx_path}")

    with xr.open_dataset(tn_path) as ds_tn, xr.open_dataset(tx_path) as ds_tx:
        ds_tn = subset_latlon(
            ds_tn,
            lat0=lat0,
            lon0=lon0,
            delta=delta,
            lat_name=cfg["lat_name"],
            lon_name=cfg["lon_name"],
        )
        ds_tx = subset_latlon(
            ds_tx,
            lat0=lat0,
            lon0=lon0,
            delta=delta,
            lat_name=cfg["lat_name"],
            lon_name=cfg["lon_name"],
        )

        ds_out = xr.Dataset(
            {
                "tn": ds_tn[cfg["tn_var"]],
                "tx": ds_tx[cfg["tx_var"]],
            }
        ).load()

    return ds_out


def assign_season_and_season_year(ds):
    """
    Add:
      - season: ["djf", "mam", "jja", "son"]
      - season_year: integer year, with DJF assigned to Jan-Feb year

    Example:
      Dec 2003, Jan 2004, Feb 2004 -> season='djf', season_year=2004
    """
    month = ds["time"].dt.month
    year = ds["time"].dt.year

    season_coord = xr.where(
        month.isin([12, 1, 2]), "djf",
        xr.where(
            month.isin([3, 4, 5]), "mam",
            xr.where(month.isin([6, 7, 8]), "jja", "son")
        )
    )

    # December belongs to the following DJF year.
    season_year = xr.where(month == 12, year + 1, year)

    ds = ds.assign_coords(
        season=("time", season_coord.data),
        season_year=("time", season_year.data),
    )
    return ds


def spatial_mean_temperature(ds, lat_name="latitude", lon_name="longitude"):
    """
    Compute daily box-mean tn and tx.

    Missing values are ignored in the spatial mean.
    """
    return xr.Dataset(
        {
            "tn": ds["tn"].mean(dim=[lat_name, lon_name], skipna=True),
            "tx": ds["tx"].mean(dim=[lat_name, lon_name], skipna=True),
        }
    )


def compute_zero_degree_crossing(ds):
    """
    Compute daily zero-degree crossing flags and valid-day mask.

    This function works both for:
    - gridded input with dimensions (time, latitude, longitude)
    - box-mean input with dimension (time)
    """
    valid = xr.ufuncs.isfinite(ds["tn"]) & xr.ufuncs.isfinite(ds["tx"])
    crossing = (ds["tn"] < 0.0) & (ds["tx"] > 0.0) & valid

    return xr.Dataset(
        {
            "crossing": crossing.astype(np.int16),
            "valid": valid.astype(np.int16),
        },
        coords=ds.coords,
    )


def aggregate_crossing_by_season(ds_cross):
    """
    Aggregate daily crossing and valid flags to seasonal statistics.

    Returns a Dataset with:
      - zdc_days
      - zdc_pct
      - n_valid_days
    """
    crossing_days = ds_cross["crossing"].groupby(["season_year", "season"]).sum("time")
    valid_days = ds_cross["valid"].groupby(["season_year", "season"]).sum("time")
    crossing_pct = xr.where(valid_days > 0, 100.0 * crossing_days / valid_days, np.nan)

    return xr.Dataset(
        {
            "zdc_days": crossing_days,
            "zdc_pct": crossing_pct,
            "n_valid_days": valid_days,
        }
    )


def reduce_gridpoint_stats_to_box(ds_stats, lat_name="latitude", lon_name="longitude"):
    """
    Average seasonal grid-point statistics across the box.

    Used when:
        spatial_method = 'gridpoint_mean'
    """
    return ds_stats.mean(dim=[lat_name, lon_name], skipna=True)


def compute_seasonal_stats_for_box(
    dataset_name,
    city_name,
    lat0,
    lon0,
    box_size_index,
    delta,
    years_list,
    seasons,
    spatial_method="gridpoint_mean",
):
    """
    For one city and one box size:
    - open all required yearly subsets
    - compute zero-degree crossing statistics
    - aggregate by season and season_year
    - return Dataset with dims (year, season)
    """
    cfg = DATASET_CONFIG[dataset_name]
    lat_name = cfg["lat_name"]
    lon_name = cfg["lon_name"]

    # Include the previous year so December can be used for DJF of the first year.
    read_years = list(range(min(years_list) - 1, max(years_list) + 1))

    ds_list = []
    for year in read_years:
        ds_y = open_tn_tx_for_box(dataset_name, year, lat0, lon0, delta)
        ds_list.append(ds_y)

    ds = xr.concat(ds_list, dim="time").sortby("time")
    ds = assign_season_and_season_year(ds)

    # Keep only requested seasons and requested season-years.
    ds = ds.where(ds["season"].isin(seasons), drop=True)
    ds = ds.where(
        (ds["season_year"] >= min(years_list)) &
        (ds["season_year"] <= max(years_list)),
        drop=True,
    )

    if spatial_method == "gridpoint_mean":
        # Crossing at each grid point first, then seasonal statistics, then spatial mean.
        ds_cross = compute_zero_degree_crossing(ds)
        ds_stats = aggregate_crossing_by_season(ds_cross)
        ds_out = reduce_gridpoint_stats_to_box(
            ds_stats,
            lat_name=lat_name,
            lon_name=lon_name,
        )

    elif spatial_method == "city_mean":
        # Spatial mean first, then crossing from the box-mean daily series.
        ds_mean = spatial_mean_temperature(ds, lat_name=lat_name, lon_name=lon_name)
        ds_mean = ds_mean.assign_coords(
            season=ds["season"],
            season_year=ds["season_year"],
        )
        ds_cross = compute_zero_degree_crossing(ds_mean)
        ds_out = aggregate_crossing_by_season(ds_cross)

    else:
        raise ValueError("spatial_method must be 'gridpoint_mean' or 'city_mean'.")

    # Rename dimension season_year -> year for final output.
    ds_out = ds_out.rename({"season_year": "year"})

    # Add city and box-size dimensions.
    ds_out = ds_out.expand_dims(
        city=[city_name],
        box_size_index=[box_size_index],
    )

    return ds_out


def combine_all_cities_and_boxes(
    dataset_name,
    city_coords,
    box_size_deltas,
    years_list,
    seasons,
    spatial_method="gridpoint_mean",
):
    """
    Loop over all cities and box sizes and combine the output into
    one Dataset with dimensions:
        (year, city, box_size_index, season)
    """
    all_ds = []

    for city_name, coord in city_coords.items():
        lat0 = coord["lat"]
        lon0 = coord["lon"]

        for box_name, delta in box_size_deltas.items():
            print(
                f"Processing {city_name:10s} | box={box_name:6s} | "
                f"delta={delta:.2f} | method={spatial_method}"
            )

            ds_box = compute_seasonal_stats_for_box(
                dataset_name=dataset_name,
                city_name=city_name,
                lat0=lat0,
                lon0=lon0,
                box_size_index=box_name,
                delta=delta,
                years_list=years_list,
                seasons=seasons,
                spatial_method=spatial_method,
            )
            all_ds.append(ds_box)

    ds_out = xr.combine_by_coords(all_ds)

    # Reorder dimensions to the requested output structure.
    dim_order = [d for d in ["year", "city", "box_size_index", "season"] if d in ds_out.dims]
    ds_out = ds_out.transpose(*dim_order)

    return ds_out


def add_city_and_box_coordinates(ds_out, city_coords, box_size_deltas):
    """
    Add city center latitude/longitude and numeric box deltas as coordinates.

    Added coordinates:
      - city_lat(city)
      - city_lon(city)
      - box_size_delta(box_size_index)
    """
    city_names = ds_out["city"].values
    box_names = ds_out["box_size_index"].values

    city_lat = [city_coords[city]["lat"] for city in city_names]
    city_lon = [city_coords[city]["lon"] for city in city_names]
    box_delta = [box_size_deltas[box] for box in box_names]

    ds_out = ds_out.assign_coords(
        city_lat=("city", city_lat),
        city_lon=("city", city_lon),
        box_size_delta=("box_size_index", box_delta),
    )

    ds_out["city_lat"].attrs = {
        "long_name": "City center latitude",
        "units": "degrees_north",
        "description": "Central latitude used to define the city box.",
    }

    ds_out["city_lon"].attrs = {
        "long_name": "City center longitude",
        "units": "degrees_east",
        "description": "Central longitude used to define the city box.",
    }

    ds_out["box_size_delta"].attrs = {
        "long_name": "Half-width of latitude-longitude box around city center",
        "units": "degrees",
        "description": (
            "The selected box is [lat-delta, lat+delta] x [lon-delta, lon+delta]. "
            "This coordinate stores the numeric delta corresponding to each "
            "box_size_index."
        ),
    }

    return ds_out


def add_output_metadata(ds_out, dataset_name, season_name, spatial_method, box_size_deltas):
    """
    Add descriptive metadata to the output Dataset so the NetCDF file
    is self-explanatory.
    """
    ds_out["zdc_days"].attrs = {
        "long_name": "Seasonal number of zero-degree crossing days",
        "units": "days",
        "description": (
            "A zero-degree crossing day is defined as tn < 0 C and tx > 0 C "
            "on the same day."
        ),
    }

    ds_out["zdc_pct"].attrs = {
        "long_name": "Seasonal percentage of zero-degree crossing days",
        "units": "%",
        "description": (
            "Percentage of valid days in each season that satisfy "
            "tn < 0 C and tx > 0 C."
        ),
    }

    ds_out["n_valid_days"].attrs = {
        "long_name": "Seasonal number of valid days used in the calculation",
        "units": "days",
        "description": (
            "Number of days with valid tn and tx values used when computing "
            "zdc_days and zdc_pct."
        ),
    }

    ds_out["year"].attrs = {
        "long_name": "Season-year",
        "description": (
            "For DJF, December is assigned to the following year. "
            "For example, Dec 2003 + Jan-Feb 2004 is labeled as DJF 2004."
        ),
    }

    ds_out["season"].attrs = {
        "long_name": "Meteorological season",
        "description": "One of djf, mam, jja, son.",
    }

    ds_out["city"].attrs = {
        "long_name": "City name",
        "description": "Name of city used to define the center of the spatial box.",
    }

    ds_out["box_size_index"].attrs = {
        "long_name": "Box size category around city center",
        "description": (
            "Categorical label for the spatial box around the city center. "
            "The corresponding numeric half-width in degrees is stored in "
            "the box_size_delta coordinate."
        ),
    }

    ds_out.attrs = {
        "title": "Historical seasonal statistics of zero-degree crossing days",
        "summary": (
            "Seasonal statistics of zero-degree crossing days for selected cities "
            "and box sizes. A zero-degree crossing day is defined as tn < 0 C "
            "and tx > 0 C on the same day."
        ),
        "dataset": dataset_name,
        "season_request": season_name,
        "spatial_method": spatial_method,
        "spatial_method_description": (
            "gridpoint_mean: zero-degree crossing is computed at each grid cell, "
            "seasonal statistics are computed per grid cell, and then averaged "
            "across the box. "
            "city_mean: daily tn and tx are first averaged across the box, then "
            "zero-degree crossing is computed from the box-mean daily series."
        ),
        "box_definition": (
            "For each city and box size, the selected spatial domain is "
            "[lat-delta, lat+delta] x [lon-delta, lon+delta], where lat/lon "
            "are the city center coordinates and delta is the half-width "
            "stored in box_size_delta."
        ),
        "box_size_delta_mapping": ", ".join(
            [f"{k}={v}" for k, v in box_size_deltas.items()]
        ),
        "Conventions": "CF-1.8",
    }

    return ds_out


def write_outputs(ds_out, output_dir, dataset_name, season_name, spatial_method,
                  years_list, write2csv=False, write2nc=False):
    """
    Write output Dataset to NetCDF and/or CSV.

    The filename includes:
      - dataset
      - season selection
      - spatial method
      - year range (start-end)

    Example:
      zero_degree_crossing_stats_eobs_all_gridpoint_mean_2004-2024.nc
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    year_start = min(years_list)
    year_end = max(years_list)

    file_stub = (
        f"scandinavian_city_zero_degree_crossing_stats_"
        f"{dataset_name}_{season_name}_{spatial_method}_"
        f"{year_start}-{year_end}"
    )

    if write2nc:
        nc_path = os.path.join(output_dir, f"{file_stub}.nc")
        ds_out.to_netcdf(nc_path)
        print(f"Wrote NetCDF: {nc_path}")

    if write2csv:
        csv_path = os.path.join(output_dir, f"{file_stub}.csv")
        df = ds_out.to_dataframe().reset_index()
        df.to_csv(csv_path, index=False)
        print(f"Wrote CSV: {csv_path}")


# ---------------------------------------------------------------------
# 4) main script
# ---------------------------------------------------------------------
if __name__ == "__main__":

    years_list = get_year_list(years)
    seasons = get_season_list(season)

    ds_out = combine_all_cities_and_boxes(
        dataset_name=dataset,
        city_coords=CITY_COORDS,
        box_size_deltas=BOX_SIZE_DELTAS,
        years_list=years_list,
        seasons=seasons,
        spatial_method=spatial_method,
    )

    ds_out = add_city_and_box_coordinates(
        ds_out=ds_out,
        city_coords=CITY_COORDS,
        box_size_deltas=BOX_SIZE_DELTAS,
    )

    ds_out = add_output_metadata(
        ds_out=ds_out,
        dataset_name=dataset,
        season_name=season,
        spatial_method=spatial_method,
        box_size_deltas=BOX_SIZE_DELTAS,
    )

    print(ds_out)

    write_outputs(
        ds_out=ds_out,
        output_dir=output_dir,
        dataset_name=dataset,
        season_name=season,
        spatial_method=spatial_method,
        years_list=years_list,  
        write2csv=write2csv,
        write2nc=write2nc,
    )
