"""
Create historical seasonal statistics for zero-degree crossing days
around Scandinavian cities.

The script can calculate either:

1) Zero-degree crossing days without precipitation:
       tn < 0 and tx > 0

2) Zero-degree crossing days with precipitation:
       tn < 0 and tx > 0 and tp > 0

Use:

    include_precipitation = True

to include the tp > 0 condition.

Use:

    include_precipitation = False

to ignore precipitation.

Output filenames match the plotting script:

With precipitation:
    scandinavian_city_zero_degree_crossing_with_precipitation_stats_...

Without precipitation:
    scandinavian_city_zero_degree_crossing_with_stats_...
"""

# ---------------------------------------------------------------------
# 1) imports
# ---------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from trygzerodegreedayscities import config


# ---------------------------------------------------------------------
# 2) user input parameters
# ---------------------------------------------------------------------
dataset = "eobs"                  # "eobs" or "era5"
years = [1951, 2024]              # inclusive range: [start_year, end_year]
season = "all"                    # "djf", "mam", "jja", "son", or "all"

# Include precipitation condition?
# True  -> event is tn < 0 and tx > 0 and tp > 0
# False -> event is tn < 0 and tx > 0
include_precipitation = False

# Spatial method:
#   "gridpoint_mean" -> event condition at each grid point, then spatial mean
#   "city_mean"      -> spatial mean first, then event condition
spatial_method = "gridpoint_mean"

# Output control
write2csv = True
write2nc = True
output_dir = config.dirs["eobs_processed"]

# Reference-year settings for snapping city centers to valid grid cells.
snap_reference_year = None        # None -> use years_list[0]
snap_valid_fraction_threshold = 0.95

# City definitions
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
    "Tromso":     {"lat": 69.6492, "lon": 18.9553},
}


# Box sizes around each city center.
# delta means:
#   lat in [lat0 - delta, lat0 + delta]
#   lon in [lon0 - delta, lon0 + delta]
BOX_SIZE_DELTAS = {
    "small": 0.0,
    "medium": 0.1,
    "large": 0.2,
}


# ---------------------------------------------------------------------
# 3) dataset configuration
# ---------------------------------------------------------------------
DATASET_CONFIG = {
    "eobs": {
        "tn_dir": os.path.join(config.dirs["eobs_raw"], "tn"),
        "tx_dir": os.path.join(config.dirs["eobs_raw"], "tx"),
        "tp_dir": os.path.join(config.dirs["eobs_raw"], "tp"),
        "tn_var": "tn",
        "tx_var": "tx",
        "tp_var": "tp",
        "lat_name": "latitude",
        "lon_name": "longitude",
        "file_template_tn": "tn_0.1x0.1_{year}.nc",
        "file_template_tx": "tx_0.1x0.1_{year}.nc",
        "file_template_tp": "tp_0.1x0.1_{year}.nc",
    },
    "era5": {
        # Placeholder config for later use.
        "tn_dir": os.path.join(config.dirs.get("era5_raw", ""), "tn"),
        "tx_dir": os.path.join(config.dirs.get("era5_raw", ""), "tx"),
        "tp_dir": os.path.join(config.dirs.get("era5_raw", ""), "tp"),
        "tn_var": "tn",
        "tx_var": "tx",
        "tp_var": "tp",
        "lat_name": "latitude",
        "lon_name": "longitude",
        "file_template_tn": "tn_0.25x0.25_{year}.nc",
        "file_template_tx": "tx_0.25x0.25_{year}.nc",
        "file_template_tp": "tp_0.25x0.25_{year}.nc",
    },
}


# ---------------------------------------------------------------------
# 4) simple helper functions
# ---------------------------------------------------------------------
def get_year_list(years_in):
    """Return inclusive list of years."""
    if len(years_in) != 2:
        raise ValueError("years must be given as [start_year, end_year].")

    y0, y1 = int(years_in[0]), int(years_in[1])

    if y1 < y0:
        raise ValueError("end_year must be >= start_year.")

    return list(range(y0, y1 + 1))


def get_season_list(season_in):
    """Return normalized season list."""
    valid = ["djf", "mam", "jja", "son"]
    s = season_in.lower()

    if s == "all":
        return valid

    if s not in valid:
        raise ValueError(f"season must be one of {valid + ['all']}")

    return [s]


def build_file_path(dataset_name, var_type, year):
    """Build path to tn, tx, or tp file."""
    cfg = DATASET_CONFIG[dataset_name]

    if var_type == "tn":
        return os.path.join(cfg["tn_dir"], cfg["file_template_tn"].format(year=year))

    if var_type == "tx":
        return os.path.join(cfg["tx_dir"], cfg["file_template_tx"].format(year=year))

    if var_type == "tp":
        return os.path.join(cfg["tp_dir"], cfg["file_template_tp"].format(year=year))

    raise ValueError("var_type must be 'tn', 'tx', or 'tp'.")


def get_output_file_stub(
    dataset_name,
    season_name,
    spatial_method,
    years_list,
    include_precipitation,
):
    """Build output filename stem matching the plotting script."""
    year_start = min(years_list)
    year_end = max(years_list)

    if include_precipitation:
        prefix = "scandinavian_city_zero_degree_crossing_with_precipitation_stats"
    else:
        prefix = "scandinavian_city_zero_degree_crossing_with_stats"

    return (
        f"{prefix}_"
        f"{dataset_name}_{season_name}_{spatial_method}_"
        f"{year_start}-{year_end}"
    )


def get_event_description(include_precipitation):
    """Return readable event definition."""
    if include_precipitation:
        return "tn < 0 C, tx > 0 C, and tp > 0"

    return "tn < 0 C and tx > 0 C"


# ---------------------------------------------------------------------
# 5) spatial selection and city snapping
# ---------------------------------------------------------------------
def find_nearest_valid_gridpoint(
    ds_tn,
    ds_tx,
    ds_tp=None,
    lat0=None,
    lon0=None,
    lat_name="latitude",
    lon_name="longitude",
    valid_fraction_threshold=0.95,
    include_precipitation=True,
):
    """
    Find the nearest valid grid point to the requested city coordinate.

    If include_precipitation=True, a valid grid point requires valid tn, tx, and tp.
    If include_precipitation=False, a valid grid point requires valid tn and tx.
    """
    valid_time = xr.ufuncs.isfinite(ds_tn) & xr.ufuncs.isfinite(ds_tx)

    if include_precipitation:
        if ds_tp is None:
            raise ValueError("ds_tp must be provided when include_precipitation=True.")
        valid_time = valid_time & xr.ufuncs.isfinite(ds_tp)

    valid_fraction = valid_time.mean(dim="time")
    valid_grid = valid_fraction > valid_fraction_threshold

    # Force consistent dimension order before using numpy indexing.
    valid_grid = valid_grid.transpose(lat_name, lon_name)

    lat_idx, lon_idx = np.where(valid_grid.values)

    if len(lat_idx) == 0:
        raise ValueError(
            "No valid grid cells found using "
            f"valid_fraction_threshold={valid_fraction_threshold}."
        )

    lat_vals = valid_grid[lat_name].values[lat_idx]
    lon_vals = valid_grid[lon_name].values[lon_idx]

    dist2 = (lat_vals - lat0) ** 2 + (lon_vals - lon0) ** 2
    idx = np.argmin(dist2)

    return float(lat_vals[idx]), float(lon_vals[idx])


def adjust_city_centers_to_valid_grid(
    dataset_name,
    city_coords,
    reference_year,
    valid_fraction_threshold=0.95,
    include_precipitation=True,
):
    """
    Snap each city center to the nearest valid grid point.

    This avoids centering a box on an ocean-only grid point.
    """
    cfg = DATASET_CONFIG[dataset_name]

    tn_path = build_file_path(dataset_name, "tn", reference_year)
    tx_path = build_file_path(dataset_name, "tx", reference_year)

    if not os.path.exists(tn_path):
        raise FileNotFoundError(f"Missing file: {tn_path}")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing file: {tx_path}")

    tp_path = None
    if include_precipitation:
        tp_path = build_file_path(dataset_name, "tp", reference_year)
        if not os.path.exists(tp_path):
            raise FileNotFoundError(f"Missing file: {tp_path}")

    adjusted = {}

    with xr.open_dataset(tn_path) as ds_tn, xr.open_dataset(tx_path) as ds_tx:
        tn = ds_tn[cfg["tn_var"]]
        tx = ds_tx[cfg["tx_var"]]

        if include_precipitation:
            with xr.open_dataset(tp_path) as ds_tp:
                tp = ds_tp[cfg["tp_var"]]

                for city_name, coord in city_coords.items():
                    lat_adj, lon_adj = find_nearest_valid_gridpoint(
                        ds_tn=tn,
                        ds_tx=tx,
                        ds_tp=tp,
                        lat0=coord["lat"],
                        lon0=coord["lon"],
                        lat_name=cfg["lat_name"],
                        lon_name=cfg["lon_name"],
                        valid_fraction_threshold=valid_fraction_threshold,
                        include_precipitation=True,
                    )

                    adjusted[city_name] = {
                        "lat": lat_adj,
                        "lon": lon_adj,
                        "orig_lat": coord["lat"],
                        "orig_lon": coord["lon"],
                    }

                    print(
                        f"{city_name:10s}: original=({coord['lat']:.4f}, {coord['lon']:.4f}) "
                        f"-> adjusted=({lat_adj:.4f}, {lon_adj:.4f})"
                    )

        else:
            for city_name, coord in city_coords.items():
                lat_adj, lon_adj = find_nearest_valid_gridpoint(
                    ds_tn=tn,
                    ds_tx=tx,
                    ds_tp=None,
                    lat0=coord["lat"],
                    lon0=coord["lon"],
                    lat_name=cfg["lat_name"],
                    lon_name=cfg["lon_name"],
                    valid_fraction_threshold=valid_fraction_threshold,
                    include_precipitation=False,
                )

                adjusted[city_name] = {
                    "lat": lat_adj,
                    "lon": lon_adj,
                    "orig_lat": coord["lat"],
                    "orig_lon": coord["lon"],
                }

                print(
                    f"{city_name:10s}: original=({coord['lat']:.4f}, {coord['lon']:.4f}) "
                    f"-> adjusted=({lat_adj:.4f}, {lon_adj:.4f})"
                )

    return adjusted


def subset_latlon(ds, lat0, lon0, delta, lat_name="latitude", lon_name="longitude"):
    """
    Subset a lat/lon box around a city center.

    If delta == 0, select the nearest single grid point.
    """
    if np.isclose(delta, 0.0):
        ds_sub = ds.sel(
            {lat_name: lat0, lon_name: lon0},
            method="nearest",
        )

        # Keep latitude and longitude as length-one dimensions.
        ds_sub = ds_sub.expand_dims(
            {
                lat_name: [ds_sub[lat_name].item()],
                lon_name: [ds_sub[lon_name].item()],
            }
        )

        return ds_sub

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
            f"No grid cells found around lat={lat0}, lon={lon0}, delta={delta}"
        )

    return ds_sub


# ---------------------------------------------------------------------
# 6) reading data
# ---------------------------------------------------------------------
def open_tn_tx_tp_for_box(
    dataset_name,
    year,
    lat0,
    lon0,
    delta,
    include_precipitation=True,
):
    """
    Open tn and tx, optionally tp, for one year and subset to city box.
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

        data_vars = {
            "tn": ds_tn[cfg["tn_var"]],
            "tx": ds_tx[cfg["tx_var"]],
        }

        if include_precipitation:
            tp_path = build_file_path(dataset_name, "tp", year)

            if not os.path.exists(tp_path):
                raise FileNotFoundError(f"Missing file: {tp_path}")

            with xr.open_dataset(tp_path) as ds_tp:
                ds_tp = subset_latlon(
                    ds_tp,
                    lat0=lat0,
                    lon0=lon0,
                    delta=delta,
                    lat_name=cfg["lat_name"],
                    lon_name=cfg["lon_name"],
                )

                data_vars["tp"] = ds_tp[cfg["tp_var"]]

        ds_out = xr.Dataset(data_vars).load()

    return ds_out


# ---------------------------------------------------------------------
# 7) season handling and event calculation
# ---------------------------------------------------------------------
def assign_season_and_season_year(ds):
    """
    Add season and season_year coordinates.

    DJF is assigned to the year of January-February.
    Example:
        Dec 2003 + Jan-Feb 2004 -> DJF 2004
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

    season_year = xr.where(month == 12, year + 1, year)

    ds = ds.assign_coords(
        season=("time", season_coord.data),
        season_year=("time", season_year.data),
    )

    return ds


def spatial_mean_temperature_precip(
    ds,
    lat_name="latitude",
    lon_name="longitude",
    include_precipitation=True,
):
    """
    Compute daily box-mean tn, tx, and optionally tp.

    Used by spatial_method='city_mean'.
    """
    data_vars = {
        "tn": ds["tn"].mean(dim=[lat_name, lon_name], skipna=True),
        "tx": ds["tx"].mean(dim=[lat_name, lon_name], skipna=True),
    }

    if include_precipitation:
        data_vars["tp"] = ds["tp"].mean(dim=[lat_name, lon_name], skipna=True)

    return xr.Dataset(data_vars)


def compute_zero_degree_crossing(ds, include_precipitation=True):
    """
    Compute daily event flags and valid-day mask.

    If include_precipitation=True:
        event = tn < 0 and tx > 0 and tp > 0

    If include_precipitation=False:
        event = tn < 0 and tx > 0
    """
    valid = xr.ufuncs.isfinite(ds["tn"]) & xr.ufuncs.isfinite(ds["tx"])
    crossing = (ds["tn"] < 0.0) & (ds["tx"] > 0.0)

    if include_precipitation:
        valid = valid & xr.ufuncs.isfinite(ds["tp"])
        crossing = crossing & (ds["tp"] > 0.0)

    crossing = crossing & valid

    return xr.Dataset(
        {
            "crossing": crossing.astype(np.int16),
            "valid": valid.astype(np.int16),
        },
        coords=ds.coords,
    )


def aggregate_crossing_by_season(ds_cross):
    """
    Aggregate daily event flags to seasonal statistics.

    Returns:
        zdc_days
        zdc_pct
        n_valid_days
    """
    crossing_days = ds_cross["crossing"].groupby(["season_year", "season"]).sum("time")
    valid_days = ds_cross["valid"].groupby(["season_year", "season"]).sum("time")

    crossing_pct = xr.where(
        valid_days > 0,
        100.0 * crossing_days / valid_days,
        np.nan,
    )

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

    Used by spatial_method='gridpoint_mean'.
    """
    return ds_stats.mean(dim=[lat_name, lon_name], skipna=True)


# ---------------------------------------------------------------------
# 8) main calculation
# ---------------------------------------------------------------------
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
    include_precipitation=True,
):
    """
    Compute seasonal statistics for one city and one box size.

    For gridpoint_mean:
        - event is computed at each grid cell
        - seasonal event counts are computed at each grid cell
        - zdc_days is averaged across grid cells
        - n_valid_days is a box-level count of valid calendar days

    For city_mean:
        - tn, tx, and optionally tp are averaged over the box first
        - event is computed from the box-mean daily time series
    """
    cfg = DATASET_CONFIG[dataset_name]
    lat_name = cfg["lat_name"]
    lon_name = cfg["lon_name"]

    # Include previous year so December is available for first DJF.
    read_years = list(range(min(years_list) - 1, max(years_list) + 1))

    ds_list = []

    for year in read_years:
        ds_y = open_tn_tx_tp_for_box(
            dataset_name=dataset_name,
            year=year,
            lat0=lat0,
            lon0=lon0,
            delta=delta,
            include_precipitation=include_precipitation,
        )
        ds_list.append(ds_y)

    ds = xr.concat(ds_list, dim="time").sortby("time")
    ds = assign_season_and_season_year(ds)

    # Keep only requested seasons and requested season-years.
    ds = ds.where(ds["season"].isin(seasons), drop=True)
    ds = ds.where(
        (ds["season_year"] >= min(years_list))
        & (ds["season_year"] <= max(years_list)),
        drop=True,
    )

    if spatial_method == "gridpoint_mean":
        ds_cross = compute_zero_degree_crossing(
            ds,
            include_precipitation=include_precipitation,
        )

        # Seasonal stats at each grid cell.
        ds_stats = aggregate_crossing_by_season(ds_cross)

        # Spatial mean for zdc_days.
        ds_out = reduce_gridpoint_stats_to_box(
            ds_stats,
            lat_name=lat_name,
            lon_name=lon_name,
        )

        # Box-level valid-day count:
        # a day is valid if at least one grid cell in the box is valid.
        valid_any = ds_cross["valid"].any(dim=[lat_name, lon_name])
        n_valid_days_box = valid_any.groupby(["season_year", "season"]).sum("time")
        ds_out["n_valid_days"] = n_valid_days_box

        # Recompute percentage with box-level denominator.
        ds_out["zdc_pct"] = xr.where(
            ds_out["n_valid_days"] > 0,
            100.0 * ds_out["zdc_days"] / ds_out["n_valid_days"],
            np.nan,
        )

    elif spatial_method == "city_mean":
        ds_mean = spatial_mean_temperature_precip(
            ds,
            lat_name=lat_name,
            lon_name=lon_name,
            include_precipitation=include_precipitation,
        )

        ds_mean = ds_mean.assign_coords(
            season=ds["season"],
            season_year=ds["season_year"],
        )

        ds_cross = compute_zero_degree_crossing(
            ds_mean,
            include_precipitation=include_precipitation,
        )

        ds_out = aggregate_crossing_by_season(ds_cross)

    else:
        raise ValueError("spatial_method must be 'gridpoint_mean' or 'city_mean'.")

    # Rename season_year to year for final output.
    ds_out = ds_out.rename({"season_year": "year"})

    # Add city and box dimensions.
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
    include_precipitation=True,
):
    """
    Loop over all cities and boxes and combine output into one Dataset.
    """
    all_ds = []

    for city_name, coord in city_coords.items():
        lat0 = coord["lat"]
        lon0 = coord["lon"]

        for box_name, delta in box_size_deltas.items():
            print(
                f"Processing {city_name:10s} | box={box_name:6s} | "
                f"delta={delta:.2f} | method={spatial_method} | "
                f"include_precipitation={include_precipitation} | "
                f"center=({lat0:.4f}, {lon0:.4f})"
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
                include_precipitation=include_precipitation,
            )

            all_ds.append(ds_box)

    ds_out = xr.combine_by_coords(all_ds)

    dim_order = [
        d for d in ["year", "city", "box_size_index", "season"]
        if d in ds_out.dims
    ]

    ds_out = ds_out.transpose(*dim_order)

    return ds_out


# ---------------------------------------------------------------------
# 9) metadata and output
# ---------------------------------------------------------------------
def add_city_and_box_coordinates(ds_out, city_coords, box_size_deltas):
    """Add city coordinates and box-size coordinates to output Dataset."""
    city_names = ds_out["city"].values
    box_names = ds_out["box_size_index"].values

    city_lat = [city_coords[city]["lat"] for city in city_names]
    city_lon = [city_coords[city]["lon"] for city in city_names]

    city_orig_lat = [
        city_coords[city].get("orig_lat", city_coords[city]["lat"])
        for city in city_names
    ]

    city_orig_lon = [
        city_coords[city].get("orig_lon", city_coords[city]["lon"])
        for city in city_names
    ]

    box_delta = [box_size_deltas[box] for box in box_names]

    ds_out = ds_out.assign_coords(
        city_lat=("city", city_lat),
        city_lon=("city", city_lon),
        city_orig_lat=("city", city_orig_lat),
        city_orig_lon=("city", city_orig_lon),
        box_size_delta=("box_size_index", box_delta),
    )

    ds_out["city_lat"].attrs = {
        "long_name": "Adjusted city center latitude used for extraction",
        "units": "degrees_north",
    }

    ds_out["city_lon"].attrs = {
        "long_name": "Adjusted city center longitude used for extraction",
        "units": "degrees_east",
    }

    ds_out["city_orig_lat"].attrs = {
        "long_name": "Original requested city latitude",
        "units": "degrees_north",
    }

    ds_out["city_orig_lon"].attrs = {
        "long_name": "Original requested city longitude",
        "units": "degrees_east",
    }

    ds_out["box_size_delta"].attrs = {
        "long_name": "Half-width of latitude-longitude box around city center",
        "units": "degrees",
        "description": (
            "The selected box is [lat-delta, lat+delta] x "
            "[lon-delta, lon+delta]."
        ),
    }

    return ds_out


def add_output_metadata(
    ds_out,
    dataset_name,
    season_name,
    spatial_method,
    box_size_deltas,
    snap_reference_year,
    snap_valid_fraction_threshold,
    include_precipitation=True,
):
    """Add descriptive metadata to output Dataset."""
    event_description = get_event_description(include_precipitation)

    if include_precipitation:
        title = (
            "Historical seasonal statistics of zero-degree crossing days "
            "with precipitation"
        )
    else:
        title = "Historical seasonal statistics of zero-degree crossing days"

    ds_out["zdc_days"].attrs = {
        "long_name": "Seasonal number of zero-degree crossing days",
        "units": "days",
        "description": f"A day is counted if {event_description}.",
    }

    ds_out["zdc_pct"].attrs = {
        "long_name": "Seasonal percentage of zero-degree crossing days",
        "units": "%",
        "description": (
            f"Percentage of valid days in each season that satisfy "
            f"{event_description}."
        ),
    }

    ds_out["n_valid_days"].attrs = {
        "long_name": "Seasonal number of valid days used in the calculation",
        "units": "days",
        "description": (
            "For gridpoint_mean, this is a box-level valid-day count: "
            "a day is valid if at least one grid cell in the box has all "
            "required variables. Required variables are tn and tx, plus tp "
            "when include_precipitation=True."
        ),
    }

    ds_out["year"].attrs = {
        "long_name": "Season-year",
        "description": (
            "For DJF, December is assigned to the following year. "
            "For example, Dec 2003 + Jan-Feb 2004 is labeled DJF 2004."
        ),
    }

    ds_out["season"].attrs = {
        "long_name": "Meteorological season",
        "description": "One of djf, mam, jja, son.",
    }

    ds_out["city"].attrs = {
        "long_name": "City name",
    }

    ds_out["box_size_index"].attrs = {
        "long_name": "Box size category around city center",
        "description": (
            "Categorical label for spatial box. Numeric half-width is stored "
            "in box_size_delta."
        ),
    }

    ds_out.attrs = {
        "title": title,
        "summary": (
            f"Seasonal statistics of days satisfying {event_description} "
            "for selected cities and box sizes."
        ),
        "event_definition": event_description,
        "include_precipitation": int(include_precipitation),
        "dataset": dataset_name,
        "season_request": season_name,
        "spatial_method": spatial_method,
        "spatial_method_description": (
            "gridpoint_mean: event is computed at each grid cell, seasonal "
            "statistics are computed per grid cell, and zdc_days is averaged "
            "across the box. n_valid_days is a box-level valid-day count. "
            "city_mean: daily variables are first averaged across the box, "
            "then the event is computed from the box-mean daily series."
        ),
        "box_definition": (
            "For each city and box size, the selected spatial domain is "
            "[lat-delta, lat+delta] x [lon-delta, lon+delta]."
        ),
        "city_center_adjustment": (
            "Original city centers were snapped to nearest valid grid point "
            "using a reference year."
        ),
        "snap_reference_year": int(snap_reference_year),
        "snap_valid_fraction_threshold": float(snap_valid_fraction_threshold),
        "box_size_delta_mapping": ", ".join(
            [f"{k}={v}" for k, v in box_size_deltas.items()]
        ),
        "Conventions": "CF-1.8",
    }

    return ds_out


def write_outputs(
    ds_out,
    output_dir,
    dataset_name,
    season_name,
    spatial_method,
    years_list,
    include_precipitation=True,
    write2csv=False,
    write2nc=False,
):
    """Write output Dataset to NetCDF and/or CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_stub = get_output_file_stub(
        dataset_name=dataset_name,
        season_name=season_name,
        spatial_method=spatial_method,
        years_list=years_list,
        include_precipitation=include_precipitation,
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
# 10) main script
# ---------------------------------------------------------------------
if __name__ == "__main__":

    years_list = get_year_list(years)
    seasons = get_season_list(season)

    if snap_reference_year is None:
        snap_reference_year = years_list[0]

    adjusted_city_coords = adjust_city_centers_to_valid_grid(
        dataset_name=dataset,
        city_coords=CITY_COORDS,
        reference_year=snap_reference_year,
        valid_fraction_threshold=snap_valid_fraction_threshold,
        include_precipitation=include_precipitation,
    )

    ds_out = combine_all_cities_and_boxes(
        dataset_name=dataset,
        city_coords=adjusted_city_coords,
        box_size_deltas=BOX_SIZE_DELTAS,
        years_list=years_list,
        seasons=seasons,
        spatial_method=spatial_method,
        include_precipitation=include_precipitation,
    )

    ds_out = add_city_and_box_coordinates(
        ds_out=ds_out,
        city_coords=adjusted_city_coords,
        box_size_deltas=BOX_SIZE_DELTAS,
    )

    ds_out = add_output_metadata(
        ds_out=ds_out,
        dataset_name=dataset,
        season_name=season,
        spatial_method=spatial_method,
        box_size_deltas=BOX_SIZE_DELTAS,
        snap_reference_year=snap_reference_year,
        snap_valid_fraction_threshold=snap_valid_fraction_threshold,
        include_precipitation=include_precipitation,
    )

    write_outputs(
        ds_out=ds_out,
        output_dir=output_dir,
        dataset_name=dataset,
        season_name=season,
        spatial_method=spatial_method,
        years_list=years_list,
        include_precipitation=include_precipitation,
        write2csv=write2csv,
        write2nc=write2nc,
    )
