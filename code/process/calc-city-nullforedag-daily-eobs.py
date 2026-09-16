"""Write daily zero-degree crossing values for Scandinavian cities.

Outputs:
    NetCDF: nullforedag(time, city, box_size_index), without seasonal statistics.
    CSV: one row per date, city, and area size, containing the same daily value.

For gridpoint_mean, nullforedag is the fraction of all selected grid cells meeting
tn < 0 and tx > 0 (and tp > 0 when enabled). Invalid cells contribute zero,
matching the original script's spatial denominator. A day with no valid cells is
missing. For city_mean, the criteria are applied to daily spatial means, giving
0 or 1. A single-grid-point box gives 0, 1, or missing with either method.

Temperatures must already be in degrees Celsius. Precipitation must use units
where positive values indicate precipitation. ERA5 paths remain placeholders.

The requested years are full calendar years, from January 1 of the first year
through December 31 of the last year, including leap days. No season coordinates
or seasonal filtering are used.

Requires numpy, pandas, xarray, a NetCDF backend (netCDF4 or scipy), and the
existing trygzerodegreedayscities configuration module.
"""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from trygzerodegreedayscities import config

# User settings-------------------------------------------------------------------
dataset = "eobs"                         # "eobs" or "era5"
years = [1950, 2024]                     # Inclusive calendar-year range
include_precipitation = True             # Also require tp > 0
spatial_method = "gridpoint_mean"        # "gridpoint_mean" or "city_mean"
write2csv = True
write2nc = True
output_dir = config.dirs["eobs_processed"]
snap_reference_year = None               # None uses the first requested year
snap_valid_fraction_threshold = 0.95

CITY_COORDS = {
    "Oslo": {"lat": 59.9139, "lon": 10.7522},
    "Bergen": {"lat": 60.3913, "lon": 5.3221},
    "Trondheim": {"lat": 63.4305, "lon": 10.3951},
    "Copenhagen": {"lat": 55.6761, "lon": 12.5683},
    "Aarhus": {"lat": 56.1629, "lon": 10.2039},
    "Odense": {"lat": 55.4038, "lon": 10.4024},
    "Stockholm": {"lat": 59.3293, "lon": 18.0686},
    "Gothenburg": {"lat": 57.7089, "lon": 11.9746},
    "Malmo": {"lat": 55.6050, "lon": 13.0038},
    "Tromso": {"lat": 69.6492, "lon": 18.9553},
}

# Half-width in latitude/longitude degrees; small selects one grid point.
BOX_SIZE_DELTAS = {"small": 0.0, "medium": 0.1, "large": 0.2}

DATASET_CONFIG = {
    "eobs": {"raw_dir": config.dirs["eobs_raw"], "resolution": "0.1x0.1"},
    "era5": {"raw_dir": config.dirs.get("era5_raw", ""), "resolution": "0.25x0.25"},
}
# --------------------------------------------------------------------------------

for settings in DATASET_CONFIG.values():
    settings.update(tn_var="tn", tx_var="tx", tp_var="tp",
                    lat_name="latitude", lon_name="longitude")


def get_year_list(years_in):
    """Return the inclusive range of requested calendar years."""
    if len(years_in) != 2:
        raise ValueError("years must be [start_year, end_year].")
    start, end = map(int, years_in)
    if end < start:
        raise ValueError("end_year must be >= start_year.")
    return list(range(start, end + 1))


def build_file_path(dataset_name, variable, year):
    settings = DATASET_CONFIG[dataset_name]
    name = f"{variable}_{settings['resolution']}_{year}.nc"
    path = Path(settings["raw_dir"]) / variable / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def match_reference_grid(data, reference, lat_name, lon_name):
    """Select reference cells and remove small coordinate rounding differences."""
    coordinates = {name: reference[name] for name in (lat_name, lon_name)}
    data = data.sel(coordinates, method="nearest")
    return data.assign_coords(coordinates)


def subset_latlon(data, lat0, lon0, delta, lat_name, lon_name):
    if np.isclose(delta, 0):
        return data.sel({lat_name: [lat0], lon_name: [lon0]}, method="nearest")
    selection = {}
    for name, center in ((lat_name, lat0), (lon_name, lon0)):
        lower, upper = center - delta, center + delta
        selection[name] = slice(lower, upper) if data[name][0] < data[name][-1] else slice(upper, lower)
    result = data.sel(selection)
    if not result.sizes[lat_name] or not result.sizes[lon_name]:
        raise ValueError(f"No grid cells around ({lat0}, {lon0}), delta={delta}.")
    return result


def adjust_city_centers_to_valid_grid(dataset_name, city_coords, reference_year, threshold):
    """Retain the original temperature-only validity criterion for city snapping."""
    settings = DATASET_CONFIG[dataset_name]
    lat_name, lon_name = settings["lat_name"], settings["lon_name"]
    with ExitStack() as stack:
        fields = {}
        for variable in ("tn", "tx"):
            source = stack.enter_context(xr.open_dataset(
                build_file_path(dataset_name, variable, reference_year)
            ))
            fields[variable] = source[settings[f"{variable}_var"]]
        tn = fields["tn"]
        tx = match_reference_grid(fields["tx"], tn, lat_name, lon_name)
        tn, tx = xr.align(tn, tx, join="exact")
        valid_fraction = (np.isfinite(tn) & np.isfinite(tx)).mean("time")
        valid_grid = (valid_fraction > threshold).transpose(lat_name, lon_name)
        lat_indices, lon_indices = np.where(valid_grid.values)
        if not len(lat_indices):
            raise ValueError(f"No valid grid cells found with threshold={threshold}.")
        latitudes = valid_grid[lat_name].values[lat_indices]
        longitudes = valid_grid[lon_name].values[lon_indices]
    adjusted = {}
    for city, coord in city_coords.items():
        distance = (latitudes - coord["lat"]) ** 2 + (longitudes - coord["lon"]) ** 2
        index = distance.argmin()
        adjusted[city] = {
            "lat": float(latitudes[index]), "lon": float(longitudes[index]),
            "orig_lat": coord["lat"], "orig_lon": coord["lon"],
        }
        print(f"{city:10s}: ({coord['lat']:.4f}, {coord['lon']:.4f}) -> "
              f"({latitudes[index]:.4f}, {longitudes[index]:.4f})")
    return adjusted


def open_weather_for_box(dataset_name, year, lat0, lon0, delta, include_precipitation):
    """Load one year's box while all source files are still open."""
    settings = DATASET_CONFIG[dataset_name]
    lat_name, lon_name = settings["lat_name"], settings["lon_name"]
    variables = ["tn", "tx", "tp"] if include_precipitation else ["tn", "tx"]
    fields = {}
    with ExitStack() as stack:
        for variable in variables:
            source = stack.enter_context(xr.open_dataset(
                build_file_path(dataset_name, variable, year)
            ))
            data = source[settings[f"{variable}_var"]]
            if variable == "tn":
                data = subset_latlon(data, lat0, lon0, delta, lat_name, lon_name)
            else:
                data = match_reference_grid(data, fields["tn"], lat_name, lon_name)
            fields[variable] = data
        aligned = xr.align(*fields.values(), join="exact")
        return xr.Dataset(dict(zip(fields, aligned))).load()


def compute_daily_nullforedag(data, method, include_precipitation, lat_name, lon_name):
    """Calculate the daily spatial fraction or flag without temporal aggregation."""
    spatial_dims = [lat_name, lon_name]
    if method == "city_mean":
        data = data.mean(spatial_dims, skipna=True)
    elif method != "gridpoint_mean":
        raise ValueError("spatial_method must be 'gridpoint_mean' or 'city_mean'.")

    valid = np.isfinite(data.tn) & np.isfinite(data.tx)
    event = (data.tn < 0) & (data.tx > 0)
    if include_precipitation:
        valid = valid & np.isfinite(data.tp)
        event = event & (data.tp > 0)

    if method == "gridpoint_mean":
        # Preserve the original denominator: all cells, including invalid cells.
        value = (event & valid).astype(float).mean(spatial_dims)
        value = value.where(valid.any(spatial_dims))
    else:
        value = event.astype(float).where(valid)
    return value.rename("nullforedag")


def compute_daily_values(dataset_name, city_coords, box_sizes, years_list,
                         method, include_precipitation):
    settings = DATASET_CONFIG[dataset_name]
    lat_name, lon_name = settings["lat_name"], settings["lon_name"]
    cities = []
    for city, coord in city_coords.items():
        boxes = []
        for box, delta in box_sizes.items():
            print(f"Processing {city:10s} | box={box:6s} | method={method}")
            daily = []
            for year in years_list:
                weather = open_weather_for_box(
                    dataset_name, year, coord["lat"], coord["lon"], delta,
                    include_precipitation,
                )
                # Restore absent dates as missing; never label them as non-events.
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                index = weather.indexes["time"]
                if index.has_duplicates or not index.isin(dates).all():
                    raise ValueError(f"Expected unique daily midnight timestamps for {year}.")
                weather = weather.reindex(time=dates)
                daily.append(compute_daily_nullforedag(
                    weather, method, include_precipitation, lat_name, lon_name
                ))
            box_values = xr.concat(daily, dim="time").sortby("time")
            boxes.append(box_values.expand_dims(box_size_index=[box]))
        cities.append(xr.concat(boxes, dim="box_size_index", join="exact").expand_dims(city=[city]))
    values = xr.concat(cities, dim="city", join="exact")
    return values.transpose("time", "city", "box_size_index").to_dataset()


def add_metadata(result, city_coords, box_sizes, reference_year):
    cities = result.city.values
    result = result.assign_coords(
        city_lat=("city", [city_coords[name]["lat"] for name in cities]),
        city_lon=("city", [city_coords[name]["lon"] for name in cities]),
        city_orig_lat=("city", [city_coords[name]["orig_lat"] for name in cities]),
        city_orig_lon=("city", [city_coords[name]["orig_lon"] for name in cities]),
        box_size_delta=("box_size_index", [box_sizes[name] for name in result.box_size_index.values]),
    )
    definition = "tn < 0 degC and tx > 0 degC"
    if include_precipitation:
        definition += " and tp > 0"
    result.nullforedag.attrs = {
        "long_name": "Daily zero-degree crossing value", "units": "1",
        "valid_min": 0.0, "valid_max": 1.0, "event_definition": definition,
        "description": (
            "gridpoint_mean: fraction of all selected grid cells satisfying the criteria; "
            "invalid cells contribute zero, but days with no valid cells are missing. "
            "city_mean: 0 or 1 from spatially averaged weather; missing if unavailable."
        ),
    }
    for name in ("city_lat", "city_orig_lat"):
        result[name].attrs["units"] = "degrees_north"
    for name in ("city_lon", "city_orig_lon"):
        result[name].attrs["units"] = "degrees_east"
    result.box_size_delta.attrs = {
        "long_name": "Latitude-longitude box half-width", "units": "degrees",
        "description": "Zero selects a single grid point; boxes use the adjusted city center.",
    }
    result.box_size_index.attrs["long_name"] = "Area size category"
    result.attrs = {
        "title": "Daily zero-degree crossing values for Scandinavian cities",
        "dataset": dataset, "spatial_method": spatial_method,
        "event_definition": definition, "include_precipitation": int(include_precipitation),
        "year_start": int(years[0]),
        "year_end": int(years[1]), "snap_reference_year": int(reference_year),
        "snap_valid_fraction_threshold": float(snap_valid_fraction_threshold),
        "city_center_adjustment": "Nearest cell with temperature validity above threshold.",
        "Conventions": "CF-1.8",
    }
    return result


def write_outputs(result):
    """Write the daily array and a flat CSV containing exactly the same values."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    precipitation = "with_precipitation" if include_precipitation else "without_precipitation"
    stem = (f"scandinavian_city_nullforedag_{precipitation}_{dataset}_"
            f"{spatial_method}_{years[0]}-{years[1]}")
    if write2nc:
        path = directory / f"{stem}.nc"
        result.to_netcdf(path, encoding={"nullforedag": {"dtype": "float64", "_FillValue": np.nan}})
        print(f"Wrote NetCDF: {path}")
    if write2csv:
        path = directory / f"{stem}.csv"
        table = result.nullforedag.to_dataframe().reset_index()
        table = table[["time", "city", "box_size_index", "nullforedag"]].rename(columns={
            "time": "Date", "city": "City", "box_size_index": "Area size",
            "nullforedag": "Zero-crossing value (0-1)",
        })
        table["Area size"] = table["Area size"].str.capitalize()
        table.to_csv(path, index=False, date_format="%Y-%m-%d", na_rep="NA")
        print(f"Wrote CSV: {path}")


def main():
    if dataset not in DATASET_CONFIG:
        raise ValueError("dataset must be 'eobs' or 'era5'.")
    if spatial_method not in ("gridpoint_mean", "city_mean"):
        raise ValueError("spatial_method must be 'gridpoint_mean' or 'city_mean'.")
    if not 0 <= snap_valid_fraction_threshold < 1:
        raise ValueError("snap_valid_fraction_threshold must be >= 0 and < 1.")
    years_list = get_year_list(years)
    reference_year = years_list[0] if snap_reference_year is None else snap_reference_year
    adjusted_cities = adjust_city_centers_to_valid_grid(
        dataset, CITY_COORDS, reference_year, snap_valid_fraction_threshold
    )
    result = compute_daily_values(
        dataset, adjusted_cities, BOX_SIZE_DELTAS, years_list,
        spatial_method, include_precipitation,
    )
    result = add_metadata(result, adjusted_cities, BOX_SIZE_DELTAS, reference_year)
    write_outputs(result)


if __name__ == "__main__":
    main()
