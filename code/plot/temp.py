"""Plot the linear trend in gridded zero-degree crossing statistics.

The input is the full-grid NetCDF produced by the grid-point statistics script.
A linear trend is fitted independently at every latitude-longitude grid point
over all available years in the selected period.

Yellow dots mark major Scandinavian cities.
"""

import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from trygzerodegreedayscities import config


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

dataset = "eobs"
plot_with_precipitation = False

season = "djf"                    # "djf", "mam", "jja", or "son"
statistic = "zdc_pct"             # "zdc_pct", "zdc_days", or "n_valid_days"

file_years = [1951, 2024]
file_season_tag = "djf"           # season used when creating the statistics file

input_dir = config.dirs["eobs_processed"]
output_dir = config.dirs["fig"] + "oppgave_26-04-27/"

savefig = False
showfig = True
fig_dpi = 200
figsize = (11, 9)

# None uses the full latitude-longitude domain in the input file.
# Example Scandinavian extent: [4.0, 21.0, 54.0, 71.5]
map_extent = [4.0, 30.0, 54.0, 71.5]

label_cities = True
city_marker_size = 45

# Require at least this many finite annual values to calculate a trend.
min_valid_years = 2

CITY_COORDS = {
    "Oslo": (59.9139, 10.7522),
    "Bergen": (60.3913, 5.3221),
    "Trondheim": (63.4305, 10.3951),
    "Copenhagen": (55.6761, 12.5683),
    "Aarhus": (56.1629, 10.2039),
    "Odense": (55.4038, 10.4024),
    "Stockholm": (59.3293, 18.0686),
    "Gothenburg": (57.7089, 11.9746),
    "Malmo": (55.6050, 13.0038),
    "Tromso": (69.6492, 18.9553),
}


# -----------------------------------------------------------------------------
# Filenames and labels
# -----------------------------------------------------------------------------

def get_precip_tag(with_precipitation):
    """Return the precipitation tag used by the statistics script."""
    return "with_precipitation" if with_precipitation else "without_precipitation"


def build_input_file(
    dataset_name, file_season, years, with_precipitation
):
    """Build the gridded-statistics NetCDF filename."""
    precip_tag = get_precip_tag(with_precipitation)
    return (
        f"xy_zero_degree_crossing_{precip_tag}_stats_grid_"
        f"{dataset_name}_{file_season}_{years[0]}-{years[1]}.nc"
    )


def build_figure_name(
    dataset_name, season_name, statistic_name, years, with_precipitation
):
    """Build the trend-map figure filename."""
    precip_tag = get_precip_tag(with_precipitation)
    return (
        f"map_linear_trend_{statistic_name}_{precip_tag}_"
        f"{dataset_name}_{season_name}_{years[0]}-{years[1]}.png"
    )


def get_statistic_label(statistic_name):
    """Return map and colorbar labels for the selected statistic."""
    labels = {
        "zdc_pct": (
            "Zero-degree crossing percentage",
            "Trend (%-points decade$^{-1}$)",
        ),
        "zdc_days": (
            "Zero-degree crossing days",
            "Trend (days decade$^{-1}$)",
        ),
        "n_valid_days": (
            "Number of valid days",
            "Trend (days decade$^{-1}$)",
        ),
    }
    if statistic_name not in labels:
        raise ValueError(
            f"statistic must be one of {list(labels)}; got {statistic_name!r}"
        )
    return labels[statistic_name]


input_file = build_input_file(
    dataset_name=dataset,
    file_season=file_season_tag,
    years=file_years,
    with_precipitation=plot_with_precipitation,
)
fig_name = build_figure_name(
    dataset_name=dataset,
    season_name=season,
    statistic_name=statistic,
    years=file_years,
    with_precipitation=plot_with_precipitation,
)


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------

def open_statistics(path_nc):
    """Open the full-grid statistics dataset."""
    if not os.path.exists(path_nc):
        raise FileNotFoundError(f"Statistics file not found: {path_nc}")
    return xr.open_dataset(path_nc, decode_timedelta=False)


def select_statistic(ds, statistic_name, season_name):
    """Select one statistic and season while retaining year, latitude, longitude."""
    if statistic_name not in ds:
        raise ValueError(
            f"Variable {statistic_name!r} not found. Available variables: {list(ds.data_vars)}"
        )

    required_dims = {"year", "latitude", "longitude"}
    missing_dims = required_dims - set(ds[statistic_name].dims)
    if missing_dims:
        raise ValueError(
            f"{statistic_name!r} is missing dimensions {sorted(missing_dims)}. "
            f"Found dimensions: {ds[statistic_name].dims}"
        )

    if "season" not in ds.coords:
        raise ValueError("Input dataset does not contain a 'season' coordinate.")

    available_seasons = [str(value) for value in ds["season"].values]
    if season_name not in available_seasons:
        raise ValueError(
            f"Season {season_name!r} not found. Available seasons: {available_seasons}"
        )

    return ds[statistic_name].sel(season=season_name)


def calculate_linear_trend(data, min_years=2):
    """Fit a least-squares trend along year at every grid point."""
    valid_years = data.count("year")
    centered_year = data["year"].astype(float) - float(data["year"].mean())

    fit_data = data.assign_coords(year=centered_year)
    coefficients = fit_data.polyfit(dim="year", deg=1, skipna=True)["polyfit_coefficients"]
    slope_per_year = coefficients.sel(degree=1)

    trend_per_decade = 10.0 * slope_per_year
    return trend_per_decade.where(valid_years >= min_years)


def get_symmetric_limits(trend):
    """Return robust symmetric color limits centered on zero."""
    finite = np.asarray(trend.values)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        raise ValueError("No finite trend values are available to plot.")

    limit = float(np.nanpercentile(np.abs(finite), 98))
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        limit = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        limit = 1.0

    return -limit, limit


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def add_city_markers(ax, label_names=True):
    """Overlay major Scandinavian cities as yellow dots."""
    for city_name, (lat, lon) in CITY_COORDS.items():
        ax.scatter(
            lon,
            lat,
            s=city_marker_size,
            c="yellow",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            transform=ccrs.PlateCarree(),
        )

        if label_names:
            ax.text(
                lon + 0.15,
                lat + 0.08,
                city_name,
                fontsize=8,
                zorder=6,
                transform=ccrs.PlateCarree(),
            )


def plot_trend_map(trend, statistic_name, season_name, years):
    """Plot the gridded linear trend on a latitude-longitude map."""
    statistic_label, colorbar_label = get_statistic_label(statistic_name)
    vmin, vmax = get_symmetric_limits(trend)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        trend["longitude"],
        trend["latitude"],
        trend,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, facecolor="none", edgecolor="black", linewidth=0.4)

    if map_extent is None:
        ax.set_extent(
            [
                float(trend["longitude"].min()),
                float(trend["longitude"].max()),
                float(trend["latitude"].min()),
                float(trend["latitude"].max()),
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    gridlines = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False

    add_city_markers(ax, label_names=label_cities)

    precip_text = " with precipitation" if plot_with_precipitation else ""
    ax.set_title(
        f"Linear trend in {statistic_label.lower()}{precip_text}\n"
        f"{season_name.upper()}, {years[0]}-{years[1]}"
    )

    colorbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.03, shrink=0.85)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    return fig


def save_figure(fig, directory, filename, dpi):
    """Save the figure."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote figure: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    path_nc = os.path.join(input_dir, input_file)

    print(f"Reading: {path_nc}")
    print(f"Statistic: {statistic}")
    print(f"Season: {season}")
    print(f"Figure name: {fig_name}")

    with open_statistics(path_nc) as ds:
        data = select_statistic(ds, statistic, season)
        trend = calculate_linear_trend(data, min_valid_years).load()

    fig = plot_trend_map(
        trend=trend,
        statistic_name=statistic,
        season_name=season,
        years=file_years,
    )

    if savefig:
        save_figure(fig, output_dir, fig_name, fig_dpi)

    if showfig:
        plt.show()
    else:
        plt.close(fig)
