"""Data clock module for chart creation.

Author: Andrew Ridyard.

License: GNU General Public License v3 or later.

Copyright (C): 2025.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Functions:
    dataclock: Create a data clock chart from a pandas DataFrame.

Types:
    Aggregation: Keys representing aggregation functions.
    Mode: Keys representing temporal bins used in each chart.
"""

from __future__ import annotations

import calendar
from collections import defaultdict
from typing import Literal, Optional, Tuple, TypeAlias, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from pandas import DataFrame, MultiIndex, NamedAgg

from dataclocklib.exceptions import AggregationError, ModeError
from dataclocklib.utility import add_text

ColourMap: TypeAlias = Literal[
    "RdYlGn_r", "CMRmap_r", "inferno_r", "YlGnBu_r", "viridis"
]
VALID_CMAPS: Tuple[ColourMap, ...] = get_args(ColourMap)

Mode: TypeAlias = Literal[
    "YEAR_MONTH", "YEAR_WEEK", "WEEK_DAY", "DOW_HOUR", "DAY_HOUR"
]
VALID_MODES: Tuple[Mode, ...] = get_args(Mode)

Aggregation: TypeAlias = Literal[
    "count", "max", "mean", "median", "min", "sum"
]
VALID_AGGREGATIONS: Tuple[Aggregation, ...] = get_args(Aggregation)


def dataclock(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: Aggregation = "count",
    mode: Mode = "DAY_HOUR",
    cmap_name: ColourMap = "RdYlGn_r",
    chart_title: Optional[str] = None,
    chart_subtitle: Optional[str] = None,
    chart_period: Optional[str] = None,
    chart_source: Optional[str] = None,
    default_text: bool = True,
) -> tuple[DataFrame, Figure, Axes]:
    """Create a data clock chart from a pandas DataFrame.

    Data clocks visually summarise temporal data in two dimensions,
    revealing seasonal or cyclical patterns and trends over time.
    A data clock is a circular chart that divides a larger unit of
    time into rings and subdivides it by a smaller unit of time into
    wedges, creating a set of temporal bins.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        agg (str): Aggregation function; 'count', 'mean', 'median',
            'mode' & 'sum'.
        agg_column (str, optional): DataFrame Column to aggregate.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.
        cmap_name: (ColourMap, optional): Matplotlib colormap name used
            to symbolise the temporal bins; 'RdYlGn_r', 'CMRmap_r',
            'inferno_r', 'YlGnBu_r' & 'viridis'.
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_period (str, optional): Chart reporting period.
        chart_source (str, optional): Chart data source.
        default_text (bool, optional): Flag to generating default chart
            annotations for the chart_title ('Data Clock Chart') and
            chart_subtitle ('[agg] by [period] (rings) & [period] (wedges)').

    Raises:
        AggregationError: Incompatible agg_column dtype & agg combination.
        ModeError: Unexpected mode value is passed.
        ValueError: Incompatible date_column dtype or empty DataFrame.

    Returns:
        A tuple containing a DataFrame with the aggregate values used to
        create the chart, the matplotlib chart Figure and Axes objects.
    """
    if data.empty:
        raise ValueError(f"DataFrame is empty.")
    if data[date_column].dtype.name != "datetime64[ns]":
        raise ValueError(f"date_column dtype is not datetime64[ns].")
    if mode not in VALID_MODES:
        raise ModeError(mode, mode_map.keys())
    if agg not in VALID_AGGREGATIONS:
        raise AggregationError(agg, agg_column)
    if agg_column is None and agg != "count":
        raise AggregationError(agg, agg_column)

    # dict map for ring & wedge features based on mode
    mode_map = defaultdict(dict)
    # year | January - December
    if mode == "YEAR_MONTH":
        mode_map[mode]["ring"] = data[date_column].dt.year
        mode_map[mode]["wedge"] = data[date_column].dt.month_name()
    # year | weeks 1 - 52
    if mode == "YEAR_WEEK":
        mode_map[mode]["ring"] = data[date_column].dt.year
        week = data[date_column].dt.isocalendar().week
        week[week == 53] = 52
        mode_map[mode]["wedge"] = week
    # weeks 1 - 52 | Monday - Sunday
    if mode == "WEEK_DAY":
        week = data[date_column].dt.isocalendar().week
        year = data[date_column].dt.year
        mode_map[mode]["ring"] = week + year * 100
        mode_map[mode]["wedge"] = data[date_column].dt.day_name()
    # days 1 - 7 (Monday - Sunday) | 00:00 - 23:00
    if mode == "DOW_HOUR":
        mode_map[mode]["ring"] = data[date_column].dt.day_of_week
        mode_map[mode]["wedge"] = data[date_column].dt.hour
    # days 1 - 365 | 00:00 - 23:00
    if mode == "DAY_HOUR":
        mode_map[mode]["ring"] = data[date_column].dt.strftime("%Y%j")
        mode_map[mode]["wedge"] = data[date_column].dt.hour

    data = data.assign(**mode_map[mode]).astype({"ring": "int64"})

    # dict map for wedge min & max range based on mode
    wedge_range_map = {
        "YEAR_MONTH": tuple(calendar.month_name[1:]),
        "YEAR_WEEK": range(1, 53),
        "WEEK_DAY": tuple(calendar.day_name),
        "DOW_HOUR": range(0, 24),
        "DAY_HOUR": range(0, 24),
    }

    index_names = ["ring", "wedge"]
    agg_column = agg_column or date_column

    data_grouped = data.groupby(index_names, as_index=False)
    data_agg = data_grouped.agg(**{agg: NamedAgg(agg_column, agg)})

    # index with all possible combinations of ring & wedge values
    product_index = MultiIndex.from_product(
        [data_agg["ring"].unique(), wedge_range_map[mode]], names=index_names
    )

    # populate any rows for missing ring/wedge combinations
    data_agg = (
        data_agg.set_index(index_names).reindex(product_index).reset_index()
    )

    # replace NaN values created for missing missing ring/wedge combinations
    data_graph = data_agg.fillna(0)

    # convert aggregate function results to int64, if possible
    if (data_graph[agg] % 1 == 0).all():
        data_graph[agg] = data_graph[agg].astype("int64")

    # create figure with polar projection
    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"}, figsize=(11, 10), dpi=96
    )

    # adjust subplots for custom title, subtitle and source text
    plt.subplots_adjust(
        left=None, bottom=0.2, right=None, top=0.9, wspace=None, hspace=None
    )

    # set white figure background
    fig.patch.set_facecolor("w")

    # set clockwise direction starting from North
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    n_wedges = data_graph["wedge"].nunique()

    # calculate angles for each wedge
    theta = np.linspace(0, 2 * np.pi, n_wedges, endpoint=False)

    # width of each bar (radians)
    width = 2 * np.pi / n_wedges

    unique_rings = data_graph["ring"].unique()

    max_radius = unique_rings.size + 1
    ax.set_rorigin(-1)
    ax.set_rlim(1, max_radius)

    # create x-axis labels
    if mode not in ("DOW_HOUR", "DAY_HOUR"):
        xaxis_labels = wedge_range_map[mode]
    # custom x-axis labels for hour of day (00:00 - 23:00)
    else:
        xaxis_labels = [f"{x:02d}:00" for x in range(24)]
    # set x-axis ticks
    ax.xaxis.set_ticks(theta)
    ax.xaxis.set_ticklabels([])

    ax.yaxis.set_ticks(range(1, max_radius))
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis="x", pad=5)

    ax.xaxis.grid(visible=True, color="black", alpha=0.8)
    ax.yaxis.grid(visible=True, color="black", alpha=0.8)

    ax.spines["polar"].set_visible(True)

    agg_max = data_graph[agg].max()

    # minimum, 25%, 50%, 75%, maximum
    colorbar_dtype = (np.float64, np.int64)[agg in ("count", "sum")]
    colourbar_ticks = np.linspace(1, agg_max, 5, dtype=colorbar_dtype)

    cmap = colormaps[cmap_name]
    cmap.set_under("w")
    cmap_norm = Normalize(1, agg_max)

    colorbar = fig.colorbar(
        ScalarMappable(norm=cmap_norm, cmap=cmap),
        ax=ax,
        orientation="vertical",
        ticks=colourbar_ticks,
        shrink=0.6,
        pad=0.2,
        extend="min",
    )

    colorbar.ax.tick_params(direction="out")

    # set text label y position based on number of rings + 1 (max_radius)
    text_y = max_radius + 0.7 if max_radius > 3 else max_radius + 0.2

    for idx, angle in enumerate(theta):
        # convert to degrees for text rotation
        angle_deg = np.rad2deg(angle)

        if 0 <= angle_deg < 90:
            rotation = -angle_deg
        elif 270 <= angle_deg <= 360:
            rotation = -angle_deg
        else:
            rotation = 180 - angle_deg

        ax.text(
            angle,
            text_y,
            xaxis_labels[idx],
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
        )

    # ring position starts from 1, creating a donut shape
    start_position = 1

    for ring_position, ring in enumerate(unique_rings):
        view = data_graph.loc[data_graph["ring"] == ring]
        graduated_colors = tuple(map(lambda x: cmap(cmap_norm(x)), view[agg]))

        ax.bar(
            # wedges/angles
            theta,
            # height
            1,
            # bars aligned to wedge
            align="edge",
            # width in radians
            width=width,
            # ring to place bar
            bottom=start_position + ring_position,
            # transparency
            alpha=0.8,
            # color map
            color=graduated_colors,
        )

    # generate default text for missing chart_title & chart_subtitle values
    if default_text:
        if chart_title is None:
            chart_title = "Data Clock Chart"

    chart_subtitle_map = {
        "YEAR_MONTH": "year (rings) & month (wedges)",
        "YEAR_WEEK": "year (rings) & week of year (wedges)",
        "WEEK_DAY": "week of year (rings) & day of week (wedges)",
        "DOW_HOUR": "day of week (rings) & hour of day (wedges)",
        "DAY_HOUR": "day of year (rings) & hour of day (wedges)",
    }
    if chart_subtitle is None:
        chart_subtitle = f"{agg.title()} by {chart_subtitle_map[mode]}"

    # chart title text
    add_text(
        ax=ax,
        x=0.12,
        y=0.93,
        text=chart_title,
        fontsize=14,
        weight="bold",
        alpha=0.8,
        ha="left",
        transform=fig.transFigure,
    )

    # chart subtitle text
    add_text(
        ax=ax,
        x=0.12,
        y=0.90,
        text=chart_subtitle,
        fontsize=12,
        alpha=0.8,
        ha="left",
        transform=fig.transFigure,
    )

    # chart reporting period text
    add_text(
        ax=ax,
        x=0.12,
        y=0.87,
        text=chart_period,
        fontsize=10,
        alpha=0.7,
        ha="left",
        transform=fig.transFigure,
    )

    # chart source text
    add_text(
        ax=ax,
        x=0.1,
        y=0.15,
        text=chart_source,
        fontsize=10,
        alpha=0.7,
        ha="left",
        transform=fig.transFigure,
    )

    return data_graph, fig, ax
