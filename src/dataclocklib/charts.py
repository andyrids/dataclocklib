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
import json
import pathlib
from typing import Literal, Optional, Tuple, TypeAlias, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from pandas import DataFrame, MultiIndex, NamedAgg
from pandas.api.types import is_numeric_dtype

from dataclocklib.exceptions import (
    AggregationColumnError, AggregationFunctionError, ModeError
)
from dataclocklib.utility import add_text, assign_ring_wedge_columns

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

config_file = pathlib.Path(__file__).parent / "config" / "annotations.json"


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
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        ModeError: Unexpected mode value is passed.
        ValueError: Incompatible date_column dtype or empty DataFrame.

    Returns:
        A tuple containing a DataFrame with the aggregate values used to
        create the chart, the matplotlib chart Figure and Axes objects.
    """
    _validate_chart_parameters(data, date_column, agg_column, agg, mode)
 
    data = assign_ring_wedge_columns(data, date_column, mode)    

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

    annotations = json.loads(config_file.read_bytes()).get("dataclock", {})

    # chart title text
    add_text(
        ax=ax,
        text=chart_title,
        transform=fig.transFigure,
        **annotations["chart_title"]
    )
        
    # chart subtitle text
    add_text(
        ax=ax,
        text=chart_title,
        transform=fig.transFigure,
        **annotations["chart_subtitle"]
    )

    # chart reporting period text
    add_text(
        ax=ax,
        text=chart_period,
        transform=fig.transFigure,
        **annotations["chart_period"]
    )

    # chart source text
    add_text(
        ax=ax,
        text=chart_source,
        transform=fig.transFigure,
        **annotations["chart_source"]
    )

    return data_graph, fig, ax


def line_chart(
        data: DataFrame,
        date_column: str,
        agg_column: Optional[str] = None,
        agg: Aggregation = "count",
        mode: Mode = "DOW_HOUR",
        chart_title: Optional[str] = None,
        chart_subtitle: Optional[str] = None,
        chart_period: Optional[str] = None,
        chart_source: Optional[str] = None,
        default_text: bool = True,
    ) -> tuple[DataFrame, Figure, Axes]:
    """Create a temporal line chart from a pandas DataFrame.

    This function will divide a larger unit of time into rings and subdivide
    them by a smaller unit of time into wedges, creating temporal bins. The
    ring values will be represented as individual lines, with the aggregation
    values on the y-axis and wedges as the x-axis.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        agg (str): Aggregation function; 'count', 'mean', 'median',
            'mode' & 'sum'.
        agg_column (str, optional): DataFrame Column to aggregate.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_period (str, optional): Chart reporting period.
        chart_source (str, optional): Chart data source.
        default_text (bool, optional): Flag to generating default chart
            annotations for the chart_title ('Data Clock Chart') and
            chart_subtitle ('[agg] by [period] (rings) & [period] (wedges)').

    Raises:
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        ModeError: Unexpected mode value is passed.
        ValueError: Incompatible date_column dtype or empty DataFrame.

    Returns:
        A tuple containing a DataFrame with the aggregate values used to
        create the chart, the matplotlib and Axes objects.
    """
    _validate_chart_parameters(data, date_column, agg_column, agg, mode)

    data = assign_ring_wedge_columns(data, date_column, mode)    

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
        data_agg
        .set_index(index_names)
        .reindex(product_index)
        .reset_index(level="wedge")
    )

    # replace NaN values created for missing missing ring/wedge combinations
    data_graph = data_agg.fillna(0)

    # convert aggregate function results to int64, if possible
    if (data_graph[agg] % 1 == 0).all():
        data_graph[agg] = data_graph[agg].astype("int64")
    
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=96)

    # adjust subplots for custom title, subtitle and source text
    plt.subplots_adjust(
        left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None
    )

    # set white figure background
    fig.patch.set_facecolor("w")

    # create chart grid
    ax.grid(which="major", axis="x", color="#DAD8D7", alpha=0.5, zorder=1)
    ax.grid(which="major", axis="y", color="#DAD8D7", alpha=0.5, zorder=1)

    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)

    ax.xaxis.set_tick_params(
        which="both", pad=2, labelbottom=True, bottom=True, labelsize=12
    )

    n_wedges = data_graph["wedge"].nunique()
    unique_wedges = data_graph["wedge"].unique()
    if mode in ("DOW_HOUR", "DAY_HOUR"):
        xaxis_labels = map(lambda x: f"{x:02d}:00", unique_wedges)
        ax.set_xticks(range(n_wedges), xaxis_labels, rotation=45, ha="right")
    else:
        ax.set_xticks(range(n_wedges), unique_wedges, rotation=45, ha="right")
    ax.set_xlabel("", fontsize=12, labelpad=10)

    ax.set_ylabel(agg.title(), fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_major_formatter(lambda s, i: f"{s:,.0f}")
    ax.yaxis.set_tick_params(
        pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12
    )

    unique_indices = data_graph.index.unique()
    if mode == "DOW_HOUR":
        line_labels = dict(enumerate(calendar.day_name))
    else:
        line_labels = dict(zip(unique_indices, unique_indices))

    cmap = plt.get_cmap("tab10")

    for idx, i in enumerate(unique_indices):
        line_data = data_graph.loc[i]
        # ensure x is always numeric
        x = list(range(line_data["wedge"].size))

        ax.plot(
            x,
            line_data[agg],
            color=cmap(idx),
            label=line_labels[i],
            zorder=2
        )

        # custom style for final point
        ax.plot(
            x[-1],
            line_data[agg].iloc[-1],
            "o",
            color=cmap(idx),
            markersize=10,
            alpha=0.3
        )

        # custom style for final point
        ax.plot(
            x[-1],
            line_data[agg].iloc[-1],
            "o",
            color=cmap(idx),
            markersize=5,
        )

    # add legend
    ax.legend(loc="best", fontsize=12)

    # generate default text for missing chart_title & chart_subtitle values
    if default_text:
        if chart_title is None:
            chart_title = "Line Chart"

    chart_subtitle_map = {
        "YEAR_MONTH": "year & month",
        "YEAR_WEEK": "year & week of year",
        "WEEK_DAY": "week of year & day of week",
        "DOW_HOUR": "day of week & hour of day",
        "DAY_HOUR": "day of year & hour of day",
    }
    if chart_subtitle is None:
        chart_subtitle = f"{agg.title()} by {chart_subtitle_map[mode]}"

    annotations = json.loads(config_file.read_bytes()).get("line_chart", {})

    # chart title text
    add_text(
        ax=ax,
        text=chart_title,
        transform=fig.transFigure,
        **annotations["chart_title"]
    )
        
    # chart subtitle text
    add_text(
        ax=ax,
        text=chart_subtitle,
        transform=fig.transFigure,
        **annotations["chart_subtitle"]
    )

    # chart reporting period text
    add_text(
        ax=ax,
        text=chart_period,
        transform=fig.transFigure,
        **annotations["chart_period"]
    )

    # chart source text
    add_text(
        ax=ax,
        text=chart_source,
        transform=fig.transFigure,
        **annotations["chart_source"]
    )

    return data_graph, fig, ax


def _validate_chart_parameters(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: Aggregation = "count",
    mode: str = "DAY_HOUR",
    )  -> None:
    """Validate chart parameters.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        agg (str): Aggregation function; 'count', 'mean', 'median',
            'mode' & 'sum'.
        agg_column (str, optional): DataFrame Column to aggregate.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.

    Raises:
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        ModeError: Unexpected mode value is passed.
        ValueError: Incompatible date_column dtype or empty DataFrame.

    Returns:
        None
    """
    if data.empty:
        raise ValueError(f"DataFrame is empty.")
    if data[date_column].dtype.name != "datetime64[ns]":
        raise ValueError(f"date_column dtype is not datetime64[ns].")
    if mode not in VALID_MODES:
        raise ModeError(mode, VALID_MODES)
    if agg not in VALID_AGGREGATIONS:
        raise AggregationFunctionError(agg, VALID_AGGREGATIONS)
    if agg_column is None and agg != "count":
        raise AggregationColumnError(agg)