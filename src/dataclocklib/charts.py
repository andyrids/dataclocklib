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
    line_chart: Create a line chart from a pandas DataFrame.

Constants:
    VALID_AGGREGATIONS: Tuple of valid aggregation function names.
    VALID_CMAPS: Tuple of valid colour map names.
    VALID_MODES: Tuple of valid chart modes.
"""

from __future__ import annotations

import calendar
import configparser
import pathlib
from typing import Optional, Tuple, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from dataclocklib.exceptions import (
    AggregationColumnError,
    AggregationFunctionError,
    EmptyDataFrameError,
    MissingDatetimeError,
    ModeError,
)
from dataclocklib.typing import Aggregation, CmapNames, Mode
from dataclocklib.utility import (
    add_colorbar,
    add_text,
    add_wedge_labels,
    aggregate_temporal_columns,
    assign_temporal_columns,
    get_figure_dimensions,
)

VALID_AGGREGATIONS: Tuple[Aggregation, ...] = get_args(Aggregation)
VALID_CMAPS: Tuple[CmapNames, ...] = get_args(CmapNames)
VALID_MODES: Tuple[Mode, ...] = get_args(Mode)

# config files for default title and subtitle text
dataclock_ini = pathlib.Path(__file__).parent / "config" / "dataclock.ini"
linechart_ini = pathlib.Path(__file__).parent / "config" / "linechart.ini"

config = configparser.ConfigParser()


def dataclock(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: Aggregation = "count",
    mode: Mode = "DAY_HOUR",
    cmap_name: str = "RdYlGn_r",
    cmap_reverse: bool = False,
    spine_color: str = "darkslategrey",
    grid_color: str = "darkslategrey",
    default_text: bool = True,
    *,  # keyword only arguments
    chart_title: Optional[str] = None,
    chart_subtitle: Optional[str] = None,
    chart_period: Optional[str] = None,
    chart_source: Optional[str] = None,
    **fig_kw,
) -> tuple[DataFrame, Figure, Axes]:
    """Create a data clock chart from a pandas DataFrame.

    Data clocks visually summarise temporal data in two dimensions,
    revealing seasonal or cyclical patterns and trends over time.
    A data clock is a circular chart that divides a larger unit of
    time into rings and subdivides it by a smaller unit of time into
    wedges, creating a set of temporal bins.

    TIP: Palettes - https://python-graph-gallery.com/color-palette-finder/

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        agg (str): Aggregation function; 'count', 'mean', 'median',
            'mode' & 'sum'.
        agg_column (str, optional): DataFrame Column to aggregate.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.
        cmap_name: (str, optional): Name of a matplotlib/PyPalettes colormap,
            to symbolise the temporal bins; 'RdYlGn_r', 'CMRmap_r',
            'inferno_r', 'Alkalay2', 'viridis', 'a_palette' etc.
        cmap_reverse (bool): Reverse cmap colors flag.
        spine_color (str): Name of color to style the polar axis spines.
        default_text (bool, optional): Flag to generating default chart
            annotations for the chart_title ('Data Clock Chart') and
            chart_subtitle ('[agg] by [period] (rings) & [period] (wedges)').
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_period (str, optional): Chart reporting period.
        chart_source (str, optional): Chart data source.
        fig_kw (dict): Chart figure kwargs passed to pyplot.subplots.

    Raises:
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        EmptyDataFrameError: Unexpected empty DataFrame.
        MissingDatetimeError: Unexpected data[date_column] dtype.
        ModeError: Unexpected mode value is passed.

    Returns:
        A tuple containing a DataFrame with the aggregate values used to
        create the chart, the matplotlib chart Figure and Axes objects.
    """
    _validate_chart_parameters(data, date_column, agg_column, agg, mode)

    data = assign_temporal_columns(data, date_column, mode)
    agg_column = agg_column or date_column
    data_graph = aggregate_temporal_columns(data, agg_column, agg, mode)

    # convert aggregate function results to int64, if possible
    if (data_graph[agg] % 1 == 0).all():
        data_graph[agg] = data_graph[agg].astype("int64")

    # calculate optimal figure dimensions (0.85 per wedge)
    figure_size = get_figure_dimensions(data_graph["wedge"].size)

    # base figure spacing (10%) made available for Text, Subtitle & Period
    base_spacing = 0.10
    # scale spacing relative to figure minimum width/height (10,10)
    spacing_scale = figure_size[0] / 10
    # create a top margin for text elements, capped at 20%
    top_margin = min(base_spacing * (spacing_scale**0.5), 0.20)

    fig_kw.update({"figsize": figure_size, "constrained_layout": False})
    if "dpi" not in fig_kw:
        fig_kw.update({"dpi": 100})

    # create figure with polar projection
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, **fig_kw)

    # plot rect parameters; left, bottom, width & height
    rect = [0.1, 0.12, 0.8, 0.88 - top_margin]

    # apply the positioning
    ax.set_position(rect)

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

    # set x-axis ticks
    ax.xaxis.set_ticks(theta)
    ax.xaxis.set_ticklabels([])

    ax.yaxis.set_ticks(range(1, max_radius))
    ax.yaxis.set_ticklabels([])

    ax.xaxis.grid(visible=True, color=grid_color, alpha=0.6)
    ax.yaxis.grid(visible=True, color=grid_color, alpha=0.6)

    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color(spine_color)
    ax.spines["inner"].set_color("w")

    values_dtype = (np.float64, np.int64)[agg in ("count", "sum")]
    # we can use colorbar.cmap(colorbar.norm(<aggregation value>)),
    # to return the RGB values to represent each aggregation result
    colorbar = add_colorbar(
        ax, fig, cmap_name, cmap_reverse, data_graph[agg].max(), values_dtype
    )

    # create x-axis labels)
    if mode == "WEEK_DAY":
        wedge_labels = tuple(calendar.day_name)
    elif mode == "YEAR_MONTH":
        wedge_labels = tuple(calendar.month_name[1:])
    # custom x-axis labels for hour of day (00:00 - 23:00)
    elif mode in ("DOW_HOUR", "DAY_HOUR"):
        wedge_labels = [f"{x:02d}:00" for x in data_graph["wedge"].unique()]
    else:
        wedge_labels = tuple(map(str, data_graph["wedge"].unique()))

    figure_width, _ = figure_size
    font_scale_factor = figure_width / 11

    ring_scale_factor = max_radius / 3
    ring_text_spacing = 0.2

    add_wedge_labels(
        ax,
        font_scale_factor,
        ring_scale_factor,
        ring_text_spacing,
        max_radius,
        theta,
        width,
        wedge_labels,
    )

    # ring position starts from 1, creating a donut shape
    start_position = 1

    for ring_position, ring in enumerate(unique_rings):
        view = data_graph.loc[data_graph["ring"] == ring]

        graduated_colors = tuple(
            colorbar.cmap(colorbar.norm(i)) for i in view[agg]
        )

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
        # read config/dataclock.ini file
        config.read(dataclock_ini)

        if chart_title is None:
            chart_title = config.get("DEFAULT", "TITLE")

        if chart_subtitle is None:
            mode_description = config.get("mode.description", mode)
            chart_subtitle = f"{agg.title()} by {mode_description}"

    text_y = 0.95
    text_spacing = 0.03

    if font_scale_factor > 1:
        text_spacing = text_spacing * (font_scale_factor**0.1)
    else:
        text_spacing = text_spacing * font_scale_factor

    # add title, subtitle and period text to the figure
    for i, (text, fontsize, weight) in enumerate(
        zip(  # text | fontsize | weight,
            (chart_title, chart_subtitle, chart_period),
            np.array((14, 12, 10)) * font_scale_factor,
            ("bold", "normal", "normal"),
        )
    ):
        if text is None:
            continue

        # chart title text
        add_text(
            ax=ax,
            x=0.1,
            y=text_y - (i * text_spacing),
            text=text,
            fontsize=fontsize * font_scale_factor,
            weight=weight,
            alpha=0.8,
            transform=fig.transFigure,
        )

    # chart source text
    add_text(
        ax=ax,
        x=0.1,
        y=0.1,
        text=chart_source,
        fontsize=10 * font_scale_factor,
        alpha=0.7,
        transform=fig.transFigure,
    )

    return data_graph, fig, ax


def line_chart(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: Aggregation = "count",
    mode: Mode = "DAY_HOUR",
    default_text: bool = True,
    *,  # keyword only arguments
    chart_title: Optional[str] = None,
    chart_subtitle: Optional[str] = None,
    chart_period: Optional[str] = None,
    chart_source: Optional[str] = None,
    **fig_kw,
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
        default_text (bool, optional): Flag to generating default chart
            annotations for the chart_title ('Data Clock Chart') and
            chart_subtitle ('[agg] by [period] (rings) & [period] (wedges)').
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_period (str, optional): Chart reporting period.
        chart_source (str, optional): Chart data source.
        fig_kw (dict): Chart figure kwargs passed to pyplot.subplots.

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

    data = assign_temporal_columns(data, date_column, mode)

    # dict map for wedge min & max range based on mode
    wedge_range_map = {
        "YEAR_MONTH": tuple(calendar.month_name[1:]),
        "YEAR_WEEK": range(1, 53),
        "WEEK_DAY": tuple(calendar.day_name),
        "DOW_HOUR": range(0, 24),
        "DAY_HOUR": range(0, 24),
    }

    agg_column = agg_column or date_column

    data_agg = aggregate_temporal_columns(data, agg_column, agg, mode)
    data_graph = data_agg.set_index("ring")

    # convert aggregate function results to int64, if possible
    if (data_graph[agg] % 1 == 0).all():
        data_graph[agg] = data_graph[agg].astype("int64")

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=96)

    # adjust subplots for custom title, subtitle and source text
    plt.subplots_adjust(
        left=None, bottom=0.25, right=None, top=0.85, wspace=None, hspace=None
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

    # create x-axis labels)
    if mode == "WEEK_DAY":
        xaxis_labels = tuple(calendar.day_name)
    elif mode == "YEAR_MONTH":
        xaxis_labels = tuple(calendar.month_name[1:])
    # custom x-axis labels for hour of day (00:00 - 23:00)
    elif mode in ("DOW_HOUR", "DAY_HOUR"):
        xaxis_labels = [f"{x:02d}:00" for x in unique_wedges]
    else:
        xaxis_labels = tuple(map(str, unique_wedges))
    ax.set_xticks(range(n_wedges), xaxis_labels, rotation=45, ha="right")
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
            x, line_data[agg], color=cmap(idx), label=line_labels[i], zorder=2
        )

        point_args = (x[-1], line_data[agg].iloc[-1])
        point_kwargs = {
            "marker": "o",
            "color": cmap(idx),
        }

        # custom style for final point
        ax.plot(*point_args, **point_kwargs, markersize=10, alpha=0.3)
        ax.plot(*point_args, **point_kwargs, markersize=5)

    # add legend
    ax.legend(loc="best", fontsize=12)

    # generate default text for missing chart_title & chart_subtitle values
    if default_text:
        # read config/linechart.ini file
        config.read(linechart_ini)

        if chart_title is None:
            chart_title = config.get("DEFAULT", "TITLE")

        if chart_subtitle is None:
            mode_description = config.get("mode.description", mode)
            chart_subtitle = f"{agg.title()} by {mode_description}"

    fig_width, _ = (13.33, 17.5)
    font_scale_factor = fig_width / 13.33

    text_y = 0.95
    text_spacing = 0.03

    if font_scale_factor > 1:
        text_spacing = text_spacing * (font_scale_factor**0.1)
    else:
        text_spacing = text_spacing * font_scale_factor

    # add title, subtitle and period text to the figure
    for i, (text, fontsize, weight) in enumerate(
        zip(  # text | fontsize | weight,
            (chart_title, chart_subtitle, chart_period),
            (14, 12, 10),
            ("bold", "normal", "normal"),
        )
    ):
        # chart text
        add_text(
            ax=ax,
            x=0.1,
            y=text_y - (i * text_spacing),
            text=text,
            fontsize=fontsize * font_scale_factor,
            weight=weight,
            alpha=0.8,
            transform=fig.transFigure,
        )

    # chart source text
    add_text(
        ax=ax,
        x=0.1,
        y=0.1,
        text=chart_source,
        fontsize=10 * font_scale_factor,
        alpha=0.7,
        transform=fig.transFigure,
    )

    return data_graph, fig, ax


def _validate_chart_parameters(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: Aggregation = "count",
    mode: str = "DAY_HOUR",
) -> None:
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
        KeyError: Column not in DataFrame.
        ModeError: Unexpected mode value is passed.

    Returns:
        None
    """
    if data.empty:
        raise EmptyDataFrameError(data)
    if date_column not in data.columns:
        raise KeyError(f"Column {date_column=} not in DataFrame.")
    if agg_column is not None and agg_column not in data.columns:
        raise KeyError(f"Column {agg_column=} not in DataFrame.")
    if data[date_column].dtype.name != "datetime64[ns]":
        raise MissingDatetimeError(date_column)
    if mode not in VALID_MODES:
        raise ModeError(mode, VALID_MODES)
    if agg not in VALID_AGGREGATIONS:
        raise AggregationFunctionError(agg, VALID_AGGREGATIONS)
    if agg_column is None and agg != "count":
        raise AggregationColumnError(agg)
