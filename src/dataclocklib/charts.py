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
from typing import TYPE_CHECKING, Any, get_args

import matplotlib.pyplot as plt
import numpy as np

# Axes, Figure & DataFrame kept at runtime (not TYPE_CHECKING) so
# typing.get_type_hints resolves them for the public dataclock/line_chart API.
from matplotlib.axes import Axes  # noqa: TC002
from matplotlib.figure import Figure  # noqa: TC002
from pandas import DataFrame  # noqa: TC002
from pandas.api.types import is_datetime64_dtype

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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.colorbar import Colorbar
    from matplotlib.projections.polar import PolarAxes
    from numpy.typing import NDArray
    from pandas import Series

VALID_AGGREGATIONS: tuple[Aggregation, ...] = get_args(Aggregation)
VALID_CMAPS: tuple[CmapNames, ...] = get_args(CmapNames)
VALID_MODES: tuple[Mode, ...] = get_args(Mode)

# config files for default title and subtitle text
dataclock_ini = pathlib.Path(__file__).parent / "config" / "dataclock.ini"
linechart_ini = pathlib.Path(__file__).parent / "config" / "linechart.ini"


def dataclock(
    data: DataFrame,
    date_column: str,
    agg_column: str | None = None,
    agg: Aggregation = "count",
    mode: Mode = "DAY_HOUR",
    cmap_name: str = "RdYlGn_r",
    cmap_reverse: bool = False,
    spine_color: str = "darkslategrey",
    grid_color: str = "darkslategrey",
    default_text: bool = True,
    *,  # keyword only arguments
    chart_title: str | None = None,
    chart_subtitle: str | None = None,
    chart_period: str | None = None,
    chart_source: str | None = None,
    **fig_kw: Any,
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
        date_column (str): Name of DataFrame naive datetime64 column, of any
            resolution ('ns', 'us', 'ms' or 's').
        agg_column (str, optional): DataFrame Column to aggregate.
        agg (Aggregation, optional): Aggregation function; 'count', 'max',
            'mean', 'median', 'min' & 'sum'.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.
        cmap_name (str, optional): Name of a matplotlib/PyPalettes colormap,
            to symbolise the temporal bins; 'RdYlGn_r', 'CMRmap_r',
            'inferno_r', 'Alkalay2', 'viridis', 'a_palette' etc.
        cmap_reverse (bool, optional): Reverse cmap colors flag.
        spine_color (str, optional): Name of color to style the polar axis
            spines.
        grid_color (str, optional): Name of color to style the polar axis
            grid lines.
        default_text (bool, optional): Flag to generating default chart
            annotations for the chart_title ('Data Clock Chart') and
            chart_subtitle ('[agg] by [period] (rings) & [period] (wedges)').
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_period (str, optional): Chart reporting period.
        chart_source (str, optional): Chart data source.
        **fig_kw (Any): Chart figure kwargs passed to pyplot.subplots.

    Raises:
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        EmptyDataFrameError: Unexpected empty DataFrame.
        KeyError: date_column or agg_column not in DataFrame.
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
    data_graph[agg] = _as_int_if_integral(data_graph[agg])

    # calculate optimal figure dimensions (0.85 per wedge)
    figure_size = get_figure_dimensions(data_graph["wedge"].size)

    # base figure spacing (10%) made available for Text, Subtitle & Period
    base_spacing = 0.10
    # scale spacing relative to figure minimum width/height (10,10)
    spacing_scale = figure_size[0] / 10
    # create a top margin for text elements, capped at 20%
    top_margin = min(base_spacing * (spacing_scale**0.5), 0.20)

    fig_kw.update({"figsize": figure_size, "constrained_layout": False})
    fig_kw.setdefault("dpi", 100)

    # create figure with polar projection
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, **fig_kw)

    # plot rect parameters; left, bottom, width & height
    ax.set_position((0.1, 0.12, 0.8, 0.88 - top_margin))

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

    max_radius = data_graph["ring"].nunique() + 1

    _style_polar_axes(ax, theta, max_radius, grid_color, spine_color)

    values_dtype = (np.float64, np.int64)[agg in ("count", "sum")]
    # we can use colorbar.cmap(colorbar.norm(<aggregation value>)),
    # to return the RGB values to represent each aggregation result
    colorbar = add_colorbar(
        ax, fig, cmap_name, cmap_reverse, data_graph[agg].max(), values_dtype
    )

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
        _wedge_labels(mode, data_graph["wedge"].unique()),
    )

    _draw_rings(ax, data_graph, agg, colorbar, theta, width)

    # generate default text for missing chart_title & chart_subtitle values
    if default_text:
        chart_title, chart_subtitle = _default_text(
            dataclock_ini, mode, agg, chart_title, chart_subtitle
        )

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
            strict=True,
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
    agg_column: str | None = None,
    agg: Aggregation = "count",
    mode: Mode = "DAY_HOUR",
    default_text: bool = True,
    *,  # keyword only arguments
    chart_title: str | None = None,
    chart_subtitle: str | None = None,
    chart_period: str | None = None,
    chart_source: str | None = None,
    **fig_kw: Any,
) -> tuple[DataFrame, Figure, Axes]:
    """Create a temporal line chart from a pandas DataFrame.

    This function will divide a larger unit of time into rings and subdivide
    them by a smaller unit of time into wedges, creating temporal bins. The
    ring values will be represented as individual lines, with the aggregation
    values on the y-axis and wedges as the x-axis.

    NOTE: fig_kw is accepted but currently unused; the figure is always
    created with figsize=(13.33, 7.5) & dpi=96.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame naive datetime64 column, of any
            resolution ('ns', 'us', 'ms' or 's').
        agg_column (str, optional): DataFrame Column to aggregate.
        agg (Aggregation, optional): Aggregation function; 'count', 'max',
            'mean', 'median', 'min' & 'sum'.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.
        default_text (bool, optional): Flag to generating default chart
            annotations for the chart_title ('Line Chart') and
            chart_subtitle ('[agg] by [period] & [period]').
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_period (str, optional): Chart reporting period.
        chart_source (str, optional): Chart data source.
        **fig_kw (Any): Chart figure kwargs (currently unused).

    Raises:
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        EmptyDataFrameError: Unexpected empty DataFrame.
        KeyError: date_column or agg_column not in DataFrame.
        MissingDatetimeError: Unexpected data[date_column] dtype.
        ModeError: Unexpected mode value is passed.

    Returns:
        A tuple containing a DataFrame with the aggregate values used to
        create the chart, the matplotlib chart Figure and Axes objects.
    """
    _validate_chart_parameters(data, date_column, agg_column, agg, mode)

    data = assign_temporal_columns(data, date_column, mode)
    agg_column = agg_column or date_column

    data_agg = aggregate_temporal_columns(data, agg_column, agg, mode)
    data_graph = data_agg.set_index("ring")
    data_graph[agg] = _as_int_if_integral(data_graph[agg])

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=96)

    # adjust subplots for custom title, subtitle and source text
    fig.subplots_adjust(
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
    xaxis_labels = _wedge_labels(mode, data_graph["wedge"].unique())
    ax.set_xticks(range(n_wedges), xaxis_labels, rotation=45, ha="right")
    ax.set_xlabel("", fontsize=12, labelpad=10)

    ax.set_ylabel(agg.title(), fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_major_formatter(lambda s, _: f"{s:,.0f}")
    ax.yaxis.set_tick_params(
        pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12
    )

    unique_indices = data_graph.index.unique()
    if mode == "DOW_HOUR":
        line_labels = dict(enumerate(calendar.day_name))
    else:
        line_labels = dict(zip(unique_indices, unique_indices, strict=True))

    cmap = plt.get_cmap("tab10")

    for idx, i in enumerate(unique_indices):
        line_data = data_graph.loc[i]
        # ensure x is always numeric
        x = list(range(line_data["wedge"].size))
        y = line_data[agg]
        colour = cmap(idx)

        ax.plot(x, y, color=colour, label=line_labels[i], zorder=2)

        # custom style for final point
        y_last = y.iloc[-1]
        ax.plot(
            x[-1], y_last, marker="o", color=colour, markersize=10, alpha=0.3
        )
        ax.plot(x[-1], y_last, marker="o", color=colour, markersize=5)

    # add legend
    ax.legend(loc="best", fontsize=12)

    # generate default text for missing chart_title & chart_subtitle values
    if default_text:
        chart_title, chart_subtitle = _default_text(
            linechart_ini, mode, agg, chart_title, chart_subtitle
        )

    text_y = 0.95
    text_spacing = 0.03

    # add title, subtitle and period text to the figure
    for i, (text, fontsize, weight) in enumerate(
        zip(  # text | fontsize | weight,
            (chart_title, chart_subtitle, chart_period),
            (14, 12, 10),
            ("bold", "normal", "normal"),
            strict=True,
        )
    ):
        # chart text
        add_text(
            ax=ax,
            x=0.1,
            y=text_y - (i * text_spacing),
            text=text,
            fontsize=fontsize,
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
        fontsize=10,
        alpha=0.7,
        transform=fig.transFigure,
    )

    return data_graph, fig, ax


def _as_int_if_integral(values: Series) -> Series:
    """Convert aggregation values to int64, if every value is integral.

    Args:
        values (Series): Aggregation values.

    Returns:
        The values as int64 if every value is integral, otherwise unchanged.
    """
    if (values % 1 == 0).all():
        return values.astype("int64")
    return values


def _default_text(
    ini: pathlib.Path,
    mode: Mode,
    agg: Aggregation,
    title: str | None,
    subtitle: str | None,
) -> tuple[str | None, str | None]:
    """Fill a missing chart title & subtitle with default ini file text.

    Args:
        ini (pathlib.Path): Config file with a default title & descriptions.
        mode (Mode): A mode key representing the temporal bins in the chart.
        agg (Aggregation): Aggregation function name.
        title (str, optional): Chart title; default text is used if None.
        subtitle (str, optional): Chart subtitle; default text is used if None.

    Returns:
        A tuple containing the chart title and subtitle.
    """
    config = configparser.ConfigParser()
    config.read(ini)

    if title is None:
        title = config.get("DEFAULT", "TITLE")

    if subtitle is None:
        mode_description = config.get("mode.description", mode)
        subtitle = f"{agg.title()} by {mode_description}"

    return title, subtitle


def _wedge_labels(mode: Mode, wedges: Iterable[int]) -> tuple[str, ...]:
    """Create a text label for each wedge, based on the chart mode.

    Args:
        mode (Mode): A mode key representing the temporal bins in the chart.
        wedges (Iterable[int]): Unique wedge values.

    Returns:
        A tuple of wedge labels; day names (WEEK_DAY), month names
        (YEAR_MONTH), hours '00:00' - '23:00' (DOW_HOUR & DAY_HOUR) or the
        wedge values as strings (YEAR_WEEK).
    """
    match mode:
        case "WEEK_DAY":
            return tuple(calendar.day_name)
        case "YEAR_MONTH":
            return tuple(calendar.month_name[1:])
        case "DOW_HOUR" | "DAY_HOUR":
            return tuple(f"{x:02d}:00" for x in wedges)
        case _:
            return tuple(map(str, wedges))


def _style_polar_axes(
    ax: PolarAxes,
    theta: NDArray[np.float64],
    max_radius: int,
    grid_color: str,
    spine_color: str,
) -> None:
    """Set the polar axis limits, ticks, grid lines and spines.

    Args:
        ax (PolarAxes): Chart polar Axes.
        theta (NDArray[np.float64]): Angles (radians) for each wedge.
        max_radius (int): Maximum radius (unique rings + 1).
        grid_color (str): Name of color to style the polar axis grid lines.
        spine_color (str): Name of color to style the polar axis spines.

    Returns:
        None
    """
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


def _draw_rings(
    ax: Axes,
    data_graph: DataFrame,
    agg: Aggregation,
    colorbar: Colorbar,
    theta: NDArray[np.float64],
    width: float,
) -> None:
    """Draw a ring of colour graduated wedge bars for each unique ring value.

    Args:
        ax (Axes): Chart polar Axes.
        data_graph (DataFrame): Aggregated 'ring', 'wedge' & agg columns.
        agg (Aggregation): Aggregation function (value column) name.
        colorbar (Colorbar): Colorbar used to map values to colours.
        theta (NDArray[np.float64]): Angles (radians) for each wedge.
        width (float): Width of each wedge (radians).

    Returns:
        None
    """
    # ring position starts from 1, creating a donut shape
    start_position = 1

    for ring_position, ring in enumerate(data_graph["ring"].unique()):
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


def _validate_chart_parameters(
    data: DataFrame,
    date_column: str,
    agg_column: str | None = None,
    agg: Aggregation = "count",
    mode: str = "DAY_HOUR",
) -> None:
    """Validate chart parameters.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame naive datetime64 column.
        agg_column (str, optional): DataFrame Column to aggregate.
        agg (Aggregation, optional): Aggregation function; 'count', 'max',
            'mean', 'median', 'min' & 'sum'.
        mode (str, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.

    Raises:
        AggregationColumnError: Expected aggregation column value.
        AggregationFunctionError: Unexpected aggregation function value.
        EmptyDataFrameError: Unexpected empty DataFrame.
        KeyError: Column not in DataFrame.
        MissingDatetimeError: date_column is not a naive datetime64 dtype.
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
    # naive datetime64 of any resolution; tz-aware dtypes are rejected
    if not is_datetime64_dtype(data[date_column]):
        raise MissingDatetimeError(date_column)
    if mode not in VALID_MODES:
        raise ModeError(mode, VALID_MODES)
    if agg not in VALID_AGGREGATIONS:
        raise AggregationFunctionError(agg, VALID_AGGREGATIONS)
    if agg_column is None and agg != "count":
        raise AggregationColumnError(agg)
