"""Data clock module for chart creation.

Author: Andrew Ridyard.

License: GNU General Public License v3 or later.

Copyright (C): 2024.

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
    Mode: Mode keys representing temporal bins used in each chart.
"""

from __future__ import annotations

import calendar
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from pandas import DataFrame, MultiIndex, NamedAgg

Mode = Literal["YEAR_MONTH", "YEAR_WEEK", "WEEK_DAY", "DOW_HOUR", "DAY_HOUR"]


def dataclock(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: str = "count",
    mode: Mode = "DAY_HOUR",
    chart_title: Optional[str] = None,
    chart_subtitle: Optional[str] = None,
    chart_source: Optional[str] = None,
) -> tuple[DataFrame, Figure, Axes]:
    """Create a data clock chart from a pandas DataFrame.

    Data clocks visually summarise temporal data in two dimensions,
    revealing seasonal or cyclical patterns and trends over time.
    A data clock is a circular chart that divides a larger unit of
    time into rings and subdivides it by a smaller unit of time into
    wedges, creating a set of temporal bins.

    YEAR_MONTH: Rings are years and wedges are the January - December.
    YEAR_WEEK: Rings are years and wedges are weeks 1 - 52.
    WEEK_DAY: Rings are weeks 1 - 52 and wedges are Monday - Sunday.
    DOW_HOUR: Rings are 1 (Monday) - 7 (Sunday) & wedges are 00:00 - 23:00.
    DAY_HOUR: Rings are days 1 - 356 & wedges are 00:00 - 23:00.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        agg (str): Aggregation function.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.
        chart_title (str, optional): Chart title.
        chart_subtitle (str, optional): Chart subtitle.
        chart_source (str, optional): Chart data source.

    Raises:
        ValueError: If an incorrect mode value is passed.

    Returns:
        A tuple containing a DataFrame with the aggregate values used to
        create the chart, the matplotlib chart Figure and Axes objects.
    """

    # dict map for ring & wedge features based on mode
    mode_map = {
        "YEAR_MONTH": {  # year | January - December
            "ring": data[date_column].dt.strftime("%Y"),
            "wedge": data[date_column].dt.strftime("%B"),
        },
        "YEAR_WEEK": {  # year | weeks 1 - 52
            "ring": data[date_column].dt.strftime("%Y"),
            "wedge": data[date_column].dt.isocalendar().week,
        },
        "WEEK_DAY": {  # weeks 1 - 52 | Monday - Sunday
            "ring": data[date_column].dt.strftime("%Y%W"),
            "wedge": data[date_column].dt.strftime("%A"),
        },
        "DOW_HOUR": {  # days 1 - 7 (Monday - Sunday) | 00:00 - 23:00
            "ring": data[date_column].dt.strftime("%w").replace("0", "7"),
            "wedge": data[date_column].dt.hour,
        },
        "DAY_HOUR": {  # days 1 - 365 | 00:00 - 23:00
            "ring": data[date_column].dt.strftime("%Y%j"),
            "wedge": data[date_column].dt.hour,
        },
    }
    if mode not in mode_map:
        raise ValueError(f"Unexpected mode value ({mode}): {Mode}")

    data = data.assign(**mode_map[mode]).astype({"ring": "int64"})

    # dict map for wedge min & max range based on mode
    wedge_range_map = {
        "YEAR_MONTH": tuple(calendar.month_name[1:]),
        "YEAR_WEEK": range(1, 53),
        "WEEK_DAY": tuple(calendar.day_name),
        "DAY_HOUR": range(0, 24),
        "DOW_HOUR": range(0, 24),
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
    width = 2 * np.pi / n_wedges * 1

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

    colours = ["darkgreen", "gold", "darkred"]
    cmap = LinearSegmentedColormap.from_list("colour_map", colours, N=256)
    cmap.set_under("w")
    cmap_norm = Normalize(1, data_graph[agg].max())

    agg_max = data_graph[agg].max()
    colourbar_ticks = [
        1,
        agg_max * 0.25,
        agg_max * 0.5,
        agg_max * 0.75,
        agg_max,
    ]

    if agg == "count":
        colourbar_ticks = list(map(int, colourbar_ticks))

    # display 1 - maximum with 25% increments on colour bar
    fig.colorbar(
        ScalarMappable(norm=cmap_norm, cmap=cmap),
        ax=ax,
        orientation="vertical",
        ticks=colourbar_ticks,
        shrink=0.6,
        pad=0.2,
    )

    # set text label y position beyond polar axis radial limit
    text_y = max_radius + 0.5 if max_radius > 3 else max_radius + 0.2

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
            text_label_y,
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
        for wedge_position, wedge in enumerate(theta):
            count = view[agg].iat[wedge_position]
            colour = cmap(cmap_norm(count))

            # plot each wedge as a separate bar along each ring
            bar = ax.bar(
                # wedge/angle
                wedge,
                # height
                1,
                # bar aligned to wedge
                align="edge",
                # width in radians
                width=width,
                # ring to place bar
                bottom=start_position + ring_position,
                # transparency
                alpha=0.8,
                # color map
                color=colour,
            )

    chart_title = chart_title or "Dataclock Chart"

    chart_subtitle_map = {
        "YEAR_MONTH": f"{agg.title()} by year & month",
        "YEAR_WEEK": f"{agg.title()} by year & week of year",
        "WEEK_DAY": f"{agg.title()} by week of year & day of week",
        "DOW_HOUR": f"{agg.title()} by Day of week & hour of day",
        "DAY_HOUR": f"{agg.title()} by Day of year & hour of day",
    }
    chart_subtitle = chart_subtitle or chart_subtitle_map[mode]

    text_kwargs = {"ha": "left", "transform": fig.transFigure}

    # create chart title, subtitle & source information
    ax.text(
        x=0.12,
        y=0.93,
        s=chart_title,
        fontsize=14,
        weight="bold",
        alpha=0.8,
        **text_kwargs,
    )

    ax.text(
        x=0.12, y=0.90, s=chart_subtitle, fontsize=12, alpha=0.8, **text_kwargs
    )

    if chart_source:
        # Set source text
        ax.text(
            x=0.1,
            y=0.12,
            s=chart_source,
            fontsize=10,
            alpha=0.7,
            **text_kwargs,
        )

    return data_graph, fig, ax
