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
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Functions:
    temporal_features: Generate a new DataFrame with set temporal features.
    dataclock: Create a data clock chart from a pandas DataFrame.

Types:
    TemporalMode: Mode keys representing temporal bins used in a chart.
"""

from __future__ import annotations

import calendar
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pandas import DataFrame, MultiIndex, NamedAgg

Mode = Literal["YEAR_MONTH", "YEAR_WEEK", "WEEK_DAY", "DAY_HOUR"]


def dataclock(
    data: DataFrame,
    date_column: str,
    agg_column: Optional[str] = None,
    agg: str = "count",
    mode: Mode = "DAY_HOUR",
) -> DataFrame:
    """Create a data clock chart from a pandas DataFrame.

    Data clocks visually summarise temporal data in two dimensions,
    revealing seasonal or cyclical patterns and trends over time.
    A data clock is a circular chart that divides a larger unit of
    time into rings and subdivides it by a smaller unit of time into
    wedges, creating a set of temporal bins.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        mode (TemporalMode, optional): A mode key representing the
            temporal bins used in the chart; "YM" (Year-Month),
            "YW" (Year-Week), "YD" (Year-Day), "WD" (Week-Day),
            "DH" (Day-Hour).

    Raises:
        ValueError: If an incorrect mode value is passed.

    Returns:
        DataFrame with aggregate values used to create the data clock chart.
    """
    mode_map = {
        "YEAR_MONTH": {
            "ring": data[date_column].dt.strftime("%Y"),
            "wedge": data[date_column].dt.strftime("%B")
        },
        "YEAR_WEEK": {
            "ring": data[date_column].dt.strftime("%Y"),
            "wedge": data[date_column].dt.isocalendar().week
        },
        "WEEK_DAY": {
            "ring": data[date_column].dt.strftime("%Y%W"),
            "wedge": data[date_column].dt.strftime("%A")
        },
        "DAY_HOUR": {
            "ring": data[date_column].dt.strftime("%Y%j"),
            "wedge": data[date_column].dt.hour
        }
    }

    if mode not in mode_map:
        raise ValueError(f"Unexpected mode value ({mode}): {Mode}")

    data = data.assign(**mode_map[mode]).astype({"ring": "int64"})

    wedge_range_map = {
        "YEAR_MONTH": tuple(calendar.month_name[1:]),
        "YEAR_WEEK": range(1, 53),
        "WEEK_DAY": tuple(calendar.day_name),
        "DAY_HOUR": range(0, 24),
    }

    index_names = ["ring", "wedge"]
    agg_column = agg_column or date_column

    data_grouped = data.groupby(index_names, as_index=False)
    data_agg = data_grouped.agg(**{agg: NamedAgg(agg_column, agg)})

    product_index = MultiIndex.from_product(
        [data_agg["ring"].unique(), wedge_range_map[mode]], names=index_names
    )

    data_agg = (
        data_agg.set_index(index_names).reindex(product_index).reset_index()
    )

    data_graph = data_agg.fillna(0)

    # convert aggregate function results to int64, if possible
    if (data_graph[agg] % 1 == 0).all():
        data_graph[agg] = data_graph[agg].astype("int64")

    # create figure with polar projection
    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"}, figsize=(10, 10), dpi=96
    )

    plt.subplots_adjust(
        left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None
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

    max_radius = max(unique_rings.size + 1, 3)
    ax.set_rlim(0, max_radius)

    if mode != "DAY_HOUR":
        xaxis_labels = wedge_range_map[mode]
    else:
        xaxis_labels = [f"{x:2d}:00" for x in range(24)]

    # set x-axis ticks
    ax.xaxis.set_ticks(theta, xaxis_labels)
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis="x", pad=20)

    ax.xaxis.grid(visible=False)
    ax.yaxis.grid(visible=False)

    ax.spines["polar"].set_visible(True)

    start_position = 2

    colours = ["darkgreen", "gold", "darkred"]
    cmap = LinearSegmentedColormap.from_list("colour_map", colours, N=256)
    cmap_norm = Normalize(1, data_graph[agg].max())

    for ring_position, ring in enumerate(unique_rings):
        view = data_graph.loc[data_graph["ring"] == ring]
        for wedge_position, wedge in enumerate(theta):
            count = view[agg].iat[wedge_position]
            colour = cmap(cmap_norm(count)) if count else "w"

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
                edgecolor="black",
                linewidth=0.8,
            )
