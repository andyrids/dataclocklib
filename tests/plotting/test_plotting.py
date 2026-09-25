"""Matplotlib image comparison unit test module.

Functions:
    test_baseline: Image comparison test function.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

import pathlib

import pandas as pd
import pytest
from matplotlib.figure import Figure

from dataclocklib.charts import dataclock

tests_directory = pathlib.Path(__file__).parent.parent
data_file = tests_directory / "data" / "traffic_data.parquet.gzip"
traffic_data = pd.read_parquet(data_file.as_posix())


@pytest.mark.mpl_image_compare
def test_baseline_year_month_chart() -> Figure:
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=tests/plotting/baseline

    Generated images are placed in a new directory called 'baseline' and moved
    as a sub-directory of the 'tests/plotting' directory, if they are correct.

    Returns:
        A matplotlib Figure, which is used to generate a baseline image.
    """
    chart_data, fig, ax = dataclock(
        data=traffic_data.query("Date_Time.dt.year.ge(2014)"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="YEAR_MONTH",
        cmap_name="RdYlGn_r",
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_baseline_week_day_chart() -> Figure:
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=tests/plotting/baseline

    Generated images are placed in a new directory called 'baseline' and moved
    as a sub-directory of the 'tests/plotting' directory, if they are correct.

    Returns:
        A matplotlib Figure, which is used to generate a baseline image.
    """
    datetime_start = "Date_Time.ge('2010-12-1 00:00:00')"
    datetime_stop = "Date_Time.le('2010-12-31 23:59:59')"
    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="WEEK_DAY",
        cmap_name="RdYlGn_r",
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_baseline_dow_hour_chart() -> Figure:
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=tests/plotting/baseline

    Generated images are placed in a new directory called 'baseline' and moved
    as a sub-directory of the 'tests/plotting' directory, if they are correct.

    Returns:
        A matplotlib Figure, which is used to generate a baseline image.
    """
    chart_data, fig, ax = dataclock(
        data=traffic_data.query("Date_Time.dt.year.eq(2010)"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="DOW_HOUR",
        cmap_name="RdYlGn_r",
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_baseline_day_hour_chart() -> Figure:
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=tests/plotting/baseline

    Generated images are placed in a new directory called 'baseline' and moved
    as a sub-directory of the 'tests/plotting' directory, if they are correct.

    Returns:
        A matplotlib Figure, which is used to generate a baseline image.
    """
    datetime_start = "Date_Time.ge('2010-12-1 00:00:00')"
    datetime_stop = "Date_Time.le('2010-12-14 23:59:59')"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="DAY_HOUR",
        cmap_name="RdYlGn_r",
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig
