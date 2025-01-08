"""Matplotlib image comparison unit test module.

Author: Andrew Ridyard.

License: GNU General Public License v3 or later.

Copyright (C): 2025.

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
    test_baseline: Image comparison test function.
"""
import pathlib

import pandas as pd
import pytest

from dataclocklib.charts import dataclock

tests_directory = pathlib.Path("__file__").parent / "tests"
data_file = tests_directory / "data" / "traffic_data.parquet.gzip"
traffic_data = pd.read_parquet(data_file.as_posix())


@pytest.mark.mpl_image_compare
def test_dow_hour_chart():
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=baseline

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
        chart_title="UK Car Accidents 2010",
        chart_subtitle=None,
        chart_source="www.kaggle.com/datasets/silicon99/dft-accident-data"
    )
    return fig

@pytest.mark.mpl_image_compare
def test_day_hour_chart():
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=baseline

    Generated images are placed in a new directory called 'baseline' and moved
    as a sub-directory of the 'tests/plotting' directory, if they are correct.

    Returns:
        A matplotlib Figure, which is used to generate a baseline image.
    """

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(
            "(Date_Time >= '2010-12-1 00:00:00') & (Date_Time <= '2010-12-14 23:59:59')"
        ),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="DAY_HOUR",
        chart_title="UK Car Accidents 1 - 14 December 2010",
        chart_subtitle=None,
        chart_source="www.kaggle.com/datasets/silicon99/dft-accident-data"
    )
    return fig


@pytest.mark.mpl_image_compare
def test_year_month_chart():
    """Image comparison test function.

    This function generates a baseline image, after running the pytest
    suite with the '--mpl-generate-path' option:

    >>> pytest --mpl-generate-path=baseline

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
        chart_title="UK Car Accidents 2014 - 2015",
        chart_subtitle=None,
        chart_source="www.kaggle.com/datasets/silicon99/dft-accident-data"
    )
    return fig



