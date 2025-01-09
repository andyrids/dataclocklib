"""Unit test module.

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

TODO: Functions
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dataclocklib.charts import dataclock

tests_directory = pathlib.Path("__file__").parent / "tests"
data_file = tests_directory / "data" / "traffic_data.parquet.gzip"
traffic_data = pd.read_parquet(data_file.as_posix())


@pytest.mark.mpl_image_compare(filename="test_dataclock.png")
def test_example():
    """_summary_

    >>> pytest --mpl

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots()
    example_function(ax, data, above_color="b", below_color="g")
    return fig

    chart_data, fig, ax = dataclock()

    return fig
