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

Functions:

"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..dataclock import example_function


@pytest.mark.mpl_image_compare(filename="test_example.png")
def test_example():
    """_summary_

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots()
    example_function(ax, data, above_color="b", below_color="g")
    return fig
