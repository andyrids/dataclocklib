"""Utility function module for chart creation.

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
    add_text: Create annotation text on an Axes.

Types:
    Aggregation: Keys representing aggregation functions.
    Mode: Keys representing temporal bins used in each chart.
"""
from typing import Literal, Optional, Tuple, get_args

from matplotlib.axes import Axes
from matplotlib.text import Text

FontStyle = Literal["normal", "italic", "oblique"]
VALID_STYLES: Tuple[FontStyle, ...] = get_args(FontStyle)


def add_text(
    ax: Axes,
    x: int,
    y: int,
    text: Optional[str] = None,
    **kwargs
    ) -> Text:
    """Annotate a position on an axis denoted by xy with text.

    Args:
        ax (Axes): Axis to annotate.
        x (int): Axis x position.
        y (int): Axis y position.
        text (str, optional): Text to annotate.

    Returns:
        Text object with annotation.
    """
    s = "" if text is None else text
    return ax.text(x, y, s, **kwargs)
