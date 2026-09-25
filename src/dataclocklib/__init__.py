"""__init__ for dataclocklib package.

NOTE:  We generate __version__ from the 'dataclocklib' package information,
facilitated by 'hatch-vcs'.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

from importlib.metadata import PackageNotFoundError, version

from dataclocklib.charts import dataclock, line_chart

try:
    __version__ = version("dataclocklib")
except PackageNotFoundError:
    pass

__all__ = ("dataclock", "line_chart")
