# Data Clock Visualisation Library

![PyPI - Version](https://img.shields.io/pypi/v/dataclocklib?style=plastic) ![PyPI - Downloads](https://img.shields.io/pypi/dm/dataclocklib?style=plastic) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fandyrids%2Fdataclocklib%2Fmain%2Fpyproject.toml&style=plastic) ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/andyrids/dataclocklib/ci.yml?branch=main&style=plastic&logo=pytest&logoColor=%230A9EDC&label=pytest)
![GitHub deployments](https://img.shields.io/github/deployments/andyrids/dataclocklib/github-pages?style=plastic&&logo=sphinx&label=sphinx)

## Introduction

This library allows the user to create data clock graphs, using the matplotlib Python library.

Data clocks visually summarise temporal data in two dimensions, revealing seasonal or cyclical patterns and trends over time. A data clock is a circular chart that divides a larger unit of time into rings and subdivides it by a smaller unit of time into wedges, creating a set of temporal bins.

These temporal bins are symbolised using graduated colors that correspond to a count or aggregated value taking place in each time period.

The table below details the currently supported chart modes and the corresponding rings and wedges:

| Mode       | Rings            | Wedges           | Description                       |
|------------|------------------|------------------|-----------------------------------|
| YEAR_MONTH | Years            | Months           | Years / January - December.       |
| YEAR_WEEK  | Years            | Weeks            | Years / weeks 1 - 52.             |
| WEEK_DAY   | Weeks            | Days of the week | Weeks 1 - 52 / Monday - Sunday.   |
| DOW_HOUR   | Days of the week | Hour of day      | Monday - Sunday / 24 hours.       |
| DAY_HOUR   | Days             | Hour of day      | Days 1 - 356 / 24 hours.          |

The full documentation can be viewed on the project [GitHub Page](https://andyrids.github.io/dataclocklib/).

### Example charts

Chart examples have been generated using UK Department for Transport data 2010 - 2015.

```python
import pandas as pd
from dataclocklib.charts import dataclock

data = pd.read_parquet(
    "https://raw.githubusercontent.com/andyrids/dataclocklib/main/tests/data/traffic_data.parquet.gzip"
)

graph_data, fig, ax = dataclock(
    data=data.query("Date_Time.dt.year.eq(2015)"),
    date_column="Date_Time",
    agg_column="Number_of_Casualties",
    agg="sum",
    mode="DOW_HOUR",
    cmap_name="CMRmap_r",
    chart_title="UK Traffic Accident Casualties",
    chart_subtitle=None,
    chart_period="Period: 2015",
    chart_source="www.kaggle.com/datasets/silicon99/dft-accident-data",
    default_text=True,
)
```

![Data clock chart](https://raw.githubusercontent.com/andyrids/dataclocklib/main/docs/source/_static/images/sphinx_index_chart_1.png)

```python
import pandas as pd
from dataclocklib.charts import dataclock

data = pd.read_parquet(
    "https://raw.githubusercontent.com/andyrids/dataclocklib/main/tests/data/traffic_data.parquet.gzip"
)

graph_data, fig, ax = dataclock(
    data=data.query("Date_Time.dt.year.eq(2010)"),
    date_column="Date_Time",
    agg_column=None,
    agg="count",
    mode="DOW_HOUR",
    cmap_name="RdYlGn_r",
    chart_title="UK Traffic Accidents",
    chart_subtitle=None,
    chart_period="Period: 2010",
    chart_source="www.kaggle.com/datasets/silicon99/dft-accident-data",
    default_text=True,
)
```

![Data clock chart](https://raw.githubusercontent.com/andyrids/dataclocklib/main/docs/source/_static/images/sphinx_index_chart_2.png)

## Installation

You can install using `pip`:

```bash
python -m pip install dataclocklib
```

To install from GitHub use:

```bash
python -m pip install git+https://github.com/andyrids/dataclocklib.git
```

## Development Installation

Astral **uv** is used as the Python package manager. To install **uv** see the installation
guide @ [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository:

```bash
git clone git@github.com:andyrids/dataclocklib.git
cd dataclocklib
```

Sync the dependencies, including the dev dependency group and optional dependencies with uv:

```bash
uv sync --all-extras
```

Activate the virtual environment:

```bash
. .venv/bin/activate
```

### Sphinx documentation

```bash
cd docs
make html
```
