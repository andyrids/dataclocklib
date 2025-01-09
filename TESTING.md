# Testing

This project uses Pytest along with the [pytest-mpl](https://github.com/matplotlib/pytest-mpl) plugin
to test matplotlib figures and chart generation functions.

## Generate Baseline Charts

To generate fresh baseline chart images with the *tests/plotting/test_plotting.py*, use the following command:

```bash
pytest --mpl-generate-path=tests/plotting/baseline
```

This will need to be done after creating new functions within *tests/plotting/test_plotting.py* or modifying their
output. A baseline sub-directory will be created within the tests/plotting directory.

## Running Tests

Tests can be run as usual, but with the `--mpl` option, which enables comparison of any returned Figure objects
in unit test functions with the reference figures in the baseline directory.

```bash
pytest --mpl --mpl-baseline-path=tests/plotting/baseline
```

Test functions can be wrapped to utilise a particular reference figure for comparison:

```python
@pytest.mark.mpl_image_compare(filename="test_year_month_chart.png")
def test_function():
    pass
```

By also passing `--mpl-generate-summary=html`, a summary of the image comparison results will be generated in HTML format:

```bash
pytest --mpl --mpl-baseline-path=tests/plotting/baseline --mpl-generate-summary=html
```

Coverage report can be generated using the following command:

```bash
pytest --mpl --cov-report term --cov=dataclocklib
```