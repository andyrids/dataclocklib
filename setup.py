from importlib.metadata import version
from setuptools import setup

__version__ = version("dataclocklib")

setup_args = dict(
    version=__version__,
)

if __name__ == "__main__":
    setup(**setup_args)
