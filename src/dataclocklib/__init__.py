#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Andrew Ridyard.
# Distributed under the terms of the Modified BSD License.

# Must import __version__ first to avoid errors importing this file during the build process.
# See https://github.com/pypa/setuptools/issues/1724#issuecomment-627241822

from importlib.metadata import version
__version__ = version("dataclocklib")