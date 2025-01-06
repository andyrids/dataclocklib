"""
------------
Example Name
------------

A short example showcasing how to use the library. The docstrings will be
converted to RST by sphinx-gallery.
"""

import matplotlib.pyplot as plt
import numpy as np

from dataclocklib import example_function

# make some fake data
rng = np.random.default_rng(0)
data = rng.standard_normal((1000, 2))

fig, ax = plt.subplots()
scatter = example_function(ax, data, above_color="b")
plt.show()
