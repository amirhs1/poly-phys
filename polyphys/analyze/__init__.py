"""
Analyze Module --- :mod:`polyphys.analyze`
==========================================


The `polyphys.analyze` module provides a suite of tools for analyzing molecular
simulation data. It includes utilities for statistical analysis, distance
calculations, clustering, and more.

Submodules
==========
- `analyzer`: Core analysis routines.
- `clusters`: Functions for detecting and analyzing clusters.
- `contacts`: Utilities for calculating contact matrices and related
  statistics.
- `correlations`: Tools for computing correlations in time and space.
- `distributions`: Functions for generating and analyzing distributions.
- `distances`: Distance and angle calculations, with support for periodic
  boundary conditions.
- `helper`: Miscellaneous helper functions.
- `measurer`: Tools for geometric and structural measurements.
"""

from polyphys.analyze import (
    measurer,
)

__all__ = [
    'measurer'
]
