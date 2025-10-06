"""\
==========================================================
:mod:`polyphys.manage.types`
==========================================================

This :mod:`~polyphys.manage.types` module provides type definitions and type
aliases to support clarity, type safety, and documentation for the data
structures and classes used within :mod:`polyphys` package.

Custom types here represent domain-specific concepts such as `PropertyT`,
`EntityT`, and `DirectionT`, which clarify the usage of fundamental
types like `str`, `int`, and `bool`.

Composite types such as tuples and dictionaries (`TimeSeriesT`,
`NonScalarHistT`, etc.) group related information for structured
data handling.

This module also defines type aliases for input sources and simulation
metadata.
"""
from typing import (
    TextIO, IO, Any, Literal, TYPE_CHECKING, TypeVar, TypeAlias, Iterable,
    Sequence
    )
from gzip import GzipFile
from os import PathLike
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .parser import ParserBase

ParserInstanceT = TypeVar("ParserInstanceT", bound="ParserBase")  # instance
ParserClassT: TypeAlias = type[ParserInstanceT]  # class constructor


Pathish = str | PathLike[str]
PathIter = Iterable[Pathish]
PathGroup = Sequence[Pathish]            # e.g., tuple[Path, ...]
PathGroupIter = Iterable[PathGroup]      # e.g., list[tuple[Path, ...]]
AnyPathIter = PathIter | PathGroupIter   # union for user-facing APIs

# --- Basic Type Aliases ---
AxisT = Literal[0, 1, 2]
BinT = Literal['ordinary', 'nonnegative', 'periodic']
DirectionT = Literal['x', 'y', 'z', 'r', 'theta']
EnsembleName = str
EntityT = str
GeometryT = Literal['cubic', 'cylindrical']
GroupT = str
HasEdgeT = bool
LineageT = Literal['segment', 'whole', 'ensemble_long', 'ensemble', 'space']
WholeRelationT = Literal['histogram', 'tseries', 'bin_edge']
PhaseT = Literal['simAll', 'simCont', 'log', 'trj', 'probe', 'analysis',
                 'viz', 'galaxy']
PlaneT = Literal['xy', 'yz', 'zx']
PrimitiveLineageT = Literal['segment', 'whole']
PropertyT = str
StageT = Literal['segment', 'wholeSim', 'ens', 'ensAvg', 'space', 'galaxy']
TopologyT = str
WholeName = str

# --- Composite Data Structures ---
EdgeDataT = dict[str, np.ndarray]
EdgeT = tuple[DirectionT, GroupT]
EnsembleT = tuple[EnsembleName, np.ndarray | pd.DataFrame]
FreqDataT = dict[str, np.ndarray]
HistogramT = tuple[DirectionT, EntityT, GroupT]
HnsStatdictT = dict[str, list[int] | np.ndarray]
NonScalarHistT = tuple[PropertyT, EntityT, GroupT, AxisT]
NonScalarMatT = tuple[PropertyT, EntityT, GroupT]
TimeSeriesT = tuple[PropertyT, EntityT, GroupT]
WholeT = dict[WholeName, np.ndarray | pd.DataFrame]

# --- IO Types ---
InputType = GzipFile | TextIO | IO[Any]
