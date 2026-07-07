"""\
==========================================================
:mod:`polyphys.manage.organizer`
==========================================================
The :mod:`polyphys.manage.organizer` module provides tools for organizing and
aggregating simulation data based on filename patterns, file structures, and
processing stages. It facilitates handling segmented data, grouping experiments
into ensembles, consolidating ensembles into spaces, and organizing spaces into
galaxies for high-level analysis.

The module is tailored for use with LAMMPS and similar simulation tools, where
it automates data consolidation across hierarchical levels. It is particularly
useful for organizing data generated in high-throughput simulations, enabling
efficient post-processing, ensemble averaging, and preparation of data for
statistical and comparative analysis.

Terminology
===========
The following terms are used throughout the module to structure and organize
simulation output data:

- **Experiment**: A complete simulation run, typically using tools like LAMMPS,
  representing a specific set of conditions.

- **Lineage**: A hierarchical structure for organizing data across different
  levels of aggregation:

    - **Segment**: A portion of data from a long simulation.
    - **Whole** (Instance): A complete dataset made by combining multiple
      segments.
    - **Ensemble**: A set of wholes with identical macroscopic properties but
      varied initial conditions (e.g., different random seeds).
    - **Space**: A group of ensembles with identical attributes except for one,
      which systematically varies across the space.
    - **Galaxy**: The highest level of aggregation, representing a collection
      of spaces with differing attributes.

- **Attributes**: Macroscopic characteristics such as box size or polymer
  topology that remain constant within a space.

- **Properties**: Physical features measured during the simulation, like
  density or radius of gyration, with optional associated measures (e.g.,
  autocorrelation).

- **Species**: The particle or molecular entity type, such as 'Mon' for
  monomers, each with specific behaviors or attributes as needed.

- **Phases**: Directories organized by processing phase, such as `logs`,
  `trjs` (trajectories), `probe` (probing data), `analysis` (post-processing),
  and `viz` (visualization-ready data).

- **Stages**: Aggregation levels within each phase, e.g., `wholeSim`, `ens`,
  or `ensAvg`.

Naming Conventions
==================
File and directory naming conventions in the `organizer` module provide
structure to data organization across phases and lineage levels. Each name
pattern encodes metadata about lineage, phase, group, property, and processing
stage. Example patterns include:

- `whole|segment.group.run.lammpstrj` for trajectory files.
- `lineage-phase-group-prop_stage.extension` for processed files.

Dependencies
============
- `numpy`: For numerical operations on arrays, such as summing or concatenating
  segments.
- `pandas`: For handling tabular data and DataFrames used in data aggregation.

Functions
=========
.. autofunction:: create_fullname
.. autofunction:: save_artifact
.. autofunction:: make_database
.. autofunction:: whole_from_segments
.. autofunction:: whole_from_file
.. autofunction:: whole_from_dist_mat_t
.. autofunction:: ens_from_bin_edge
.. autofunction:: ens_from_vec
.. autofunction:: ens_from_mat_t
.. autofunction:: ens_from_df
.. autofunction:: ensemble
.. autofunction:: ens_avg_from_bin_edge
.. autofunction:: ens_avg_from_ndarray
.. autofunction:: ens_avg_from_df
.. autofunction:: ensemble_avg
.. autofunction:: children_stamps
.. autofunction:: parents_stamps
.. autofunction:: find_unique_properties
.. autofunction:: space_tseries
.. autofunction:: space_hists

Usage
=====
The :mod:`organizer` module streamlines organization and aggregation of
simulation data. Users can assemble segmented data, organize data into
ensembles, and generate ensemble-averaged data at various aggregation stages.
Its multi-level organization supports comparative analysis across varied
conditions within spaces and galaxies.

Examples
========
Example of grouping segments into a whole (histogram):

>>> segments = [("segment1.npy",), ("segment2.npy",)] # doctest: +SKIP
>>> whole_data = whole_from_segments( # doctest: +SKIP
                                     'density', segments, parser=my_parser,
                                     geometry='cubic', group='bug',
                                     topology='linear', relation='histogram')

Creating an ensemble from multiple wholes:

>>> wholes = { # doctest: +SKIP
...     'whole1': np.array([1, 2, 3]),
...     'whole2': np.array([2, 3, 4]),
... }
>>> ensemble_data = ensemble( # doctest: +SKIP
                             'density', wholes, parser=my_parser,
                             geometry='cubic', group='bug',
                             topology='linear', whole_type='vector')

Notes
=====
- Filenames must follow consistent patterns for the `organizer` module to
  accurately parse and organize data.
- This module supports only cylindrical, slit, or cubic geometries.
- Functions in this module rely on consistent naming conventions for groups,
  properties, species, and stages.
"""

import warnings
import logging
from typing import Callable, Literal, Sequence, Iterator
import os
import pathlib
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from .types import (
    Pathish,
    PathGroup,
    AnyPathIter,
    GroupT,
    PropertyT,
    ParserInstanceT,
    ParserClassT,
    StageT,
    PhaseT,
    WholeRelationT,
    PrimitiveLineageT,
)
from .utils import invalid_keyword, sort_filenames, round_up_nearest

logger = logging.getLogger(__name__)


def create_fullname(name: str, group: GroupT, prop: PropertyT) -> str:
    """
    Create a structured filename based on a base name, particle group,
    and property.

    Parameters
    ----------
    name : str
        The base name.
    group : :py:data:`GroupT`
        Particle group (e.g., 'bug' or 'nucleoid').
    prop : :py:data:`PropertyT`
        Physical property name (e.g., 'density').

    Returns
    -------
    str
        A string that combines `name`, `group`, and `prop` in a
        hyphen-separated format.

    Examples
    --------
    >>> create_fullname('sample', 'bug', 'density') # doctest: +SKIP
    'sample-bug-density'
    """
    return "-".join([name, group, prop])


def save_artifact(
    filename: str,
    data: np.ndarray | pd.DataFrame | dict[str, np.ndarray],
    save_to: Pathish,
) -> None:
    """
    Save the `data` to a specified file format, allowing structured file
    naming.

    Parameters
    ----------
    filename : str
        The output file name.
    data : np.ndarray | pd.DataFrame | dict[str, np.ndarray]
        Data to be saved. Accepted types:

        - `np.ndarray`: Saved as a .npy file.
        - `pd.DataFrame`: Saved as a .csv file.
        - `dict` of `np.ndarray`: Each entry saved as a separate .npy file
        with suffix.

    save_to : str
        Path to the directory where the file will be saved.

    Raises
    ------
    TypeError
        If `data` type does not match any of the supported formats.

    Examples
    --------
    Save a numpy array to a `.npy` file in the specified directory:

    >>> save_artifact( # doctest: +SKIP
    ...     'output',
    ...     np.array([1, 2, 3]),
    ...     'density',
    ...     'all',
    ...     "/data/",
    ... )

    Save a DataFrame to a `.csv` file:

    >>> import pandas as pd # doctest: +SKIP
    >>> df = pd.DataFrame({'A': [1, 2, 3]}) # doctest: +SKIP
    >>> save_artifact( # doctest: +SKIP
    ...     'output',
    ...     df,
    ...     'density',
    ...     'all',
    ...     "/data/",
    ... )
    """
    file_path = os.path.join(save_to, filename)

    # Save data based on type
    if isinstance(data, pd.DataFrame):
        data.to_csv(f"{file_path}.csv", index=False)
    elif isinstance(data, np.ndarray):
        np.save(f"{file_path}.npy", data)
    elif isinstance(data, dict):
        for prop_key, prop_data in data.items():
            _, prop_measure = prop_key.split("-")
            np.save(f"{file_path}-{prop_measure}.npy", prop_data)
    else:
        raise TypeError(
            f"Unsupported data type {type(data).__name__}."
            + "Expected pd.DataFrame, np.ndarray, or dict of np.ndarray."
        )


def make_database(
    old_database: Pathish,
    phase: PhaseT | str,
    stage: StageT | str,
    group: GroupT | None,
) -> str:
    """
    Create a new directory path based on the provided `old_database` path and
    specified parameters (`phase`, `group`, and `stage`). If the directory does
    not already exist, it will be created.

    The `old_database` is expected to follow a structured naming convention:
    `prefix-old_phase-old_group-old_stage`, where each part represents an
    aspect of the directory's content. This base structure helps generate the
    new path.

    The newly constructed directory name follows:
        `prefix-phase-group-stage`

    Parameters
    ----------
    old_database : Pathish
        Path to the reference directory whose structure will be used as a base.
    phase : {'simAll', 'simCont', 'log', 'trj', 'probe', 'analysis', 'viz',
            'galaxy'}
        The new phase name for the directory, specifying its role in
        the workflow.
    stage : {'segment', 'wholeSim', 'ens', 'ensAvg', 'space', 'galaxy'}
        The stage of processing or data type represented in the new directory.
    group : GroupT
        Particle group related to the data in the directory.

    Returns
    -------
    str
        The path to the newly created directory or the existing path if it
        already exists.

    Raises
    ------
    ValueError
        If any parameter is not in its list of accepted values.

    Examples
    --------
    Examples of directory transformation:
        - Input: `old_database = root/parent1/.../parentN/old_phase/old_dir`
        - Result: `new_database = root/parent1/.../parentN/phase/new_dir`

    Construct a new directory path based on an existing database path:

    >>> make_database( # doctest: +SKIP
                     '/root/data/old_analysis/simulations', 'analysis',
                     'wholeSim', 'bug')
    '/root/data/analysis/prefix-bug-wholeSim/'
    """
    invalid_keyword(
        phase,
        ["simsAll", "simsCont", "logs", "trjs", "probes", "analysis", "viz",
         "galaxy", "allInOne"],
    )

    old_path = pathlib.Path(old_database)

    # Common prefix derived from old_database to create new directory name
    prefix = old_path.parts[-1].split("*")[0].split("-")[0]
    new_directory = "-".join([part for part in [prefix, group, stage] if part])

    # Construct the new database path in the same parent directory level as
    # the old one
    new_database_parts = list(old_path.parts[:-2]) + [phase, new_directory]
    new_database = pathlib.Path(*new_database_parts)

    # Avoid double slashes at the start of the path
    if str(new_database).startswith("//"):
        new_database = pathlib.Path(str(new_database)[1:])

    # Create the new directory if it doesn't already exist
    try:
        new_database.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        print(error)
        print("Files are saved/overwritten in an existing directory.")

    return str(new_database) + "/"


def _as_path(item: Pathish | PathGroup) -> Path:
    """Normalize a single item into a pathlib.Path.

    Accepts either:
      - a bare pathish (str, Path, os.PathLike)
      - a sequence of pathish values (we take the first element)
    """
    # Bare pathish?
    if isinstance(item, (str, os.PathLike)):
        return Path(item)

    # Sequence of pathish (e.g., tuple[Path, ...] from sort_filenames)
    if isinstance(item, Sequence) and item:
        first = item[0]
        if isinstance(first, (str, os.PathLike)):
            return Path(first)

    raise TypeError(f"Unsupported path item: {type(item)!r} -> {item!r}")


def _normalize_paths(paths: AnyPathIter) -> Iterator[Path]:
    """Yield normalized Path objects from either a flat or grouped iterable."""
    for item in paths:
        # item is either Pathish or PathGroup; _as_path handles both
        yield _as_path(item)


def _assert_equal_bin_edges(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """
    Ensure all input arrays of bin edges are identical.

    Parameters
    ----------
    arrays : Sequence[np.ndarray]
        Sequence of 1D arrays representing bin edges from different wholes
        or segments.

    Returns
    -------
    np.ndarray
        A copy of the first array if all arrays are identical.

    Raises
    ------
    ValueError
        If any array differs from the first one.
    """
    ref = arrays[0]
    if not all(np.array_equal(ref, a) for a in arrays[1:]):
        raise ValueError("Bin edges are not identical across all arrays.")
    return ref.copy()


def whole_from_segments(
    prop: PropertyT,
    segments: AnyPathIter,
    parser: ParserClassT,
    group: GroupT,
    relation: WholeRelationT,
    save_to: Pathish | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate 'whole' arrays by combining data segments for a specified
    property of a particle `group` within a defined `geometry`. The combination
    method is determined by the `relation` parameter.

    Parameters
    ----------
    prop: :py:data:`PropertyT`
        The physical property to process (e.g., 'density').
    segments : :py:data:`AnyPathIter`
        An iterable of paths to segment files (e.g., to CSV files) for the
        `prop_`. Each path should point to a single segment file, which will be
    parser : :py:class:`ParserClassT`
        A parser instance from `PolyPhys.manage.parser` module that interprets
        file paths or names and extracts attributes such as the `whole` name.
    group : :py:data:`GroupT`
        Type of the particle group.
    relation : :py:data:`WholeRelationT`
        Specifies how to combine segments into a whole. Accepted values:

        - 'histogram'
            'segment' is an N-dimensional histogram-like file, so the segments
            of a 'whole' should be sum along "axis=0" in numpy lingo. In the
            two-dimensional space, a histogram has a (x_nbins, y_nbins) shape.

        - 'tseries'
            'segment' is a time-series-like file, so the segments of a 'whole'
            should be concatenated vertically (along "axis=0" in pandas lingo).

        - 'bin_edge'
            'segment' is a bin-edge file. All the the segments are similar,
            so one 'whole' bin-edge file is created by picking one of
            segments.

    save_to : str, optional
        Directory path where the output files are saved, if specified.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with whole names as keys and the combined whole data as
        numpy arrays.

    Raises
    ------
    ValueError
        If `relation` have invalid values.

    Notes
    -----
    Refer to the 'organizer' module documentation for additional context on the
    terms and keywords such as 'geometry' and 'group'.

    Examples
    --------
    >>> whole_data = whole_from_segment( # doctest: +SKIP
            'density',
            [('path/to/segment1.npy',), ('path/to/segment2.npy',)],
            SumRuleCyl,
            'bug',
            relation='histogram',
            save_to='/output/path'
        )
    """
    # Combine segments into wholes based on segment names and relation type
    grouped_segments: dict[str, list[np.ndarray]] = {}
    for seg_path in _normalize_paths(segments):
        seg_info = parser(seg_path, "segment", group)
        whole_name = getattr(seg_info, "whole")
        grouped_segments.setdefault(whole_name, []).append(np.load(seg_path))

    # Define reduction strategies for each relation type
    reducers: dict[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = {
        "histogram": lambda arrs: np.sum(arrs, axis=0),
        "tseries": lambda arrs: np.concatenate(arrs, axis=0),
        "bin_edge": _assert_equal_bin_edges,
    }
    reducer = reducers[relation]

    # Apply the appropriate mapping function to merge each group of segments
    # into a whole.
    wholes: dict[str, np.ndarray] = \
        {name: reducer(vals) for name, vals in grouped_segments.items()}

    if save_to is not None:
        for name, data in wholes.items():
            save_artifact(create_fullname(name, group, prop), data, save_to)

    return wholes


def whole_from_file(
    whole_paths: AnyPathIter,
    parser: ParserClassT,
    group: GroupT,
) -> dict[str, np.ndarray]:
    """
    Load data from 'whole' files for a specified physical property of a
    particle `group` within a given `geometry`. Each file path in `whole_paths`
    points to a single whole file.

    Parameters
    ----------
    whole_paths : AnyPathIter
        List of tuples where each tuple contains the path to a single file
        representing a whole dataset for the `prop_`. Each file is loaded
        and processed independently.
    parser : ParserClassT
        A parser instance from `PolyPhys.manage.parser` module that interprets
        file paths and extracts attributes such as the `whole` name.
    group : GroupT
        The particle group to which the data pertains.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary where keys are the `whole` names (derived from each file
        path)and values are numpy arrays containing the loaded data.

    Notes
    -----
    This function assumes that each file in `whole_paths` contains data related
    to a complete simulation or experiment segment, referred to as a "whole."
    For further information on terminology, see the `organizer` module
    documentation.

    Examples
    --------
    Load and organize whole data for a specific property from a list of file
    paths:

    >>> whole_data = whole_from_file( # doctest: +SKIP
            whole_paths=[('path/to/whole1.npy',), ('path/to/whole2.npy',)],
            TransFociCyl,
            'bug'
        )
    """
    wholes: dict[str, np.ndarray] = {}
    for whole_path in _normalize_paths(whole_paths):
        whole_info = parser(whole_path, "whole", group)
        wholes[getattr(whole_info, "whole")] = np.load(whole_path)

    return wholes


def whole_from_dist_mat_t(
    whole_paths: list[tuple[str]],
    parser: ParserClassT,
    group: GroupT,
    func: Callable[[str, ParserInstanceT],
                   tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
) -> tuple[
    dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]
]:
    """
    Load pair distance matrix 'whole' data from given file paths, processing
    specific distance-related physical properties (i.e., frequencies, radial
    distribution functions, time series).

    Depending on the project, you must supply the `func` argument that knows
    how to process the distance matrix for your specific project.

    Parameters
    ----------
    whole_paths : list[tuple[str]]
        List of tuples where each tuple contains the path to a file
        representing a 'whole' dataset (typically a NumPy `.npy` distance
        matrix).
    parser : ParserClassT
        A parser class used to create parser instances for each file. The
        parser provides attributes such as `whole` (name identifier) and
        simulation metadata.
    group : GroupT
        The particle group to which the data pertains (e.g., 'bug', 'all').
    func : Callable[[str, ParserInstanceT],
                    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Project-specific processing function. It must accept two arguments:

        - `whole_path` (str): path to the distance matrix file.
        - `whole_info` (ParserInstanceT): parser instance for the file.

        It must return a tuple of three pandas DataFrames corresponding to
        frequency/histogram data, radial distribution data, and time-series
        data.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame],
          dict[str, pd.DataFrame]]
        Three dictionaries with whole names as keys and DataFrames as values:

        - Frequency/histogram DataFrames
        - Radial distribution function (RDF) DataFrames
        - Time series DataFrames

    Raises
    ------
    ValueError
        If `geometry`, `group`, or `topology` are invalid.

    Examples
    --------
    Load foci distance matrix data with a project-specific processor:

    >>> freqs, rdfs, tseries = whole_from_dist_mat_t( # doctest: +SKIP
    ...     [('path/to/whole1.npy',), ('path/to/whole2.npy',)],
    ...     my_parser_class,
    ...     'bug',
    ...     whole_dist_mat_foci
    ... )
    """
    wholes_freqs: dict[str, pd.DataFrame] = {}
    wholes_rdfs: dict[str, pd.DataFrame] = {}
    wholes_tseries: dict[str, pd.DataFrame] = {}
    for whole_path in whole_paths:
        whole_info = parser(whole_path[0], "whole", group)
        whole_freqs, whole_rdfs, whole_tseries = \
            func(whole_path[0], whole_info)
        whole_name = getattr(whole_info, "whole")
        wholes_freqs[whole_name] = whole_freqs
        wholes_rdfs[whole_name] = whole_rdfs
        wholes_tseries[whole_name] = whole_tseries
    return wholes_freqs, wholes_rdfs, wholes_tseries


def ens_from_bin_edge(
    ens_data: tuple[str, dict[str, np.ndarray]],
) -> tuple[str, np.ndarray]:
    """
    Combine multiple bin edge arrays into a single unique array for an
    'ensemble' by removing duplicates. This is typically used when bin edges
    are identical across 'whole' datasets in an ensemble.

    Parameters
    ----------
    ens_data : tuple[str, dict[str, np.ndarray]]
        A tuple where the first element is the ensemble name (str) and the
        second element is a dictionary of bin edge arrays from different
        'whole' datasets within that ensemble. Within the dictionary, keys are
        whole names while values are whole bin edges.

    Returns
    -------
    tuple[str, np.ndarray]
        A tuple where the first element is the ensemble name, and the second
        is a 1D numpy array of unique bin edges.

    Examples
    --------
    >>> ensemble_bin_edges = ens_from_bin_edge( # doctest: +SKIP
            ('ensemble1', {'whole1': np.array([0.1, 0.2, 0.2, 0.3],
             'whole2': np.array([0.1, 0.2, 0.2, 0.3]})
        )
    >>> print(ensemble_bin_edges) # doctest: +SKIP
    ('ensemble1', array([0.1, 0.2, 0.3]))
    """
    ensemble_name, whole_edges = ens_data
    unique_bin_edges = np.unique(list(whole_edges.values()))
    return ensemble_name, unique_bin_edges


def ens_from_vec(
    ens_data: tuple[str, dict[str, np.ndarray]],
) -> tuple[str, pd.DataFrame]:
    """
    Convert a dictionary of 1D arrays (representing different 'whole'
    datasets) into a single ensemble-level DataFrame. This is useful when each
    'whole' dataset in the ensemble contains a vector (1D array) of
    measurements.

    Parameters
    ----------
    ens_data : tuple[str, dict[str, np.ndarray]]
        A tuple where the first element is the ensemble name, and the second
        is a dictionary mapping each 'whole' dataset name to a 1D numpy array
        of measurements.

    Returns
    -------
    tuple[str, pd.DataFrame]
        A tuple where the first element is the ensemble name, and the second
        is a pandas DataFrame. Each column represents a 'whole' dataset, and
        rows contain corresponding vector values from each dataset.

    Examples
    --------
    >>> ens_name, ensemble_df = ens_from_vec( # doctest: +SKIP
            ('ensemble1', {'whole1': np.array([0.1, 0.2, 0.3]),
                           'whole2': np.array([0.4, 0.5, 0.6])})
        )
    >>> print(ens_name) # doctest: +SKIP
    'ensemble1'
    >>> print(ensemble_df) # doctest: +SKIP
       whole1  whole2
    0     0.1     0.4
    1     0.2     0.5
    2     0.3     0.6
    """
    ensemble_name, whole_vectors = ens_data
    ensemble_df = pd.DataFrame.from_dict(whole_vectors, orient="columns")
    return ensemble_name, ensemble_df


def ens_from_mat_t(
    ens_data: tuple[str, dict[str, np.ndarray]],
) -> tuple[str, np.ndarray]:
    """
    Combine multiple 2D arrays into a single 3D array for a specified
    'ensemble', where each 2D array corresponds to a 'whole' dataset in the
    ensemble.

    Parameters
    ----------
    ens_data : tuple[str, dict[str, np.ndarray]]
        A tuple where the first element is the ensemble name (str) and the
        second element is a dictionary with 'whole' names as keys and 2D numpy
        arrays as values.

    Returns
    -------
    tuple[str, np.ndarray]
        A tuple where the first element is the ensemble name and the second
        is a 3D numpy array formed by stacking the 2D arrays along the first
        axis.

    Examples
    --------
    >>> ens_data = ( # doctest: +SKIP
            'ensemble1',
            {'whole1': np.array([[1, 2], [3, 4]]),
            'whole2': np.array([[5, 6], [7, 8]])}
        )
    >>> combined_array = ens_from_mat_t(ens_data) # doctest: +SKIP
    >>> print(combined_array) # doctest: +SKIP
    ('ensemble1', array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    """
    ensemble_name, whole_matrices = ens_data
    combined_matrix = np.stack(list(whole_matrices.values()), axis=0)
    return ensemble_name, combined_matrix


def ens_from_df(
    ens_data: tuple[str, dict[str, pd.DataFrame]],
) -> tuple[str, pd.DataFrame]:
    """
    Average multiple DataFrames in an ensemble by grouping them into a single
    averaged DataFrame. Each DataFrame corresponds to a 'whole' dataset in the
    ensemble.

    In a 'whole' DataFrame, headers are 'elements' of a 'matrix' or 2D
    quantity and columns are the values of a given property.

    In each 'whole' DataFrame, headers represent 'elements' of a 2D quantity
    (matrix-like structure), and columns contain the values for a specified
    property. Each 'whole' DataFrame may be either a 'histogram' or a
    'time-series':

    - In a 'histogram' DataFrame, an additional column, `bin_center`, holds
      bin center values that are invariant under averaging and are retained
      in the output DataFrame. The length of histogram DataFrame is equal to
      the number of bins.

    - In a 'time-series' DataFrame, headers (columns) are elements of a matrix,
      and rows represent the values of each element over time. The length of
      timeseries DataFrame is equal to the number of time frames.


    Parameters
    ----------
    ens_data : tuple[str, dict[str, pd.DataFrame]]
        A tuple where the first element is the ensemble name and the second
        is a DataFrame containing the mean of all input DataFrames, grouped
        by index.

    Returns
    -------
    tuple[str, pd.DataFrame]
        A tuple with:
            - The ensemble name (str).
            - A DataFrame with the averaged values across all 'whole'
            DataFrames.

    Notes
    -----
    A clear distinction exists between ensembles of 'vector' or 'matrix' types
    and ensembles of 'dataframe' type. In vector or matrix types, each column
    in the resulting ensemble DataFrame is named after a 'whole' and represents
    that whole's values for a specific property. In contrast, for DataFrame
    type ensembles, the columns are elements of a 2D quantity (e.g., matrix or
    time-series values) and hold averaged values across all whole DataFrames.

    This function assumes that averaging non-element headers, such as
    'bin_center,' does not change their values, and these are carried forward
    without modification.

    Examples
    --------
    >>> df1 = pd.DataFrame( # doctest: +SKIP
    ...     {'A': [1, 2, 3], 'B': [4, 5, 6]}
    ... )
    >>> df2 = pd.DataFrame( # doctest: +SKIP
    ...     {'A': [2, 3, 4], 'B': [5, 6, 7]}
    ... )
    >>> ens_data = ( # doctest: +SKIP
    ...     'ensemble1', {'whole1': df1, 'whole2': df2}
    ... )
    >>> averaged_df = ens_from_df(ens_data) # doctest: +SKIP
    >>> print(averaged_df) # doctest: +SKIP
    ('ensemble1',
       A    B
    0  1.5  4.5
    1  2.5  5.5
    2  3.5  6.5)
    """
    ensemble_name, whole_data = ens_data
    ensemble_df = pd.concat(list(whole_data.values())).groupby(level=0).mean()
    return ensemble_name, ensemble_df


def ensemble(
    prop: PropertyT,
    wholes: dict[str, np.ndarray] | dict[str, pd.DataFrame],
    parser: ParserClassT,
    group: GroupT,
    whole_type: Literal["vector", "matrix", "dataframe", "bin_edge"],
    whole_edges: dict[str, np.ndarray] | None = None,
    save_to: Pathish | None = None,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """
    Generate an ensemble by  merging 'whole' data arrays or DataFrames
    representing a specified `prop_` of a particle `group` within a
    defined `geometry`. The `whole_type` determines the structure of each
    'whole' and the merging approach.

    Parameters
    ----------
    prop: PropertyT
        The physical property for which the ensemble is generated (e.g.,
        'density').
    wholes : dict[str, np.ndarray] | dict[str, pd.DataFrame]
        Dictionary containing 'whole' names as keys and their corresponding
        data as values, represented as numpy arrays or DataFrames.
    parser : ParserClassT
        An instance of the parser class to infer attributes like ensemble name
        from each 'whole'.
    group : GroupT
        Type of the particle group being processed.
    whole_type : {'vector', 'matrix', 'dataframe', 'bin_edge'}
        Specifies the type of each 'whole' and defines the merging method:

        - 'vector': A 1D numpy array, such as a time series or histogram.
        - 'matrix': A 2D numpy array, such as a gyration tensor.
        - 'dataframe': A DataFrame representing either a time series (with
          headers as matrix elements and rows as values over time) or a
          histogram (headers as elements, rows as counts or frequencies,
          and an additional 'bin_center' column for bin centers).
        - 'bin_edge': A 1D array representing bin edges. Assumes all bin edges
          are identical and selects one to represent the ensemble.

    whole_edges : dict[str, np.ndarray], optional
        Dictionary with 'whole' names as keys and bin edge arrays as values,
        useful when `whole_type` is 'vector' and bin edges are shared across
        wholes in the ensemble.
    save_to : str, optional
        Directory path where output files are saved, if specified.

    Returns
    -------
    dict[str, np.ndarray | pd.DataFrame]
        Dictionary where keys are ensemble names and values are ensemble data
        as numpy arrays or DataFrames.

    Raises
    ------
    ValueError
        If any parameter has an invalid value.

    Notes
    -----
    Different averaging methods apply based on `whole_type`. When `whole_type`
    is "DataFrame," the resulting DataFrame averages the values of each matrix
    element across wholes, retaining headers that are mean-invariant (e.g.,
    'bin_center'). For 'vector' and 'matrix' types, columns represent
    individual wholes in the ensemble DataFrame.

    Examples
    --------
    >>> wholes = { # doctest: +SKIP
            'whole1': np.array([1, 2, 3]),
            'whole2': np.array([2, 3, 4])
        }
    >>> ensembles = ensemble( # doctest: +SKIP
            'density',
            wholes,
            HnsCyl,
            'bug',
            whole_type='vector',
            save_to="/output/path"
        )
    """
    invalid_keyword(whole_type, ["vector", "matrix", "dataframe", "bin_edge"])

    merging_func: dict[str, Callable] = {
        "vector": ens_from_vec,
        "matrix": ens_from_mat_t,
        "dataframe": ens_from_df,
        "bin_edge": ens_from_bin_edge,
    }
    # Collecting wholes for each ensemble
    ens_wholes: dict[str, dict[str, np.ndarray | pd.DataFrame]] = {}
    bin_centers: dict[str, np.ndarray] = {}
    if whole_edges is not None:
        for whole_name, whole_data in wholes.items():
            whole_info = parser(whole_name, "whole", group)
            ens_name = getattr(whole_info, "ensemble_long")
            if ens_name not in ens_wholes:
                ens_wholes[ens_name] = {}
            ens_wholes[ens_name][whole_name] = whole_data
            bin_centers[ens_name] = 0.5 * (
                whole_edges[whole_name][:-1] + whole_edges[whole_name][1:]
            )
    else:
        for whole_name, whole_data in wholes.items():
            whole_info = parser(whole_name, "whole", group)
            ens_name = getattr(whole_info, "ensemble_long")
            if ens_name not in ens_wholes:
                ens_wholes[ens_name] = {}
            ens_wholes[ens_name][whole_name] = whole_data

    # Merging wholes into a single ensemble data object
    ensembles: dict[str, np.ndarray | pd.DataFrame] = dict(
        map(merging_func[whole_type], ens_wholes.items())
    )

    # Save ensemble data object if save_to is specified
    if save_to:
        for ens_name, data in ensembles.items():
            ens_fullname = create_fullname(ens_name, group, prop)
            save_artifact(ens_fullname, data, save_to)

    return ensembles


def ens_avg_from_bin_edge(ens_data: np.ndarray) -> np.ndarray:
    """
    Generate a unique set of bin edges for an 'ensemble' dataset based on
    the bin edges in `ens_data`. This function assumes that the bin edges
    across the ensemble are identical and selects one representative set.

    Parameters
    ----------
    ens_data : list[np.ndarray]
        A 1D array of bin edges across multiple wholes in the ensemble. The
        function assumes these bin edges are identical across all wholes.

    Returns
    -------
    np.ndarray
        A unique, representative set of bin edges for the ensemble.

    Notes
    -----
    This function does not perform any averaging or statistical calculations.
    Instead, it identifies unique bin edges across all wholes, assuming they
    are identical and discards duplicates.

    Examples
    --------
    >>> ens_data = [ # doctest: +SKIP
    ...     np.array([0.0, 0.1, 0.2]),
    ...     np.array([0.0, 0.1, 0.2]),
    ... ]
    >>> unique_bin_edges = ens_avg_from_bin_edge(ens_data) # doctest: +SKIP
    >>> print(unique_bin_edges) # doctest: +SKIP
    [0.  0.1 0.2 0.3]
    """
    return np.unique(ens_data)


def ens_avg_from_ndarray(
    ens_prop: PropertyT, ens_data: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute ensemble statistics (mean, variance, and standard error of the
    mean)across an 'ensemble' 3D array where each slice along the first axis
    (axis=0) represents a 'whole' data matrix. The statistics are calculated
    along this axis.

    Parameters
    ----------
    ens_prop : PropertyT
        The name of the property to which this 'ensemble' data belongs. Used
        as a prefix for the computed statistics.
    ens_data : np.ndarray
        A 3D numpy array where each matrix slice along axis=0 represents a
        'whole' dataset.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where keys are statistical measures (mean, variance, SEM)
        with `ens_prop` as a prefix while values are numpy arrays of the
        computed statistics across 'whole' matrices.

    Notes
    -----
    - The SEM is calculated with a degrees-of-freedom adjustment (`ddof=1`) for
      an unbiased estimate and is scaled by the square root of the number of
      wholes.
    - This function performs averaging along axis=0 for each element across
      the 'whole' data matrices.

    Examples
    --------
    >>> ens_data = np.array([ # doctest: +SKIP
            [[1, 2], [3, 4]],
            [[2, 3], [4, 5]]
        ])
    >>> ens_avg = ens_avg_from_ndarray( # doctest: +SKIP
    ...     'gyration_tensor', ens_data, exclude=[]
    ... )
    >>> print(ens_avg) # doctest: +SKIP
    {
        'gyration_tensor-mean': array([[1.5, 2.5],
                               [3.5, 4.5]]),
        'gyration_tensor-var': array([[0.5, 0.5],
                              [0.5, 0.5]]),
        'gyration_tensor-sem': array([[0.5, 0.5],
                              [0.5, 0.5]])
    }
    """
    n_wholes_sqrt = ens_data.shape[0] ** 0.5
    ens_avg = {
        f"{ens_prop}-mean": np.mean(ens_data, axis=0),
        f"{ens_prop}-var": np.var(ens_data, axis=0, ddof=1),
        f"{ens_prop}-sem": np.std(ens_data, axis=0, ddof=1) / n_wholes_sqrt,
    }
    return ens_avg


def ens_avg_from_df(
    ens_prop: PropertyT, ens_data: pd.DataFrame, exclude: list[str]
) -> pd.DataFrame:
    """
    Calculate the mean, variance, and standard error of the mean (SEM) for
    columns representing 'whole' data within an 'ensemble' DataFrame. The
    resulting DataFrame retains excluded columns (e.g., bin_center), while
    replacing 'whole' columns with computed statistics.

    Parameters
    ----------
    ens_prop : PropertyT
        The name of the property to which this 'ensemble' data belongs, used
        as a prefix for the generated statistics columns.
    ens_data : pd.DataFrame
        A DataFrame containing ensemble data. Each 'whole' dataset is in its
        own column, and any columns specified in `exclude` remain unaffected.
    exclude : list of str
        A list of column names to exclude from averaging. For example,
        `exclude` might contain 'bin_center' for histogram-type ensembles.

    Returns
    -------
    pd.DataFrame
        The input `ens_data` DataFrame with 'whole' columns replaced by
        calculated statistics: mean, variance, and SEM of each row across the
        'whole' data columns.

    Notes
    -----
    - The function averages only the 'whole' columns, excluding any specified
      in `exclude`.
    - The standard error of the mean (SEM) is computed with a
    degrees-of-freedom adjustment (ddof=1) for an unbiased estimate.

    Examples
    --------
    >>> import pandas as pd # doctest: +SKIP
    >>> ens_data = pd.DataFrame({ # doctest: +SKIP
            'bin_center': [1, 2, 3],
            'whole1': [4, 5, 6],
            'whole2': [5, 6, 7]
        })
    >>> exclude = ['bin_center'] # doctest: +SKIP
    >>> ens_avg_from_df('density', ens_data, exclude) # doctest: +SKIP
       bin_center  density-mean  density-var  density-sem
    0           1           4.5        0.5       0.5
    1           2           5.5        0.5       0.5
    2           3           6.5        0.5       0.5
    """
    wholes = [col for col in ens_data.columns if col not in exclude]

    ens_data[ens_prop + "-mean"] = ens_data[wholes].mean(axis=1)
    ens_data[ens_prop + "-var"] = ens_data[wholes].var(axis=1, ddof=1)
    ens_data[ens_prop + "-sem"] = ens_data[wholes].sem(axis=1, ddof=1)

    ens_data.drop(columns=wholes, inplace=True)
    return ens_data


def ensemble_avg(
    prop: PropertyT,
    ensembles: dict[str, np.ndarray | pd.DataFrame],
    group: GroupT,
    ens_type: Literal["dataframe", "ndarray", "bin_edge"],
    exclude: list[str] | None = None,
    save_to: Pathish | None = None,
) -> dict[str, np.ndarray] | dict[str, pd.DataFrame]:
    """
    Generate ensemble-averaged data for a specified `prop_` by averaging
    'whole' data in each 'ensemble' DataFrame or array within `ensembles`.
    Columns listed in `exclude` are omitted from averaging.

    Parameters
    ----------
    prop: PropertyT
        The physical property for which ensemble averages are calculated.
    ensembles : dict[str, np.ndarray | pd.DataFrame]
        A dictionary where each key is an ensemble name and each value is
        either a DataFrame or ndarray representing 'whole' data.
    group : GroupT
        Type of the particle group.
    ens_type : {'dataframe', 'ndarray', 'bin_edge'}
        The data format for each ensemble:

        - 'dataframe': DataFrame format, with columns as 'whole' simulations
          or items in `exclude`.
        - 'ndarray': 3D ndarray format, with each slice along axis=0
          representing a 'whole' matrix.
        - 'bin_edge': 1D ndarray format for bin edges, with bin edges assumed
          to be identical across wholes.

    exclude : list[str], default None
        List of column names or items not included in ensemble averaging,
        such as 'bin_center' for histograms.
    save_to : str, optional
        Path to the directory where the averaged ensemble data will be saved.

    Returns
    -------
    dict[str, np.ndarray] | dict[str, pd.DataFrame]
        A dictionary where each key is an ensemble name, and each value is a
        DataFrame or ndarray of ensemble-averaged data.

    Raises
    ------
    ValueError
        If `geometry`, `group`, or `ens_type` contains invalid values.

    Examples
    --------
    >>> ensembles = { # doctest: +SKIP
            'ensemble1': pd.DataFrame({'whole1': [1, 2, 3],
                                       'whole2': [2, 3, 4]}),
            'ensemble2': pd.DataFrame({'whole1': [3, 4, 5],
                                       'whole2': [4, 5, 6]})
        }
    >>> avg_ensembles = ensemble_avg( # doctest: +SKIP
            'density',
            ensembles,
            'bug',
            ens_type='DataFrame',
            exclude=['bin_center'],
            save_to="/output/path"
        )
    """
    if exclude is None:
        exclude = ["bin_center"]

    # Compute ensemble averages
    ens_avgs: dict[str, np.ndarray] | dict[str, pd.DataFrame] = {}
    if ens_type == "bin_edge":
        for ens_name, ens_data in ensembles.items():
            ens_avg = ens_avg_from_bin_edge(ens_data)  # type: ignore
            ens_avgs[ens_name] = ens_avg  # type: ignore
    elif ens_type == "ndarray":
        for ens_name, ens_data in ensembles.items():
            ens_avg = ens_avg_from_ndarray(prop, ens_data)  # type: ignore
            ens_avgs[ens_name] = ens_avg  # type: ignore
    elif ens_type == "dataframe":
        for ens_name, ens_data in ensembles.items():
            ens_avg = ens_avg_from_df(prop, ens_data, exclude)  # type: ignore
            ens_avgs[ens_name] = ens_avg  # type: ignore
    else:
        invalid_keyword(ens_type, ["dataframe", "ndarray", "bin_edge"])

    # Save averaged ensemble data if save_to path is provided
    if save_to is not None:
        save_property = prop + "-ensAvg"
        for ens_name, data in ens_avgs.items():
            ens_fullname = create_fullname(ens_name, group, save_property)
            save_artifact(ens_fullname, data, save_to)

    return ens_avgs


def children_stamps(
    stamps: list[tuple[str]],
    lineage: PrimitiveLineageT,
    save_to: Pathish | None = None,
) -> pd.DataFrame:
    """
    Combine individual 'stamp' CSV files into a single DataFrame for a given
    `lineage` in a *space*. Optionally, saves the resulting
    DataFrame to a specified directory.

    Parameters
    ----------
    stamps : list[tuple[str]]
        List of tuples where each tuple contains a single file path to a
        'stamp' CSV file.
    lineage : {'segment', 'whole'}
        Lineage type, either 'segment' or 'whole'.
    save_to : str, optional
        Directory path where the combined DataFrame is saved, if specified.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing data from all the 'stamp' files.

    Raises
    ------
    ValueError
        If `group` or `lineage` contains invalid values.

    Notes
    -----
    - It is assumed that all the stamp files have belong to the same *group* in
      a given *space*.
    - If `lineage` is 'whole', columns 'segment' and 'segment_id' are removed,
    if present, as they are redundant for whole-lineage data.

    Examples
    --------
    >>> stamps = [ # doctest: +SKIP
    ...     ('path/to/stamp1.csv',),
    ...     ('path/to/stamp2.csv',),
    ... ]
    >>> df = children_stamps(stamps, lineage='whole') # doctest: +SKIP
    """
    invalid_keyword(lineage, ["segment", "whole"])

    # Load and concatenate all stamps
    space_stamps = pd.concat(
        [pd.read_csv(stamp[0]) for stamp in stamps], ignore_index=True
    )

    # Drop specific columns for 'whole' lineage
    cols_to_drop = ["segment", "segment_id"]
    if lineage == "whole":
        try:
            space_stamps.drop(columns=cols_to_drop, inplace=True)
            warnings.warn(
                "'segment' and 'segment_id' columns are dropped when"
                " individual 'whole' stamps combined to create a single"
                " dataframe of 'whole' stamps by 'children_stamps'.",
                UserWarning,
            )
        except KeyError:
            print(f"'{cols_to_drop}' are not among columns.")

    # Save the resulting DataFrame to a CSV if specified
    if save_to is not None:
        space_name = str(space_stamps.loc[0, "space"])
        group = str(space_stamps.loc[0, "group"])
        filename = f"{space_name}-{group}-{lineage}-stamps.csv"
        space_stamps.to_csv(os.path.join(save_to, filename), index=False)

    return space_stamps


def parents_stamps(
    stamps: pd.DataFrame,
    lineage: PrimitiveLineageT,
    properties: dict[str, Callable] | None = None,
    save_to: Pathish | None = None,
) -> pd.DataFrame:
    """
    Aggregate data from child 'stamps' into a parent stamp by applying
    specified aggregation functions in a *space*. Optionally, saves the
    resulting DataFrame to a CSV file.

    Parameters
    ----------
    stamps : pd.DataFrame
        DataFrame containing child stamp data.
    lineage : {'segment', 'whole'}
        Lineage type, either 'segment' or 'whole'.
    properties : dict of str -> Callable, optional
        Dictionary specifying aggregation functions for specific properties.
    save_to : str, optional
        Directory path where the parent DataFrame is saved, if specified.

    Returns
    -------
    pd.DataFrame
        Aggregated parent DataFrame.

    Raises
    ------
    ValueError
        If `group`, or `lineage` contain invalid values.

    Notes
    -----
    - It is assumed that all the stamp files have belong to the same *group* in
      a given *space*.
    - If `lineage` is 'segment', stamps correspond to individual segments,
      each with a unique `segment_id`.
    - If `lineage` is 'whole', stamps correspond to whole simulations, each
      identified by an `ensemble_id`.

    Examples
    --------
    >>> df = pd.DataFrame( # doctest: +SKIP
    ...     {'segment': [1, 2], 'value': [10, 20]}
    ... )
    >>> parent_df = parents_stamps(df, group='bug', # doctest: +SKIP
                                   lineage='segment')
    """
    invalid_keyword(lineage, ["segment", "whole"])

    # Base aggregation functions for columns
    base_columns = list(stamps.columns)
    try:
        base_columns.remove("lineage_name")
        base_columns.remove(lineage)
    except ValueError:
        print(
            f"'lineage_name' and '{lineage}'"
            " columns are not among in stamps column:"
            f"'{base_columns}', they are probably removed in"
            " a previous call of 'parents_stamps' function."
        )
    agg_funcs: dict[str, str | Callable] = {
        col: "last" for col in base_columns
    }
    if properties is not None:
        agg_funcs.update(properties)

    # Define grouping column and lineage-specific aggregations
    if lineage == "whole":
        parent_groupby = "ensemble_long"
        agg_funcs.update({"ensemble_id": "count", "n_frames": "last"})
    else:
        parent_groupby = "whole"
        agg_funcs.update({"segment_id": "count", "n_frames": "sum"})

    # Perform the groupby and aggregate
    agg_funcs.pop(parent_groupby)
    parent_stamps = stamps.groupby(parent_groupby).agg(agg_funcs).reset_index()

    # Rename and drop columns based on lineage
    if lineage == "whole":
        parent_stamps.rename(
            columns={"ensemble_id": "n_ensembles"}, inplace=True
        )
        # If the 'whole' stamps are generated directly in the 'probe' phase,
        # then 'segment' and 'segment_id' columns are "N/A" and are removed
        # from the list of stamps columns that are added to the parents
        # stamps.
        # There is no need to have the 'n_segment' column in the parent
        # stamps, so it is removed. The 'whole' stamps directly generated
        # in the 'probe' phase do not have such a column, but those generated
        # from 'segment' stamps have.
        # Dropping redundant columns silently:
        parent_stamps.drop(
            columns=["n_segments", "segment_id", "segment"],
            inplace=True,
            errors="ignore",
        )
    else:
        parent_stamps.rename(
            columns={"segment_id": "n_segments"}, inplace=True
        )

    if save_to is not None:
        space_name = str(parent_stamps.loc[0, "space"])
        group = str(parent_stamps.loc[0, "group"])
        file_suffix = "ensAvg" if lineage == "whole" else "whole"
        filename = f"{space_name}-{group}-stamps-{file_suffix}.csv"
        parent_stamps.to_csv(os.path.join(save_to, filename), index=False)

    return parent_stamps


def find_unique_properties(
    filepath: str,
    prop_idx: int,
    extensions: list[str],
    drop_properties: list[str] | None = None,
    sep: str = "-",
) -> tuple[list[str], list[str]]:
    """
    Extract unique physical properties and property-measures from filenames
    matched by a glob pattern. The function identifies unique segments in
    filenames based on specified extensions and index position, then sorts and
    returns them.

    Parameters
    ----------
    filepath : str
        The glob-friendly pattern used to locate filenames, e.g.,
        `path/to/files/*`.
    prop_idx : int
        The index position in the filename where the property or
        property-measure name starts after splitting by the separator.
    extensions : list[str]
        A list of suffixes that indicate the end of a property or
        property-measure name (e.g., `'-ensAvg'`, `'-ens'`, `'-whole'`). These
        are not file extensions like `".csv"` or `".npy"`.
    drop_properties : list[str] | None, default None
        A list of property names to ignore when determining unique properties
        and property-measures.
    sep : str, default '-'
        The separator used to split properties and measures within a filename.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - **uniq_props** (*list[str]*): A sorted list of unique physical
          properties.
        - **uniq_prop_measures** (*list[str]*): A sorted list of unique
          property-measures.

    Examples
    --------
    Given the following filenames:

    - `path/to/files/gyrT-acf-ensAvg.npy`
    - `path/to/files/gyrR-acf-ensAvg.npy`
    - `path/to/files/temp-ens.npy`

    The function can be used as follows:

    >>> find_unique_properties("path/to/files/*", prop_idx=0, # doctest: +SKIP
                               extensions=['-ensAvg', '-ens'])
    (['gyrT', 'fsdT'], ['gyrT-acf', 'fsdT-acf'])

    Notes
    -----
    - Ensure `prop_idx` accurately reflects the location in the filename where
      the property or measure name starts after splitting by `sep`.
    - This function assumes a consistent filename format where property-measure
      segments follow each other with defined separators and suffixes.
    """
    props_measures = glob(filepath)
    uniq_prop_measures = set()
    for ext in extensions:
        for prop in props_measures:
            prop_name = sep.join(
                prop.split("/")[-1].split(ext)[0].split(sep)[prop_idx:]
            )
            uniq_prop_measures.add(prop_name)

    if drop_properties is not None:
        uniq_prop_measures.difference_update(drop_properties)

    # Extract unique properties (first segment before `sep`)
    # from property-measures
    uniq_props = {prop.split(sep)[0] for prop in uniq_prop_measures}

    # Remove any full properties that match unique property names from
    # prop-measures
    uniq_prop_measures.difference_update(uniq_props)

    # Sort and return the results as lists
    return sorted(uniq_props), sorted(uniq_prop_measures)


def space_tseries(
    input_database: str,
    prop: PropertyT,
    parser: ParserClassT,
    hierarchy: str,
    physical_attrs: list[str],
    group: GroupT,
    dump_key: str,
    divisor: float = 0.025,
    round_to: int = 3,
    is_save: bool | None = False,
) -> pd.DataFrame:
    """
    Aggregate ensemble-averaged time-series data for a specified physical
    property across multiple files, adds specified physical attributes as
    columns, and concatenates into a single DataFrame.

    Parameters
    ----------
    input_database : str
        Path to the directory containing time-series data files.
    prop : str
        Name of the physical property of interest.
    parser : ParserClassT
        Parser class to infer file-specific information from filenames or
        paths.
    hierarchy : str
        Pattern prefix for the time-series filenames (e.g., `"N*"`).
    physical_attrs : list[str]
        List of physical attributes to add as new columns in the output
        DataFrame.
    group : GroupT
        Particle group type.
    dump_key : str
        Dumping attribute/keyword used to extract dumping frequency from an
        instance of the `parser`.
    divisor : float, default 0.025
        Rounding step for `phi_c_bulk` attribute.
    round_to : int, default 3
        Number of decimal places for rounding `phi_c_bulk` values.
    is_save : bool, default False
        If True, saves the concatenated DataFrame to a CSV file.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all time-series with added physical
        attributes.

    Examples
    --------
    >>> df = space_tseries( # doctest: +SKIP
                           "path/to/database", 'density', parser=SomeParser,
                           hierarchy="N*", physical_attrs=['dmon', 'dcyl'],
                           group='all', geometry='cubic', topology='linear')
    >>> df.head() # doctest: +SKIP

    Notes
    -----
    - This function assumes the presence of `phi_c_bulk` attribute in the
      parser output.
    - Requires a parser class with methods to retrieve attribute information
      for each file.
    """
    prop_ext = f"-{prop}-ensAvg.csv"
    ens_avg_csvs = sort_filenames(
        glob(input_database + hierarchy + prop_ext), [prop_ext]
    )
    prop_csvs = []

    for ens_avg_csv in ens_avg_csvs:
        ens_avg = pd.read_csv(ens_avg_csv[0], header=0)
        prop_info = parser(ens_avg_csv[0], "ensemble_long", group)

        ens_avg.reset_index(inplace=True)
        ens_avg.rename(columns={"index": "t_index"}, inplace=True)

        # Calculate `t_index` and `time` columns based on `dumping_freq`
        ens_avg["t_index"] *= getattr(prop_info, dump_key)
        ens_avg["time"] = ens_avg["t_index"] * getattr(prop_info, "dt")

        # Add physical attributes
        for attr_name in physical_attrs:
            ens_avg[attr_name] = getattr(prop_info, attr_name)

        # Apply rounding to `phi_c_bulk`
        ens_avg["phi_c_bulk_round"] = ens_avg["phi_c_bulk"].apply(
            lambda x: round_up_nearest(x, divisor, round_to)
        )

        prop_csvs.append(ens_avg)

    # Concatenate all time-series DataFrames
    prop_db = pd.concat(prop_csvs, axis=0)
    prop_db.reset_index(drop=True, inplace=True)

    # Optionally save to file
    if is_save:
        save_to_space = make_database(
            input_database, "analysis", stage="space", group=group
        )
        space = save_to_space.split("/")[-2].split("-")[0]
        filepath = (
            save_to_space + f"{create_fullname(space, group, prop)}-space.csv"
        )
        prop_db.to_csv(filepath, index=False)

    return prop_db


def space_hists(
    input_database: str,
    prop: PropertyT,
    parser: ParserClassT,
    hierarchy: str,
    physical_attrs: list[str],
    group: GroupT,
    bin_center: np.ndarray | None = None,
    normalize: bool | None = False,
    divisor: float = 0.025,
    round_to: int = 3,
    is_save: bool | None = False,
) -> pd.DataFrame:
    """
    Aggregate ensemble-averaged histogram data for a specified physical
    property across multiple files, normalizes data if specified, and
    concatenates into a single DataFrame.

    Parameters
    ----------
    input_database : str
        Path to the directory containing histogram data files.
    prop: str
        Name of the physical property of interest.
    parser : ParserClassT
        Parser class to infer file-specific information from filenames or
        paths.
    hierarchy : str
        Pattern prefix for the histogram filenames (e.g., `"N*"`).
    physical_attrs : list[str]
        List of physical attributes to add as new columns in the output
        DataFrame.
    group : GroupT
        Particle group type.
    bin_center : np.ndarray, optional
        Array of bin centers. If not provided, must be present in the
        DataFrames.
    normalize : bool, default False
        If True, normalizes the histogram data.
    divisor : float, default 0.025
        Rounding step for `phi_c_bulk` attribute.
    round_to : int, default 3
        Number of decimal places for rounding `phi_c_bulk` values.
    is_save : bool, default False
        If True, saves the concatenated DataFrame to a CSV file.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all histograms with added physical
        attributes.

    Examples
    --------
    >>> df = space_hists( # doctest: +SKIP
                         "path/to/database", 'density', parser=SomeParser,
                         hierarchy="N*", physical_attrs=['temperature'],
                         group='all', geometry='cylindrical',
                         topology='linear')
    >>> df.head() # doctest: +SKIP

    Notes
    -----
    - If `normalize` is True, histogram values will be scaled to sum to 1.
    - The `bin_center` should be provided if it is not available in the input
      data. When all the ensemble-averaged DataFrames have the same bin
      center, like 'clustersHistTFoci' or 'bondsHistTFoci' properties, the
      `bij_center` may not provided.
    """
    prop_ext = f"-{prop}-ensAvg.csv"
    ens_avg_csvs_ungrouped = glob(input_database + hierarchy + prop_ext)
    ens_avg_csvs = sort_filenames(ens_avg_csvs_ungrouped, formats=[prop_ext])
    prop_csvs = []

    for ens_avg_csv in ens_avg_csvs:
        ens_avg = pd.read_csv(ens_avg_csv[0], header=0)
        prop_info = parser(ens_avg_csv[0], "ensemble_long", group)

        # Handle bin_center if provided
        if bin_center is not None:
            ens_avg["bin_center"] = bin_center.tolist()
            ens_avg["bin_center-norm"] = (
                ens_avg["bin_center"] / ens_avg["bin_center"].max()
            )

        # Normalize data if specified
        if normalize:
            normalizer = ens_avg[prop + "-mean"].sum()
            if normalizer != 0:
                ens_avg[prop + "-norm"] = ens_avg[prop + "-mean"] / normalizer
            else:
                warnings.warn(
                    "All values are zero; normalized values set to zero.",
                    UserWarning,
                )
                ens_avg[prop + "-norm"] = 0

        # Add physical attributes
        for attr_name in physical_attrs:
            ens_avg[attr_name] = getattr(prop_info, attr_name)

        # Apply rounding to `phi_c_bulk`
        ens_avg["phi_c_bulk_round"] = ens_avg["phi_c_bulk"].apply(
            lambda x: round_up_nearest(x, divisor, round_to)
        )

        prop_csvs.append(ens_avg)

    # Concatenate all histograms
    prop_db = pd.concat(prop_csvs, axis=0)
    prop_db.reset_index(drop=True, inplace=True)

    # Optionally save to file
    if is_save:
        save_to_space = make_database(
            input_database, "analysis", stage="space", group=group
        )
        space = save_to_space.split("/")[-2].split("-")[0]
        filepath = (
            save_to_space + f"{create_fullname(space, group, prop)}-space.csv"
        )
        prop_db.to_csv(filepath, index=False)

    return prop_db
