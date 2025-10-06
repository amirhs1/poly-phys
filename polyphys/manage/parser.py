"""\
==========================================================
:mod:`polyphys.manage.parser`
==========================================================

This :mod:`~polyphys.manage.parser` module defines a suite of base and
specialized parser classes for extracting structured, lineage-based information
from filenames and file paths in molecular simulation projects. Each parser
subclass represents a unique project type and geometry (e.g., cubic,
cylindrical), allowing lineage-specific parsing of artifact attributes from
filenames.

The lineage attribute hierarchy includes the following levels: 'segment',
'whole', 'ensemble_long', 'ensemble', and 'space'. Each level represents a
different scope within the project, ranging from individual simulation segments
to complete simulation spaces. Different classes support parsing of project
attributes such as particle dimensions, densities, and simulation box
dimensions. These classes employ various methods to dynamically interpret and
calculate attributes based on parsed lineage and project requirements.

Classes
=======
.. autoclass:: ParserBase
   :members:
.. autoclass:: TwoMonDep
.. autoclass:: SumRuleCyl
.. autoclass:: SumRuleCubHeteroRing
.. autoclass:: SumRuleCubHeteroLinear
.. autoclass:: TransFociCyl
.. autoclass:: TransFociCub
.. autoclass:: HnsCub
.. autoclass:: HnsCyl

Dependencies
============
- `os`: For handling file paths.
- `re`: For regular expressions in filename parsing.
- `typing`: For type hinting.
- `abc`: For defining abstract base classes.
- `collections`: For ordered dictionary functionality.
- `.utilizer`: Utility functions, e.g., `invalid_keyword`.
- `..analyze.measurer`: Measurement functions such as `number_density_cube` and
  `volume_fraction_cube`.

Usage
=====
Classes in this module are instantiated with artifact filenames, lineages, and
group types, which they parse into a rich set of attributes. These classes
facilitate programmatic access to file-based information for complex
projects.

Examples
========
>>> artifact = SumRuleCyl('N200D10.0ac2.0nc0', 'ensemble', 'all')
>>> print(artifact.dcrowd)
2.0

>>> artifact = HnsCyl('N200D20.0nh16ac1.0epshc1.0nc0', 'space', 'nucleoid')
>>> print(artifact.eps_hc)
1.0

Notes
=====
These classes use dynamic attribute setting based on parsed filename data.
To handle attributes that may or may not exist at runtime, the `getattr`
function is commonly used, ensuring robustness to varied filename patterns.
"""
import os
import re
from pathlib import Path
from typing import ClassVar, Optional, Self
from abc import ABC, abstractmethod
from collections import OrderedDict
from .utils import (
    invalid_keyword,
    number_density_cube,
    number_density_cylinder,
    volume_fraction_cube,
    volume_fraction_cylinder
)
from .types import LineageT, TopologyT, GroupT, GeometryT, Pathish


class ParserBase(ABC):
    """
    Base parser class for extracting information from filenames or file paths
    in a structured project. Designed to enforce lineage, geometry, and group
    conventions across subclasses for specific project types.

    Parameters
    ----------
    artifact : Pathish
        The artifact to be parsed. Must be a relative or absolute file path
        (string or Path). The path may include or omit directories, but must
        represent a valid file. Pure name tokens without extensions should
        instead be passed via :meth:`from_name`.
    lineage : :py:data:`LineageT`
        The lineage of the artifact, specifying the hierarchical level within
        the project. Must be one of: ``'segment'``, ``'whole'``,
        ``'ensemble_long'``, ``'ensemble'``, or ``'space'``.
    group : :py:data:`GroupT`
        The particle group type in the project. Used for specific group-based
        parsing.

    Alternative Constructors
    ------------------------
    .. method:: from_name(name, lineage, group)
       Construct a parser directly from a *name token* (not a path).
       This bypasses filesystem resolution. Intended for cases where only
       a lineage token is available, e.g., when working with metadata or
       generated identifiers. In this mode, ``filepath`` and ``ext`` are
       set to empty strings, and the given token is treated as both the
       ``filename`` and ``name`` prior to :meth:`_find_name` normalization.

    Attributes
    ----------
    filepath : str
        Absolute directory path if `artifact` is a file path, or ``""`` if
        constructed via :meth:`from_name`.
    filename : str
        The filename extracted from `artifact`, or the provided token if
        constructed via :meth:`from_name`.
    ext : str
        The file extension (including the leading dot), or ``""`` if none
        was available (e.g., from :meth:`from_name`).
    name : str
        The unique name derived from `filename` based on `lineage` and
        `group` conventions.
    group : :py:data:`GroupT`
        Particle group type in the project.
    lineage : :py:data:`LineageT`
        The lineage of the artifact, specifying the hierarchical level within
        the project. Must be one of: ``'segment'``, ``'whole'``,
        ``'ensemble_long'``, ``'ensemble'``, or ``'space'``.
    project_name : str
        The name of the parser subclass (automatically assigned).
    lineage_genealogy : list of :py:data:`LineageT`
        List of parent lineages for the artifact’s `lineage`.
    lineage_attributes : list of str
        Parsed or dynamically defined attributes specific to the artifact’s
        `lineage`.
    physical_attributes : list of str
        Computed or system attributes specific to the artifact’s `lineage`.
    attributes : list of str
        Union of lineage-specific and physical attributes for the artifact.

    Class Attributes
    ----------------
    lineages : list of :py:data:`LineageT`
        List of valid lineage types.
    genealogy : dict of (:py:data:`LineageT` → list of :py:data:`LineageT`)
        Dictionary defining the hierarchical relationships between lineages.
        Each lineage points to its parent lineages in the hierarchy.

    Abstract Class Properties
    -------------------------
    geometry : :py:data:`GeometryT`
        Specifies the geometry of the system.
    topology : :py:data:`TopologyT`
        Specifies how particles are connected (or not) in the system.
    groups : list of :py:data:`GroupT`
        Allowed particle groups in the system.
    genealogy_attributes : dict of (
        :py:data:`LineageT` -> OrderedDict[str, str])
        Mapping from lineage type to an OrderedDict of attribute names and
        their short forms.
    project_attributes : dict of (:py:data:`LineageT` -> list of str)
        Mapping from lineage type to project-level attributes that remain
        constant but are not extractable from filenames.

    Methods
    -------
    _find_name() -> None
        Parse and sets the unique name based on `lineage` and `group`.
    _set_parents() -> None
        Set pattern names for each `lineage` based on `_genealogy`.
    _initiate_attributes() -> None
        Define and initializes subclass-specific attributes. (Abstract method)
    _parse_name() -> None
        Parse lineage-specific attributes based on the filename.
        (Abstract method)
    _bulk_attributes() -> None
        Compute physical attributes for the current lineage based on primary
        attributes. (Abstract method)
    """
    _lineages: ClassVar[list[LineageT]] = \
        ['segment', 'whole', 'ensemble_long', 'ensemble', 'space']
    _genealogy: ClassVar[dict[LineageT, list[LineageT]]] = {
        'segment': ['segment', 'whole', 'ensemble_long', 'ensemble', 'space'],
        'whole': ['whole', 'ensemble_long', 'ensemble', 'space'],
        'ensemble_long': ['ensemble_long', 'ensemble', 'space'],
        'ensemble': ['ensemble', 'space'],
        'space': ['space'],
    }
    _geometry: ClassVar[Optional[GeometryT]] = None
    _topology: ClassVar[Optional[TopologyT]] = None
    _groups: ClassVar[Optional[list[GroupT]]] = None
    _genealogy_attributes: \
        ClassVar[Optional[dict[LineageT, OrderedDict[str, str]]]] = None
    _project_attributes: ClassVar[Optional[dict[LineageT, list[str]]]] = None

    def __init__(
        self,
        artifact: Pathish,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        artifact_str = os.fspath(artifact)
        if artifact_str == '':
            raise ValueError("'artifact' cannot be an empty string.")

        invalid_keyword(lineage, self.lineages)
        invalid_keyword(group, self.groups)

        # Normalize: expand env/~ and convert to absolute path
        p = Path(
            os.path.expandvars(os.path.expanduser(artifact_str))).resolve()

        self._filepath: str = str(p.parent)  # absolute dir
        self._filename: str = p.name         # e.g., '...bug.data' or '...bug'
        self._name: str = p.stem             # filename without last suffix
        self._ext: str = p.suffix            # e.g., '.data' or '' if none

        self._project_name = self.__class__.__name__
        self._lineage: LineageT = lineage
        self._group: GroupT = group

        # Update the self._name by dropping 'group' and/or any other extra
        # beyond the template defined above.
        self._find_name()
        self._lineage_genealogy: list[LineageT] = self._genealogy[lineage]
        self._lineage_attributes = \
            list(self.genealogy_attributes[lineage].keys())
        self._physical_attributes = self.project_attributes[lineage]
        self._attributes = \
            self._lineage_attributes + self._physical_attributes

    def __str__(self) -> str:
        """
        Provide a formatted summary of the parser instance.
        """
        observation = (
            f"Artifact:\n"
            f"    Name: '{self.filename}',\n"
            f"    Geometry: '{self.geometry}',\n"
            f"    Group: '{self.group}',\n"
            f"    Lineage: '{self.lineage}',\n"
            f"    Topology: '{self.topology}',\n"
            f"    Project: '{self.project_name}'"
        )
        return observation

    def __repr__(self) -> str:
        return (
            f"Artifact(filename='{self.filename}', "
            f"geometry='{self.geometry}', "
            f"group='{self.group}', lineage='{self.lineage}', "
            f"topology='{self.topology}', project='{self.project_name}')"
        )

    @classmethod
    def from_name(
        cls,
        name: str,
        lineage: LineageT,
        group: GroupT
    ) -> Self:
        """
        Construct a parser from a *name token* (not a path).

        Sets `filepath` and `ext` to empty strings and keeps the token as both
        `filename` and `name` before `_find_name()` normalization.
        """
        if not name:
            raise ValueError("'name' cannot be an empty string.")

        # Bypass __init__; we populate fields manually
        self = cls.__new__(cls)  # type: ignore[misc]

        # Minimal identity (class attributes must be available)
        self._project_name = cls.__name__
        invalid_keyword(lineage, self.lineages)
        invalid_keyword(group, self.groups)
        self._lineage = lineage
        self._group = group

        # Token semantics (no filesystem)
        self._filepath = ""
        self._filename = name
        self._name = name
        self._ext = ""

        # Continue same setup pipeline
        self._find_name()
        self._lineage_genealogy = self._genealogy[lineage]
        self._lineage_attributes = \
            list(self.genealogy_attributes[lineage].keys())
        self._physical_attributes = self.project_attributes[lineage]
        self._attributes = self._lineage_attributes + self._physical_attributes
        return self

    @property
    def lineages(self) -> list[LineageT]:
        """
        List of all the acceptable lineages.
        """
        return self._lineages

    @property
    def genealogy(self) -> dict[LineageT, list[LineageT]]:
        """
        Dictionary of lineages and their parent lineages.
        """
        return self._genealogy

    @property
    def geometry(self) -> GeometryT:
        """
        System geometry in a molecular dynamics system.
        """
        if self._geometry is None:
            raise AttributeError("'_geometry' has not been initialized.")
        return self._geometry

    @property
    def groups(self) -> list[GroupT]:
        """
        List of valid group names for the subclass.
        """
        if self._groups is None:
            raise AttributeError("'_groups' has not been initialized.")
        return self._groups

    @property
    def topology(self) -> TopologyT:
        """
        Define the polymer topology for the parser subclass.
        """
        if self._topology is None:
            raise AttributeError("'_topology' has not been initialized.")
        return self._topology

    @property
    def genealogy_attributes(self) -> dict[LineageT, OrderedDict[str, str]]:
        """
        Dictionary of lineage-specific attributes. Each key is a lineage type,
        and each value is an OrderedDict mapping attribute names to their
        short-form representations.
        """
        if self._genealogy_attributes is None:
            raise AttributeError(
                "'_genealogy_attributes' has not been initialized.")
        return self._genealogy_attributes

    @property
    def project_attributes(self) -> dict[LineageT, list[str]]:
        """
        Dictionary of project attributes. Each key is a lineage type,
        and each value is an dict mapping attribute names to their
        short-form representations.
        """
        if self._project_attributes is None:
            raise AttributeError(
                "'_project_attributes' has not been initialized.")
        return self._project_attributes

    @property
    def filename(self) -> str:
        """
        Return the filename, either extracted from the path or the name
        itself.
        """
        return self._filename

    @property
    def filepath(self) -> str:
        """
        Return the full filepath or 'N/A' if not a valid path.
        """
        return self._filepath

    @property
    def group(self) -> GroupT:
        """
        Return the current group.
        """
        return self._group

    @property
    def ext(self) -> str:
        """
        Return the current file extension.
        """
        return self._ext

    @property
    def lineage(self) -> LineageT:
        """
        Return the current lineage.
        """
        return self._lineage

    @property
    def name(self) -> str:
        """
        Return the unique name parsed from the filename.
        """
        return self._name

    @property
    def project_name(self) -> str:
        """
        Return the project (parser class) name,
        """
        return self._project_name

    @property
    def attributes(self) -> list[str]:
        """
        Return lineage-specific and project attributes for an artifact.
        """
        return self._attributes

    @property
    def lineage_genealogy(self) -> list[LineageT]:
        """
        Return the parents of a given `lineage`.
        """
        return self._lineage_genealogy

    @property
    def lineage_attributes(self) -> list[str]:
        """
        Return lineage-specific attributes for an artifact with a given
        `lineage`.
        """
        return self._lineage_attributes

    @property
    def physical_attributes(self) -> list[str]:
        """
        Return project-level attributes for an artifact with a given
        `lineage`.
        """
        return self._physical_attributes

    def _find_name(self) -> None:
        """
        parses the unique lineage_name (the first substring of filename
        and/or the segment keyword middle substring) of a filename.
        """
        if self.lineage in ['segment', 'whole']:
            # a 'segment' lineage only used in 'probe' phase
            # a 'whole' lineage used in 'probe' or 'analyze' phases
            # so its lineage_name is either ended by 'group' keyword or '-'.
            # these two combined below:
            base = self.name.split(f".{self.group}")[0]
            self._name = base.split('-')[0]
        else:  # 'ensemble' or 'space' lineages
            self._name = self.name.split('-')[0]

    @abstractmethod
    def _initiate_attributes(self) -> None:
        """
        Define and initiate the project attributes. Lineage attributes are
        set dynamically via `_parse_name` method.
        """

    def _set_parents(self) -> None:
        """
        Set parent lineage names for each lineage type, following the
        hierarchy defined in `_genealogy`.

        Notes
        -----
        The `self._genealogy` defines the following parent-child hierarchy:

            - space -> ensemble -> ensemble_long -> whole -> segment

        Each lineage on the left has all the lineages on its right.
        """
        for lineage_name in self.lineages:
            lineage_value = 'N/A'
            if lineage_name in self.genealogy[self.lineage]:
                lineage_value = ''
                lineage_attr = self.genealogy_attributes[lineage_name]
                for attr_long, attr_short in lineage_attr.items():
                    lineage_value += \
                            f"{attr_short}{getattr(self, attr_long)}"
            setattr(self, lineage_name, lineage_value)

    @abstractmethod
    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """

    @abstractmethod
    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """


class TwoMonDepCub(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *TwoMonDep* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - ``segment``: am#nm#ac#nc#hl#sd#dt#bdump#adump#tdump#ens#.j#
      One of multiple chunks of a complete artifact.
    - ``whole``: am#nm#ac#nc#hl#sd#dt#bdump#adump#tdump#ens#
      A complete artifact. It may be a collection of segments.
    - ``ensemble_long``: am#nm#ac#nc#hl#sd#dt#bdump#adump#tdump#
      Detailed name for an 'ensemble' artifact.
    - ``ensemble``: nm#am#ac#nc#sd#
      Short name for an 'ensemble' artifact.
    - ``space``: nm#am#ac#nc#
      A 'space' artifact.

    For the above four lineages, the short-form keys (e.g., ``am``, ``nm``,
    ``ac``)  are physical attributes where their values (shown by '#' sign) are
    float or integer number. See `genealogy_attributes` below for long names
    of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : LineageT
        Lineage level of the artifact. Must be one of:
        ``'segment'``, ``'whole'``, ``'ensemble_long'``, ``'ensemble'``,
        ``'space'``.
    group : GroupT
        Particle group type, either ``'bug'`` or ``'all'``.

    Attributes
    ----------
    dmon : float
        Size (diameter) of a monomer. Its associated keyword is 'am'.
    nmon : int
        Number of monomers. Its associated keyword is 'N'.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    lcube : float
        Length of the simulation box, inferred from 'hl' keyword
        (half-length of the simulation box).
    d_sur : float
        Surface-to-surface distance between two monomers fixed in space.
        Its associated keyword is 'sd'.
    dt : float
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    tdump : int
        Frequency by which 'thermo' variables are written in a 'lammps'
        log file. Its associated keyword is 'tdump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Examples
    --------
    >>> artifact = TwoMonDepCub('nm2am5.0ac1.0nc1000', 'space', 'bug')
    >>> print(artifact.nmon)
    2
    """
    _geometry = 'cubic'
    _topology = 'atomic'
    _groups = ['bug', 'all']

    _genealogy_attributes = {
        # Pattern: am#nm#ac#nc#hl#sd#dt#bdump#adump#tdump#ens#.j#
        'segment': OrderedDict({
            'dmon': 'am', 'nmon': 'nm', 'dcrowd': 'ac', 'ncrowd': 'nc',
            'lcube': 'hl', 'd_sur': 'sd', 'dt': 'dt', 'bdump': 'bdump',
            'adump': 'adump', 'tdump': 'tdump', 'ensemble_id': 'ens',
            'segment_id': 'j'}
            ),
        # Pattern: am#nm#ac#nc#hl#sd#dt#bdump#adump#tdump#ens#
        'whole': OrderedDict({
            'dmon': 'am', 'nmon': 'nm', 'dcrowd': 'ac', 'ncrowd': 'nc',
            'lcube': 'hl', 'd_sur': 'sd', 'dt': 'dt', 'bdump': 'bdump',
            'adump': 'adump', 'tdump': 'tdump', 'ensemble_id': 'ens'}
            ),
        # Pattern: am#nm#ac#nc#hl#sd#dt#bdump#adump#tdump# :
        'ensemble_long': OrderedDict({
            'dmon': 'am', 'nmon': 'nm', 'dcrowd': 'ac', 'ncrowd': 'nc',
            'lcube': 'hl', 'd_sur': 'sd', 'dt': 'dt', 'bdump': 'bdump',
            'adump': 'adump', 'tdump': 'tdump'}
            ),
        # Pattern: nm#am#ac#nc#sd# :
        'ensemble': OrderedDict(
            {'nmon': 'nm', 'dmon': 'am', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'd_sur': 'sd'}
             ),
        # pattern: nm#am#ac#nc# :
        'space': OrderedDict(
            {'nmon': 'nm', 'dmon': 'am', 'dcrowd': 'ac', 'ncrowd': 'nc'}
            )
    }

    _project_attributes = {
        'segment': ['phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'whole': ['phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble_long': ['phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c',
                          'rho_bulk_c'],
        'ensemble': [],
        'space': []
        }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['dmon', 'lcube', 'dcrowd', 'dt', 'd_sur']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'hl':
                    # Cube full side from its half-side
                    setattr(self, attr, 2 * getattr(self, attr))
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m = number_density_cube(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m = volume_fraction_cube(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_c = number_density_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )


class SumRuleCyl(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *SumRuleCyl* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: N#epsilon#r#lz#sig#nc#dt#bdump#adump#ens#.j#
      One of multiple chunks of a complete artifact.
    - `whole`: N#epsilon#r#lz#sig#nc#dt#bdump#adump#ens#
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: N#epsilon#r#lz#sig#nc#dt#bdump#adump#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: N#D#ac#nc#
      Short name for an 'ensemble' artifact.
    - `space`: N#D#ac#
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'bug', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon : float
        Size (diameter) of a monomer. Its associated keyword is 'am'.
    nmon : int
        Number of monomers. Its associated keyword is 'N'.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    lcyl : float
        Length of the cylindrical confinement along z axis (the periodic,
        direction), inferred from 'lz' keyword (half of the length of the
        cylindrical confinement along z axis).
    dcyl : float
        Size (or diameter) of the cylindrical confinement, inferred
        from either 'r' keyword (the radius of a cylindrical confinement
        with open ends) or 'D' keyword (size of that confinement).
    epsilon: float
        Wall-particle LJ interaction strength. Its associated keyword is
        'epsilon' keyword.
    dt : float
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Notes
    -----
    The cylindrical wall is implemented in LAMMPS by using wall-forming
    particles of size 1.0. Thus, the actual size of the cylinder size
    (diameter), :math:`D`, is :math:`D=2r-1.0`,  :math:`r` is the radius of
    the cylindrical region defined in LAMMPS.

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = SumRuleCyl('N200D10.0ac2.0nc0', 'ensemble', 'all')
    >>> print(artifact.dcrowd)
    2.0
    """
    _geometry = 'cylindrical'
    _topology = 'linear'
    _groups = ['bug', 'all']

    _genealogy_attributes = {
        # Pattern: N#epsilon#r#lz#sig#nc#dt#bdump#adump#ens#.j#
        'segment':  OrderedDict(
            {'nmon': 'N', 'epsilon': 'epsilon', 'dcyl': 'r', 'lcyl': 'lz',
             'dcrowd': 'sig', 'ncrowd': 'nc', 'dt': 'dt', 'bdump': 'bdump',
             'adump': 'adump', 'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: N#epsilon#r#lz#sig#nc#dt#bdump#adump#ens#
        'whole':  OrderedDict(
            {'nmon': 'N', 'epsilon': 'epsilon', 'dcyl': 'r', 'lcyl': 'lz',
             'dcrowd': 'sig', 'ncrowd': 'nc', 'dt': 'dt', 'bdump': 'bdump',
             'adump': 'adump', 'ensemble_id': 'ens'}),
        # Pattern: N#epsilon#r#lz#sig#nc#dt#bdump#adump#
        'ensemble_long':  OrderedDict(
            {'nmon': 'N', 'epsilon': 'epsilon', 'dcyl': 'r', 'lcyl': 'lz',
             'dcrowd': 'sig', 'ncrowd': 'nc', 'dt': 'dt', 'bdump': 'bdump',
             'adump': 'adump'}),
        # Pattern: N#D#ac#nc#
        'ensemble':  OrderedDict(
            {'nmon': 'N', 'dcyl': 'D', 'dcrowd': 'ac', 'ncrowd': 'nc'}),
        # Pattern: N#D#ac#
        'space':  OrderedDict({'nmon': 'N', 'dcyl': 'D', 'dcrowd': 'ac'})
    }
    _project_attributes = {
        'segment': ['dmon', 'phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c',
                    'rho_bulk_c'],
        'whole': ['dmon',  'phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c',
                  'rho_bulk_c'],
        'ensemble_long': ['dmon', 'phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c',
                          'rho_bulk_c'],
        'ensemble': ['dmon'],
        'space': ['dmon']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon: float = 1
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['dmon', 'dcyl', 'lcyl', 'epsilon', 'dcrowd', 'dt']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'lz':
                    # Cylinder full length from its half-length
                    setattr(self, attr, 2 * getattr(self, attr))
                if keyword == 'r':
                    # Cylinder size is twice its radius, correcting for
                    # wall-forming particles with size 1
                    setattr(self, attr, 2 * getattr(self, attr) - 1.0)
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m = number_density_cylinder(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_m = volume_fraction_cylinder(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.rho_bulk_c = number_density_cylinder(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cylinder(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )


class SumRuleCubHeteroRing(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *SumRuleCubHeteroRing* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#[.ring
]      One of multiple chunks of a complete artifact.
    - `whole`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.ring
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: ns#nl#al#ac#nc#
      Short name for an 'ensemble' artifact.
    - `space`: ns#nl#al#ac#
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'bug', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon_small: float
        Size (diameter) of a monomer
    nmon_small: int
        number of small monomers. Its associated keyword is 'ns'.
    mmon_small: float
        Mass of a small monomer
    dmon_large: float
        Size (diameter) of a large monomer. Its associated keyword is 'al'.
    nmon_large: int
        number of large monomers. Its associated keyword is 'nl'.
    mmon_large: float, default np.nan
        Mass of a large monomer. Its associated keyword is 'ml'.
    nmon: int
        Total number of monomers.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    mcrowd: float, default np.nan
        Mass of a crowder.
    lcube : float
        Length of the simulation box, inferred from 'l' keyword
        (half-length of the simulation box).
    dt : float
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m_small : float
        Bulk number density fraction of small monomers.
    phi_bulk_m_small : float
        Bulk volume fraction of small monomers
    rho_bulk_m_large : float
        Bulk number density fraction of large monomers.
    phi_bulk_m_large : float
        Bulk volume fraction of large monomers
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Notes
    -----
    The mass density is uniform across all species. For any species whose mass
    is not explicitly parsed, the mass is defined as :math:`m_{i} = d_{i}^3`,
    where :math:`d_{i}` represents the species' diameter.

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = SumRuleCubHeteroRing('ns400nl5al5.0ac1.0', 'space', 'bug')
    >>> print(artifact.dcrowd)
    1.0
    """
    _geometry = 'cubic'
    _topology = 'ring'
    _groups = ['bug', 'all']

    _genealogy_attributes = {
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.ring
        'segment': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'lcube': 'l', 'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.ring
        'whole': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc', 'lcube': 'l',
             'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens'}),
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#
        'ensemble_long': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'lcube': 'l', 'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump'}),
        # Pattern: ns#nl#al#ac#nc#
        'ensemble': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcrowd': 'ac', 'ncrowd': 'nc'}),
        # Pattern: ns#nl#al#ac#
        'space': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcrowd': 'ac'})
    }

    _project_attributes = {
        'segment': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                    'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                    'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'whole': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                  'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                  'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble_long': ['dmon_small', 'mmon_small',  'mcrowd',
                          'phi_bulk_m_small', 'rho_bulk_m_small',
                          'phi_bulk_m_large', 'phi_bulk_m_large', 'rho_bulk_m',
                          'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble': ['dmon_small', 'mmon_small', 'mcrowd'],
        'space': ['dmon_small', 'mmon_small', 'mcrowd']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon_small: float = 1
        self.mmon_small: float = self.dmon_small**3
        self.mcrowd: float = -1
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m_small: float = -1
            self.rho_bulk_m_small: float = -1
            self.phi_bulk_m_large: float = -1
            self.rho_bulk_m_large: float = -1
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['dmon_large', 'lcube', 'mmon_large', 'dcrowd', 'dt']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'l':
                    # Cube full side from its half-side
                    setattr(self, attr, 2 * getattr(self, attr))
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m_small = number_density_cube(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m_small = volume_fraction_cube(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_m_large = number_density_cube(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m_large = volume_fraction_cube(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_m = self.rho_bulk_m_small + self.rho_bulk_m_large
        self.phi_bulk_m = self.phi_bulk_m_small + self.phi_bulk_m_large

        self.mcrowd = getattr(self, 'dcrowd') ** 3
        self.rho_bulk_c = number_density_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )


class SumRuleCubHeteroLinear(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *SumRuleCubHeteroRing* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.linear
      One of multiple chunks of a complete artifact.
    - `whole`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.linear
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: ns#nl#al#ac#nc#
      Short name for an 'ensemble' artifact.
    - `space`: ns#nl#al#ac#
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'bug', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon_small: float
        Size (diameter) of a monomer
    nmon_small: int
        number of small monomers. Its associated keyword is 'ns'.
    mmon_small: float
        Mass of a small monomer
    dmon_large: float
        Size (diameter) of a large monomer. Its associated keyword is 'al'.
    nmon_large: int
        number of large monomers. Its associated keyword is 'nl'.
    mmon_large: float, default np.nan
        Mass of a large monomer. Its associated keyword is 'ml'.
    nmon: int
        Total number of monomers.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    mcrowd: float, default np.nan
        Mass of a crowder.
    lcube : float
        Length of the simulation box, inferred from 'l' keyword
        (half-length of the simulation box).
    dt : float
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m_small : float
        Bulk number density fraction of small monomers.
    phi_bulk_m_small : float
        Bulk volume fraction of small monomers
    rho_bulk_m_large : float
        Bulk number density fraction of large monomers.
    phi_bulk_m_large : float
        Bulk volume fraction of large monomers
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Notes
    -----
    The mass density is uniform across all species. For any species whose mass
    is not explicitly parsed, the mass is defined as :math:`m_{i} = d_{i}^3`,
    where :math:`d_{i}` represents the species' diameter.

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = SumRuleCubHeteroLinear('ns800nl5al6.0ac3.0', 'space', 'all')
    >>> print(artifact.dmon_large)
    6.0
    """
    _geometry = 'cubic'
    _topology = 'linear'
    _groups = ['bug', 'all']

    _genealogy_attributes = {
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.linear
        'segment': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'lcube': 'l', 'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.linear
        'whole': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc', 'lcube': 'l',
             'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens'}),
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#
        'ensemble_long': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'lcube': 'l', 'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump'}),
        # Pattern: ns#nl#al#ac#nc#
        'ensemble': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcrowd': 'ac', 'ncrowd': 'nc'}),
        # Pattern: ns#nl#al#ac#
        'space': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcrowd': 'ac'})
    }

    _project_attributes = {
        'segment': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                    'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                    'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'whole': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                  'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                  'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble_long': ['dmon_small', 'mmon_small',  'mcrowd',
                          'phi_bulk_m_small', 'rho_bulk_m_small',
                          'phi_bulk_m_large', 'phi_bulk_m_large', 'rho_bulk_m',
                          'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble': ['dmon_small', 'mmon_small', 'mcrowd'],
        'space': ['dmon_small', 'mmon_small', 'mcrowd']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon_small: float = 1
        self.mmon_small: float = self.dmon_small**3
        self.mcrowd: float = -1
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m_small: float = -1
            self.rho_bulk_m_small: float = -1
            self.phi_bulk_m_large: float = -1
            self.rho_bulk_m_large: float = -1
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['dmon_large', 'lcube', 'mmon_large', 'dcrowd', 'dt']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'l':
                    # Cube full side from its half-side
                    setattr(self, attr, 2 * getattr(self, attr))
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m_small = number_density_cube(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m_small = volume_fraction_cube(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_m_large = number_density_cube(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m_large = volume_fraction_cube(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_m = self.rho_bulk_m_small + self.rho_bulk_m_large
        self.phi_bulk_m = self.phi_bulk_m_small + self.phi_bulk_m_large

        self.mcrowd = getattr(self, 'dcrowd') ** 3
        self.rho_bulk_c = number_density_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )


class TransFociCyl(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *TransFociCyl* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: epss#epsl#r#al#nl#ml#ns#ac#nc#lz#dt#bdump#adump#ens#.j#.ring
      One of multiple chunks of a complete artifact.
    - `whole`: epss#epsl#r#al#nl#ml#ns#ac#nc#lz#dt#bdump#adump#ens#.ring
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: epss#epsl#r#al#nl#ml#ns#ac#nc#lz#dt#bdump#adump#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: ns#nl#al#D#ac#nc#
      Short name for an 'ensemble' artifact.
    - `space`: ns#nl#al#ac#
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'bug', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon_small: float
        Size (diameter) of a monomer
    nmon_small: int
        number of small monomers. Its associated keyword is 'ns'.
    mmon_small: float
        Mass of a small monomer
    dmon_large: float
        Size (diameter) of a large monomer. Its associated keyword is 'al'.
    nmon_large: int
        number of large monomers. Its associated keyword is 'nl'.
    mmon_large: float, default np.nan
        Mass of a large monomer. Its associated keyword is 'ml'.
    nmon: int
        Total number of monomers.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    mcrowd: float, default np.nan
        Mass of a crowder.
    lcyl : float
        Length of the cylindrical confinement along z axis (the periodic,
        direction), inferred from 'lz' keyword (half of the length of the
        cylindrical confinement along z axis).
    dcyl : float
        Size (or diameter) of the cylindrical confinement, inferred
        from either 'r' keyword (the radius of a cylindrical confinement
        with open ends) or 'D' keyword (size of that confinement).
    epsilon_s: float
        Wall-small-monomer LJ interaction strength. Its associated keyword is
        'epss' keyword.
    epsilon_l: float
        Wall-large-monomer LJ interaction strength. Its associated keyword is
        'espl' keyword.
    dt : float
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m_small : float
        Bulk number density fraction of small monomers.
    phi_bulk_m_small : float
        Bulk volume fraction of small monomers
    rho_bulk_m_large : float
        Bulk number density fraction of large monomers.
    phi_bulk_m_large : float
        Bulk volume fraction of large monomers
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Notes
    -----
    - The mass density is uniform across all species. For any species whose
      mass is not explicitly parsed, the mass is defined as
      :math:`m_{i} = d_{i}^3`, where :math:`d_{i}` represents the species'
      diameter.

    - The cylindrical wall is implemented in LAMMPS by using wall-forming
      particles of size 1.0. Thus, the actual size of the cylinder size
      (diameter), :math:`D`, is :math:`D=2r-1.0`,  :math:`r` is the radius of
      the cylindrical region defined in LAMMPS.

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = TransFociCyl(
    ... 'ns500nl5al3.0D20.0ac2.0nc0',
    ... 'ensemble',
    ... 'bug'
    ... )
    >>> print(artifact.dcyl)
    20.0
    """
    _geometry = 'cylindrical'
    _topology = 'ring'
    _groups = ['bug', 'all']

    _genealogy_attributes = {
        # Pattern: epss#epsl#r#al#nl#ml#ns#ac#nc#lz#dt#bdump#adump#ens#.j#
        # .ring
        'segment': OrderedDict(
            {'epsilon_small': 'epss', 'epsilon_large': 'epsl', 'dcyl': 'r',
             'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc', 'lcyl': 'lz',
             'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: epss#epsl#r#al#nl#ml#ns#ac#nc#lz#dt#bdump#adump#ens#.ring
        'whole': OrderedDict(
            {'epsilon_small': 'epss', 'epsilon_large': 'epsl', 'dcyl': 'r',
             'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc', 'lcyl': 'lz',
             'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens'}),
        # Pattern: epss#epsl#r#al#nl#ml#ns#ac#nc#lz#dt#bdump#adump#
        'ensemble_long': OrderedDict(
            {'epsilon_small': 'epss', 'epsilon_large': 'epsl', 'dcyl': 'r',
             'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc', 'lcyl': 'lz',
             'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump'}),
        # Pattern: ns#nl#al#D#ac#nc#
        'ensemble': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcyl': 'D', 'dcrowd': 'ac', 'ncrowd': 'nc'}),
        # Pattern: ns#nl#al#D#ac#
        'space': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcyl': 'D', 'dcrowd': 'ac'})
    }

    _project_attributes = {
        'segment': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                    'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                    'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'whole': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                  'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                  'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble_long': ['dmon_small', 'mmon_small',  'mcrowd',
                          'phi_bulk_m_small', 'rho_bulk_m_small',
                          'phi_bulk_m_large', 'phi_bulk_m_large', 'rho_bulk_m',
                          'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble': ['dmon_small', 'mmon_small', 'mcrowd'],
        'space': ['dmon_small', 'mmon_small', 'mcrowd']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon_small: float = 1
        self.mmon_small: float = self.dmon_small**3
        self.mcrowd: float = -1
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m_small: float = -1
            self.rho_bulk_m_small: float = -1
            self.phi_bulk_m_large: float = -1
            self.rho_bulk_m_large: float = -1
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['dmon_large', 'dcyl', 'lcyl', 'epsilon_small',
                       'epsilon_large', 'mmon_large', 'dcrowd', 'dt']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'lz':
                    # Cylinder full length from its half-length
                    setattr(self, attr, 2 * getattr(self, attr))
                if keyword == 'r':
                    # Cylinder size is twice its radius, correcting for
                    # wall-forming particles with size 1
                    setattr(self, attr, 2 * getattr(self, attr) - 1.0)
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m_small = number_density_cylinder(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_m_small = volume_fraction_cylinder(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.rho_bulk_m_large = number_density_cylinder(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_m_large = volume_fraction_cylinder(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.rho_bulk_m = self.rho_bulk_m_small + self.rho_bulk_m_large
        self.phi_bulk_m = self.phi_bulk_m_small + self.phi_bulk_m_large

        self.mcrowd = getattr(self, 'dcrowd') ** 3
        self.rho_bulk_c = number_density_cylinder(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cylinder(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )


class TransFociCub(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *TransFociCub* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.ring
      One of multiple chunks of a complete artifact.
    - `whole`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.ring
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: ns#nl#al#ac#nc#
      Short name for an 'ensemble' artifact.
    - `space`: ns#nl#al#ac#
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'bug', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon_small: float
        Size (diameter) of a monomer
    nmon_small: int
        number of small monomers. Its associated keyword is 'ns'.
    mmon_small: float
        Mass of a small monomer
    dmon_large: float
        Size (diameter) of a large monomer. Its associated keyword is 'al'.
    nmon_large: int
        number of large monomers. Its associated keyword is 'nl'.
    mmon_large: float, default np.nan
        Mass of a large monomer. Its associated keyword is 'ml'.
    nmon: int
        Total number of monomers.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    mcrowd: float, default np.nan
        Mass of a crowder.
    lcube : float
        Length of the simulation box, inferred from 'l' keyword
        (half-length of the simulation box).
    dt : float
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m_small : float
        Bulk number density fraction of small monomers.
    phi_bulk_m_small : float
        Bulk volume fraction of small monomers
    rho_bulk_m_large : float
        Bulk number density fraction of large monomers.
    phi_bulk_m_large : float
        Bulk volume fraction of large monomers
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Notes
    -----
    The mass density is uniform across all species. For any species whose mass
    is not explicitly parsed, the mass is defined as :math:`m_{i} = d_{i}^3`,
    where :math:`d_{i}` represents the species' diameter.

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = TransFociCub(
    ... 'al5.0nl5ml125.0ns400ac1.0nc0l100.0dt0.005bdump2000adump5000.ring',
    ... 'ensemble_long',
    ... 'all'
    ... )
    >>> print(artifact.ncrowd)
    0
    """
    _geometry = 'cubic'
    _topology = 'ring'
    _groups = ['bug', 'all']

    _genealogy_attributes = {
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.ring
        'segment': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'lcube': 'l', 'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#ens#.j#.ring
        'whole': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc', 'lcube': 'l',
             'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump',
             'ensemble_id': 'ens'}),
        # Pattern: al#nl#ml#ns#ac#nc#l#dt#bdump#adump#
        'ensemble_long': OrderedDict(
            {'dmon_large': 'al', 'nmon_large': 'nl', 'mmon_large': 'ml',
             'nmon_small': 'ns', 'dcrowd': 'ac', 'ncrowd': 'nc',
             'lcube': 'l', 'dt': 'dt', 'bdump': 'bdump', 'adump': 'adump'}),
        # Pattern: ns#nl#al#ac#nc#
        'ensemble': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcrowd': 'ac', 'ncrowd': 'nc'}),
        # Pattern: ns#nl#al#ac#
        'space': OrderedDict(
            {'nmon_small': 'ns', 'nmon_large': 'nl', 'dmon_large': 'al',
             'dcrowd': 'ac'})
    }

    _project_attributes = {
        'segment': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                    'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                    'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'whole': ['dmon_small', 'mmon_small', 'mcrowd', 'phi_bulk_m_small',
                  'rho_bulk_m_small', 'phi_bulk_m_large', 'phi_bulk_m_large',
                  'rho_bulk_m', 'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble_long': ['dmon_small', 'mmon_small',  'mcrowd',
                          'phi_bulk_m_small', 'rho_bulk_m_small',
                          'phi_bulk_m_large', 'phi_bulk_m_large', 'rho_bulk_m',
                          'rho_bulk_m', 'phi_bulk_c', 'rho_bulk_c'],
        'ensemble': ['dmon_small', 'mmon_small', 'mcrowd'],
        'space': ['dmon_small', 'mmon_small', 'mcrowd']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon_small: float = 1
        self.mmon_small: float = self.dmon_small**3
        self.mcrowd: float = -1
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m_small: float = -1
            self.rho_bulk_m_small: float = -1
            self.phi_bulk_m_large: float = -1
            self.rho_bulk_m_large: float = -1
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['dmon_large', 'lcube', 'mmon_large', 'dcrowd', 'dt']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'l':
                    # Cube full side from its half-side
                    setattr(self, attr, 2 * getattr(self, attr))
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m_small = number_density_cube(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m_small = volume_fraction_cube(
            getattr(self, 'nmon_small'),
            getattr(self, 'dmon_small'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_m_large = number_density_cube(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m_large = volume_fraction_cube(
            getattr(self, 'nmon_large'),
            getattr(self, 'dmon_large'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_m = self.rho_bulk_m_small + self.rho_bulk_m_large
        self.phi_bulk_m = self.phi_bulk_m_small + self.phi_bulk_m_large

        self.mcrowd = getattr(self, 'dcrowd') ** 3
        self.rho_bulk_c = number_density_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )


class HnsCub(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *HnsCub* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: N#kbmm#nh#ac#l#epshc#nc#ens#.#j#.ring
      One of multiple chunks of a complete artifact.
    - `whole`: N#kbmm#nh#ac#l#epshc#nc#ens#.ring
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: N#kbmm#nh#ac#l#epshc#nc#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: N#kbmm#nh#ac#epshc#nc#
      Short name for an 'ensemble' artifact.
    - `space`: N#kbmm#nh#ac#epshc#ring
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'nucleoid', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon: float
        Size (diameter) of a monomer
    nmon: int
        number of small monomers. Its associated keyword is 'ns'.
    dhns: float, default 1.0
        Size (diameter) of a H-NS protein.
    dhns_patch: float, default 0.178
        Size (diameter) of a H-NS protein patch at its pole.
    nhns: int
        Number of H-NS protein. Its associated keyword is 'nh'.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    bend_mm: float
        Bending rigidity of DNA monomers. Its associated keyword is 'kbmm'.
    eps_hm: float, default 29.0
        The strength of attractive LJ interaction between hns poles and
        monomers.
    eps_hc: float
        The strength of attractive LJ interaction between H-NS cores and
        crowders. Its associated keyword is 'epshc'.
    lcube : float
        Length of the simulation box, inferred from 'l' keyword
        (half-length of the simulation box).
    dt : float, default 0.005
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int, default 5000
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int, default 10000
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_hns : float
        Bulk number density fraction of H-NS proteins.
    phi_bulk_hns : float
        Bulk volume fraction of H-NS proteins.
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = HnsCub(
    ... 'N200kbmm2.0nh8ac1.0l25.0epshc1.0nc0ens1.ring',
    ... 'whole',
    ... 'nucleoid'
    ... )
    >>> print(artifact.nhns)
    8
    """
    _geometry = 'cubic'
    _topology = 'ring'
    _groups = ['nucleoid', 'all']

    _genealogy_attributes = {
        # Pattern: N#kbmm#nh#ac#l#epshc#nc#ens#.#j#.ring
        'segment': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'nhns': 'nh', 'dcrowd': 'ac',
             'lcube': 'l', 'eps_hc': 'epshc', 'ncrowd': 'nc',
             'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: N#kbmm#nh#ac#l#epshc#nc#ens#.ring
        'whole': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'nhns': 'nh', 'dcrowd': 'ac',
             'lcube': 'l', 'eps_hc': 'epshc', 'ncrowd': 'nc',
             'ensemble_id': 'ens'}),
        # Pattern: N#kbmm#nh#ac#l#epshc#nc#
        'ensemble_long': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'nhns': 'nh', 'dcrowd': 'ac',
             'lcube': 'l', 'eps_hc': 'epshc', 'ncrowd': 'nc'}),
        # Pattern: N#kbmm#nh#ac#epshc#nc#
        'ensemble': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'nhns': 'nh', 'dcrowd': 'ac',
             'eps_hc': 'epshc', 'ncrowd': 'nc'}),
        # Pattern: N#kbmm#nh#ac#epshc#
        'space': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'nhns': 'nh', 'dcrowd': 'ac',
             'eps_hc': 'epshc'})
    }

    _project_attributes = {
        'segment': ['dmon', 'dhns', 'phi_bulk_m', 'rho_bulk_m',
                    'phi_bulk_c', 'rho_bulk_c', 'phi_bulk_hns',
                    'rho_bulk_hns', 'dt', 'ndump', 'adump', 'eps_hm'],
        'whole': ['dmon', 'dhns', 'phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c',
                  'rho_bulk_c', 'phi_bulk_hns', 'rho_bulk_hns', 'dt', 'ndump',
                  'adump', 'eps_hm'],
        'ensemble_long': ['dmon', 'dhns', 'phi_bulk_m', 'rho_bulk_m',
                          'phi_bulk_c', 'rho_bulk_c', 'phi_bulk_hns',
                          'rho_bulk_hns', 'dt', 'ndump', 'adump', 'eps_hm'],
        'ensemble': ['dmon', 'dhns', 'dt', 'ndump', 'adump', 'eps_hm'],
        'space': ['dmon', 'dhns', 'dt', 'ndump', 'adump', 'eps_hm']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon: float = 1
        self.dhns: float = 1
        self.dhns_patch: float = 0.178
        self.dt: float = 0.005
        self.bdump: int = 5000
        self.adump: int = 10000
        self.eps_hm: float = 29
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_hns: float = -1
            self.rho_bulk_hns: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['kbmm', 'lcube', 'dcrowd', 'epshc']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'l':
                    # Cube full side from its half-side
                    setattr(self, attr, 2 * getattr(self, attr))
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m = number_density_cube(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_m = volume_fraction_cube(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_hns = number_density_cube(
            getattr(self, 'nhns'),
            getattr(self, 'dhns'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_hns = volume_fraction_cube(
            getattr(self, 'nhns'),
            getattr(self, 'dhns'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.rho_bulk_c = number_density_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cube(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcube'),
            pbc=True
        )


class HnsCyl(ParserBase):
    """
    Extract structured information about an artifact from its name in the
    *HnsCyl* project, utilizing specific filename patterns.

    Each lineage level has a unique naming pattern used to parse key physical
    and system attributes:

    - `segment`: N#kbmm#r#nh#ac#lz#epshc#nc#ens#.j#.ring
      One of multiple chunks of a complete artifact.
    - `whole`: N#kbmm#r#nh#ac#lz#epshc#nc#ens#.ring
      A complete artifact. It may be a collection of segments.
    - `ensemble_long`: N#kbmm#r#nh#ac#lz#epshc#nc#
      Detailed name for an 'ensemble' artifact.
    - `ensemble`: N#D#nh#ac#epshc#nc#
      Short name for an 'ensemble' artifact.
    - `space`: N#D#nh#ac#epshc#nc#
      A 'space' artifact.

    For the above four lineages, the short names (keywords) are physical
    attributes where their values (shown by '#' sign) are float or integer
    number. See `genealogy_attributes` below for long names of attributes.

    Other than attributes inherited from the parent class `ParserBase`, this
    class dynamically defines new attributes based on the list of physical
    attributes of a given `lineage` as define in the `genealogy_attributes`
    class attribute.

    Parameters
    ----------
    artifact : str
        Name to be parsed, either a filename or filepath.
    lineage : {'segment', 'whole', 'ensemble_long', 'ensemble', 'space'}
        Type of the lineage of the name.
    group : {'nucleoid', 'all'}
        Particle group type, with `bug` representing a single polymer.

    Attributes
    ----------
    dmon: float
        Size (diameter) of a monomer
    nmon: int
        number of small monomers. Its associated keyword is 'ns'.
    dhns: float, default 1.0
        Size (diameter) of a H-NS protein.
    dhns_patch: float, default 0.178
        Size (diameter) of a H-NS protein patch at its pole.
    nhns: int
        Number of H-NS protein. Its associated keyword is 'nh'.
    dcrowd: float
        Size (diameter) of a crowder. Its associated keyword is 'ac'.
    ncrowd : int
        Number of crowders. Its associated keyword is 'nc'.
    bend_mm: float
        Bending rigidity of DNA monomers. Its associated keyword is 'kbmm'.
    eps_hm: float, default 29.0
        The strength of attractive LJ interaction between hns poles and
        monomers.
    eps_hc: float
        The strength of attractive LJ interaction between H-NS cores and
        crowders. Its associated keyword is 'epshc'.
    lcyl : float
        Length of the cylindrical confinement along z axis (the periodic,
        direction), inferred from 'lz' keyword (half of the length of the
        cylindrical confinement along z axis).
    dcyl : float
        Size (or diameter) of the cylindrical confinement, inferred
        from either 'r' keyword (the radius of a cylindrical confinement
        with open ends) or 'D' keyword (size of that confinement).
    dt : float, default 0.005
        Simulation timestep. Its associated keyword is 'dt'.
    bdump : int, default 5000
        Frequency by which 'bug' configurations are dumped in a 'bug'
        trajectory file. Its associated keyword is 'bdump'.
    adump : int, default 10000
        Frequency by which 'all' configurations are dumped in a 'segment'
        trajectory file. Its associated keyword is 'adump'.
    ensemble_id : int
        The ensemble number of a 'whole' artifact in an ensemble. Its
        associated keyword is 'ens'.
    segment_id : int
        The 'segment_id' keyword starts with 'j', ends with a 'padded'
        number such as '05' or '14', showing the succession of segments
        in a artifact file. Its associated keyword is 'j'.
    rho_bulk_m : float
        Bulk number density fraction of monomers.
    phi_bulk_m : float
        Bulk volume fraction of monomers
    rho_bulk_hns : float
        Bulk number density fraction of H-NS proteins.
    phi_bulk_hns : float
        Bulk volume fraction of H-NS proteins.
    rho_bulk_c : float
        Bulk number density fraction of crowders
    phi_bulk_c : float
        Bulk volume fraction of crowders
    space : str
        A space's name.
    ensemble : str, 'N/A'
        An ensemble's name if applicable, otherwise 'N/A'
    ensemble_long : str, 'N/A'
        The name of ensemble derived from 'whole' name if applicable,
        otherwise 'N/A'
    whole : str, 'N/A'
        A whole's name if applicable, otherwise 'N/A'
    segment : str, 'N/A'
        A segment's name if applicable, otherwise 'N/A'

    Notes
    -----
    The cylindrical wall is implemented in LAMMPS by using wall-forming
    particles of size 1.0. Thus, the actual size of the cylinder size
    (diameter), :math:`D`, is :math:`D=2r-1.0`,  :math:`r` is the radius of
    the cylindrical region defined in LAMMPS.

    Examples
    --------
    Creating a instance to parse a filename with specified lineage and group.

    >>> artifact = HnsCyl(
    ... 'N200D20.0nh16ac1.0epshc1.0nc0',
    ... 'space',
    ... 'nucleoid'
    ... )
    >>> print(artifact.eps_hc)
    1.0
    """
    _geometry = 'cylindrical'
    _topology = 'ring'
    _groups = ['nucleoid', 'all']

    _genealogy_attributes = {
        # Pattern: N#kbmm#r#nh#ac#lz#epshc#nc#ens#.j#.ring
        'segment': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'dcyl': 'r', 'nhns': 'nh',
             'dcrowd': 'ac', 'lcyl': 'lz', 'eps_hc': 'epshc', 'ncrowd': 'nc',
             'ensemble_id': 'ens', 'segment_id': 'j'}),
        # Pattern: N#kbmm#r#nh#ac#lz#epshc#nc#ens#.ring
        'whole': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'dcyl': 'r', 'nhns': 'nh',
             'dcrowd': 'ac', 'lcyl': 'lz', 'eps_hc': 'epshc', 'ncrowd': 'nc',
             'ensemble_id': 'ens'}),
        # Pattern: N#kbmm#r#nh#ac#lz#epshc#nc#
        'ensemble_long': OrderedDict(
            {'nmon': 'N', 'bend_mm': 'kbmm', 'dcyl': 'r', 'nhns': 'nh',
             'dcrowd': 'ac', 'lcyl': 'lz', 'eps_hc': 'epshc', 'ncrowd': 'nc'}),
        # Pattern: N#D#nh#ac#epshc#nc#
        'ensemble': OrderedDict(
            {'nmon': 'N', 'dcyl': 'D', 'nhns': 'nh', 'dcrowd': 'ac',
             'eps_hc': 'epshc', 'ncrowd': 'nc'}),
        # Pattern: N#D#nh#ac#epshc#nc#
        'space': OrderedDict(
            {'nmon': 'N', 'dcyl': 'D', 'nhns': 'nh', 'dcrowd': 'ac',
             'eps_hc': 'epshc'})
    }

    _project_attributes = {
        'segment': ['dmon', 'dhns', 'phi_bulk_m', 'rho_bulk_m',
                    'phi_bulk_c', 'rho_bulk_c', 'phi_bulk_hns',
                    'rho_bulk_hns', 'dt', 'ndump', 'adump', 'eps_hm'],
        'whole': ['dmon', 'dhns', 'phi_bulk_m', 'rho_bulk_m', 'phi_bulk_c',
                  'rho_bulk_c', 'phi_bulk_hns', 'rho_bulk_hns', 'dt', 'ndump',
                  'adump', 'eps_hm'],
        'ensemble_long': ['dmon', 'dhns', 'phi_bulk_m', 'rho_bulk_m',
                          'phi_bulk_c', 'rho_bulk_c', 'phi_bulk_hns',
                          'rho_bulk_hns', 'dt', 'ndump', 'adump', 'eps_hm'],
        'ensemble': ['dmon', 'dhns', 'dt', 'ndump', 'adump', 'eps_hm'],
        'space': ['dmon', 'dhns', 'dt', 'ndump', 'adump', 'eps_hm']
    }

    def __init__(
        self,
        artifact: str,
        lineage: LineageT,
        group: GroupT
    ) -> None:
        super().__init__(artifact, lineage, group)
        self._initiate_attributes()
        self._parse_name()
        self._set_parents()
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self._dependent_attributes()

    def _initiate_attributes(self) -> None:
        """
        Define and initiates the project attributes.

        Notes
        -----
        The negative initial values are unphysical.
        """
        self.dmon: float = 1
        self.dhns: float = 1
        self.dhns_patch: float = 0.178
        self.dt: float = 0.005
        self.bdump: int = 5000
        self.adump: int = 10000
        self.eps_hm: float = 29
        if self.lineage in ['segment', 'whole', 'ensemble_long']:
            self.phi_bulk_m: float = -1
            self.rho_bulk_m: float = -1
            self.phi_bulk_hns: float = -1
            self.rho_bulk_hns: float = -1
            self.phi_bulk_c: float = -1
            self.rho_bulk_c: float = -1

    def _parse_name(self) -> None:
        """
        Parse lineage attributes from the `name` attribute, assigning them
        dynamically as class attributes.

        Notes
        -----
        Lineage attributes are macroscopic physical attributes of the systems.
        They are added to the class dynamically as new class attribute upon
        class instantiation.
        """
        name_strs = re.compile(r"([a-zA-Z\-]+)")
        words = name_strs.split(self.name)
        attrs_float = ['bend_mm', 'dcyl', 'lcyl', 'dcrowd', 'eps_hc']
        for attr, keyword in self.genealogy_attributes[self.lineage].items():
            try:
                val = words[words.index(keyword) + 1]
                setattr(self,
                        attr,
                        float(val) if attr in attrs_float else int(float(val)))
                if keyword == 'lz':
                    # Cylinder full length from its half-length
                    setattr(self, attr, 2 * getattr(self, attr))
                if keyword == 'r':
                    # Cylinder size is twice its radius, correcting for
                    # wall-forming particles with size 1
                    setattr(self, attr, 2 * getattr(self, attr) - 1.0)
            except ValueError:
                print(f"'{keyword}' attribute not found in '{self.name}'")

    def _dependent_attributes(self) -> None:
        """
        Calculate system attributes based on parsed values.
        """
        self.rho_bulk_m = number_density_cylinder(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_m = volume_fraction_cylinder(
            getattr(self, 'nmon'),
            getattr(self, 'dmon'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.rho_bulk_hns = number_density_cylinder(
            getattr(self, 'nhns'),
            getattr(self, 'dhns'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_hns = volume_fraction_cylinder(
            getattr(self, 'nhns'),
            getattr(self, 'dhns'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.rho_bulk_c = number_density_cylinder(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
        self.phi_bulk_c = volume_fraction_cylinder(
            getattr(self, 'ncrowd'),
            getattr(self, 'dcrowd'),
            getattr(self, 'lcyl'),
            getattr(self, 'dcyl'),
            pbc=True
        )
