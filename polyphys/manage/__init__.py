"""
The `manage` subpackage does research data management by handling and
combining a variety of simulation files based on templates for their
filenames, and then organizes them into a hierarchy of directories.
"""
from polyphys.manage import parser
from polyphys.manage import utils
from polyphys.manage import types
from polyphys.manage import organizer

__all__ = ["parser", "utils", "types", "organizer"]
