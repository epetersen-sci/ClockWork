"""ClockWork: analysis of Drosophila Activity Monitor (DAM) data.

``clockwork.core`` computes, ``clockwork.app`` is the Streamlit GUI, and
``clockwork.cli`` is the ``clockwork`` command.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("clockwork-sci")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0+unknown"
