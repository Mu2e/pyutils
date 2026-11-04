
"""pyutils

Lightweight package metadata and exports.

This module purposefully avoids importing heavy submodules at import
time to keep the package light-weight for users who only need a
minor part of the library.
"""

__version__ = "1.9.0"  # keep in sync with setup.py

__all__ = ["__version__"]
