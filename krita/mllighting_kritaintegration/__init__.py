import sys
import os


def _setup_pythonpath():
    """Add Python module from the PYTHONPATH."""
    _syspath = set(sys.path)
    pythonpath = os.environ.get('PYTHONPATH', '')
    for path in reversed(pythonpath.split(os.pathsep)):
        path = path.strip()
        # Skip already loaded paths.
        if not path or path in _syspath:
            continue
        # Insert afer the Krita python libs.
        sys.path.insert(1, path)
        _syspath.add(path)


_setup_pythonpath()

# Initialize the logging at module initialization.
from mllighting_kritaintegration import log  # noqa: E402
from mllighting_kritaintegration.main import *  # noqa: E402 F401 F403

__all__ = ['log']
