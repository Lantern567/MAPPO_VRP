"""
VRP Scenarios package.
"""

import importlib.util
import os


def load(name):
    """
    Load a scenario module by name.

    Args:
        name: Scenario filename (e.g., 'truck_drone_basic.py')

    Returns:
        Loaded module
    """
    pathname = os.path.join(os.path.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location("scenario", pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
