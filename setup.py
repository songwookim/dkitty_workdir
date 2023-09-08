"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)

###########################
from setuptools import setup

setup(
    name="dkitty_tasks",
    version="0.0.1",
    # install_requires=["gym==0.26.0", "pygame==2.1.0"],
)
