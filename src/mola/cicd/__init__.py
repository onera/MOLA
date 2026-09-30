"""
MOLA CI/CD utilities for example execution and validation.

This module provides generic CI/CD utilities that can be used by any MOLA user
for automated testing and validation of examples.
"""

from .examples_runner import ExampleCIState, detect_execution_mode, main

__all__ = ['ExampleCIState', 'detect_execution_mode', 'main']