"""
Localization system for the MetaTrader Python Framework GUI.

This package provides comprehensive internationalization support including
Hungarian and English localization, dynamic language switching, and translation management.
"""

from __future__ import annotations

from .localization_manager import LocalizationManager
from .translator import Translator

__all__ = [
    "LocalizationManager",
    "Translator",
]