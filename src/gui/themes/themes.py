"""
Theme definitions for the MetaTrader Python Framework GUI.

This module defines various themes including dark and light themes
with comprehensive color schemes and styling rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette

from src.core.logging import get_logger

logger = None  # Will be initialized after LoggerFactory setup


def _ensure_logger():
    """Ensure logger is initialized."""
    global logger
    if logger is None:
        logger = get_logger(__name__)


class ThemeType(Enum):
    """Theme type enumeration."""
    DARK = "dark"
    LIGHT = "light"


class BaseTheme(ABC):
    """
    Base class for all themes.

    Defines the interface for theme implementations and common functionality.
    """

    def __init__(self, name: str, theme_type: ThemeType):
        """
        Initialize base theme.

        Args:
            name: Theme display name
            theme_type: Theme type
        """
        self.name = name
        self.theme_type = theme_type
        self._colors = self._define_colors()
        self._styles = self._define_styles()

    @abstractmethod
    def _define_colors(self) -> Dict[str, QColor]:
        """Define color scheme for the theme."""
        pass

    @abstractmethod
    def _define_styles(self) -> Dict[str, str]:
        """Define style sheets for different widgets."""
        pass

    def get_color(self, color_name: str) -> Optional[QColor]:
        """Get color by name."""
        return self._colors.get(color_name)

    def get_style(self, widget_type: str) -> Optional[str]:
        """Get style sheet for widget type."""
        return self._styles.get(widget_type)

    def get_palette(self) -> QPalette:
        """Get QPalette for this theme."""
        palette = QPalette()

        # Set basic colors
        if 'background' in self._colors:
            palette.setColor(QPalette.ColorRole.Window, self._colors['background'])
            palette.setColor(QPalette.ColorRole.Base, self._colors['background'])

        if 'foreground' in self._colors:
            palette.setColor(QPalette.ColorRole.WindowText, self._colors['foreground'])
            palette.setColor(QPalette.ColorRole.Text, self._colors['foreground'])

        if 'selection' in self._colors:
            palette.setColor(QPalette.ColorRole.Highlight, self._colors['selection'])

        if 'selection_text' in self._colors:
            palette.setColor(QPalette.ColorRole.HighlightedText, self._colors['selection_text'])

        return palette

    def get_all_colors(self) -> Dict[str, QColor]:
        """Get all colors in the theme."""
        return self._colors.copy()

    def get_all_styles(self) -> Dict[str, str]:
        """Get all styles in the theme."""
        return self._styles.copy()


class DarkTheme(BaseTheme):
    """
    Dark theme implementation optimized for trading applications.

    Features a dark background with high contrast elements
    and colors optimized for extended use.
    """

    def __init__(self):
        """Initialize dark theme."""
        super().__init__("Dark", ThemeType.DARK)

    def _define_colors(self) -> Dict[str, QColor]:
        """Define dark theme colors."""
        return {
            # Base colors
            'background': QColor(45, 45, 48),           # Dark gray
            'surface': QColor(37, 37, 38),              # Darker gray
            'panel': QColor(51, 51, 55),                # Panel background
            'border': QColor(63, 63, 70),               # Borders
            'foreground': QColor(255, 255, 255),        # White text
            'secondary_text': QColor(204, 204, 204),    # Gray text
            'disabled_text': QColor(109, 109, 109),     # Disabled text

            # Interactive colors
            'selection': QColor(0, 120, 215),           # Blue selection
            'selection_text': QColor(255, 255, 255),    # White on selection
            'hover': QColor(70, 70, 75),                # Hover state
            'pressed': QColor(90, 90, 95),              # Pressed state

            # Status colors
            'success': QColor(40, 167, 69),             # Green
            'warning': QColor(255, 193, 7),             # Amber
            'error': QColor(220, 53, 69),               # Red
            'info': QColor(23, 162, 184),               # Cyan

            # Trading colors
            'buy': QColor(34, 139, 34),                 # Forest Green
            'sell': QColor(220, 20, 60),                # Crimson
            'neutral': QColor(255, 165, 0),             # Orange

            # Chart colors
            'grid': QColor(60, 60, 65),                 # Chart grid
            'axis': QColor(180, 180, 180),              # Chart axis
            'bullish': QColor(0, 150, 0),               # Bullish candle
            'bearish': QColor(200, 0, 0),               # Bearish candle

            # Data visualization
            'positive': QColor(40, 167, 69),            # Positive values
            'negative': QColor(220, 53, 69),            # Negative values
            'highlight': QColor(255, 235, 59),          # Highlight color

            # Special colors
            'accent': QColor(102, 187, 106),            # Accent color
            'link': QColor(100, 181, 246),              # Links
            'visited_link': QColor(156, 39, 176),       # Visited links
        }

    def _define_styles(self) -> Dict[str, str]:
        """Define dark theme styles."""
        colors = self._colors

        return {
            'application': f"""
                QWidget {{
                    background-color: {colors['background'].name()};
                    color: {colors['foreground'].name()};
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 9pt;
                }}

                QWidget:disabled {{
                    color: {colors['disabled_text'].name()};
                }}
            """,

            'main_window': f"""
                QMainWindow {{
                    background-color: {colors['background'].name()};
                }}

                QMainWindow::separator {{
                    background-color: {colors['border'].name()};
                    width: 1px;
                    height: 1px;
                }}
            """,

            'table': f"""
                QTableView {{
                    background-color: {colors['surface'].name()};
                    alternate-background-color: {colors['panel'].name()};
                    gridline-color: {colors['border'].name()};
                    selection-background-color: {colors['selection'].name()};
                    selection-color: {colors['selection_text'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QTableView::item {{
                    padding: 4px;
                    border-right: 1px solid {colors['border'].name()};
                }}

                QTableView::item:selected {{
                    background-color: {colors['selection'].name()};
                    color: {colors['selection_text'].name()};
                }}

                QTableView::item:hover {{
                    background-color: {colors['hover'].name()};
                }}

                QHeaderView::section {{
                    background-color: {colors['panel'].name()};
                    color: {colors['foreground'].name()};
                    padding: 6px;
                    border: 1px solid {colors['border'].name()};
                    font-weight: bold;
                }}

                QHeaderView::section:hover {{
                    background-color: {colors['hover'].name()};
                }}
            """,

            'button': f"""
                QPushButton {{
                    background-color: {colors['panel'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 6px 12px;
                    font-weight: bold;
                }}

                QPushButton:hover {{
                    background-color: {colors['hover'].name()};
                    border-color: {colors['selection'].name()};
                }}

                QPushButton:pressed {{
                    background-color: {colors['pressed'].name()};
                }}

                QPushButton:disabled {{
                    background-color: {colors['surface'].name()};
                    border-color: {colors['disabled_text'].name()};
                    color: {colors['disabled_text'].name()};
                }}
            """,

            'combo_box': f"""
                QComboBox {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 4px 8px;
                    min-width: 80px;
                }}

                QComboBox:hover {{
                    border-color: {colors['selection'].name()};
                }}

                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                }}

                QComboBox::down-arrow {{
                    image: url(down_arrow_dark.png);
                    width: 12px;
                    height: 12px;
                }}

                QComboBox QAbstractItemView {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    selection-background-color: {colors['selection'].name()};
                }}
            """,

            'line_edit': f"""
                QLineEdit {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 4px 8px;
                }}

                QLineEdit:focus {{
                    border-color: {colors['selection'].name()};
                }}

                QLineEdit:disabled {{
                    background-color: {colors['background'].name()};
                    color: {colors['disabled_text'].name()};
                }}
            """,

            'group_box': f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    margin-top: 6px;
                    padding-top: 6px;
                }}

                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px 0 4px;
                    background-color: {colors['background'].name()};
                }}
            """,

            'frame': f"""
                QFrame[frameShape="4"] {{
                    border: 1px solid {colors['border'].name()};
                }}

                QFrame[frameShape="5"] {{
                    border: 1px solid {colors['border'].name()};
                }}
            """,

            'menu': f"""
                QMenuBar {{
                    background-color: {colors['panel'].name()};
                    border-bottom: 1px solid {colors['border'].name()};
                }}

                QMenuBar::item {{
                    background: transparent;
                    padding: 6px 8px;
                }}

                QMenuBar::item:selected {{
                    background-color: {colors['hover'].name()};
                }}

                QMenu {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QMenu::item {{
                    padding: 6px 24px 6px 8px;
                }}

                QMenu::item:selected {{
                    background-color: {colors['selection'].name()};
                }}

                QMenu::separator {{
                    height: 1px;
                    background-color: {colors['border'].name()};
                    margin: 2px 0px;
                }}
            """,

            'toolbar': f"""
                QToolBar {{
                    background-color: {colors['panel'].name()};
                    border: 1px solid {colors['border'].name()};
                    spacing: 2px;
                }}

                QToolBar::separator {{
                    background-color: {colors['border'].name()};
                    width: 1px;
                    margin: 2px;
                }}
            """,

            'status_bar': f"""
                QStatusBar {{
                    background-color: {colors['panel'].name()};
                    border-top: 1px solid {colors['border'].name()};
                }}

                QStatusBar::item {{
                    border: none;
                }}
            """,

            'progress_bar': f"""
                QProgressBar {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    text-align: center;
                }}

                QProgressBar::chunk {{
                    background-color: {colors['selection'].name()};
                    border-radius: 2px;
                }}
            """,

            'checkbox': f"""
                QCheckBox::indicator {{
                    width: 13px;
                    height: 13px;
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QCheckBox::indicator:checked {{
                    background-color: {colors['selection'].name()};
                }}

                QCheckBox::indicator:hover {{
                    border-color: {colors['selection'].name()};
                }}
            """,

            'spinbox': f"""
                QSpinBox {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 2px;
                }}

                QSpinBox:focus {{
                    border-color: {colors['selection'].name()};
                }}

                QSpinBox::up-button, QSpinBox::down-button {{
                    background-color: {colors['panel'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                    background-color: {colors['hover'].name()};
                }}
            """,
        }


class LightTheme(BaseTheme):
    """
    Light theme implementation for daytime use.

    Features a light background with good contrast
    and colors suitable for well-lit environments.
    """

    def __init__(self):
        """Initialize light theme."""
        super().__init__("Light", ThemeType.LIGHT)

    def _define_colors(self) -> Dict[str, QColor]:
        """Define light theme colors."""
        return {
            # Base colors
            'background': QColor(255, 255, 255),        # White
            'surface': QColor(248, 249, 250),           # Light gray
            'panel': QColor(241, 243, 244),             # Panel background
            'border': QColor(206, 212, 218),            # Borders
            'foreground': QColor(33, 37, 41),           # Dark text
            'secondary_text': QColor(108, 117, 125),    # Gray text
            'disabled_text': QColor(173, 181, 189),     # Disabled text

            # Interactive colors
            'selection': QColor(0, 123, 255),           # Blue selection
            'selection_text': QColor(255, 255, 255),    # White on selection
            'hover': QColor(233, 236, 239),             # Hover state
            'pressed': QColor(222, 226, 230),           # Pressed state

            # Status colors
            'success': QColor(40, 167, 69),             # Green
            'warning': QColor(255, 193, 7),             # Amber
            'error': QColor(220, 53, 69),               # Red
            'info': QColor(23, 162, 184),               # Cyan

            # Trading colors
            'buy': QColor(40, 167, 69),                 # Green
            'sell': QColor(220, 53, 69),                # Red
            'neutral': QColor(255, 193, 7),             # Amber

            # Chart colors
            'grid': QColor(222, 226, 230),              # Chart grid
            'axis': QColor(73, 80, 87),                 # Chart axis
            'bullish': QColor(40, 167, 69),             # Bullish candle
            'bearish': QColor(220, 53, 69),             # Bearish candle

            # Data visualization
            'positive': QColor(40, 167, 69),            # Positive values
            'negative': QColor(220, 53, 69),            # Negative values
            'highlight': QColor(255, 235, 59),          # Highlight color

            # Special colors
            'accent': QColor(0, 123, 255),              # Accent color
            'link': QColor(0, 123, 255),                # Links
            'visited_link': QColor(108, 117, 125),      # Visited links
        }

    def _define_styles(self) -> Dict[str, str]:
        """Define light theme styles."""
        colors = self._colors

        return {
            'application': f"""
                QWidget {{
                    background-color: {colors['background'].name()};
                    color: {colors['foreground'].name()};
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 9pt;
                }}

                QWidget:disabled {{
                    color: {colors['disabled_text'].name()};
                }}
            """,

            'main_window': f"""
                QMainWindow {{
                    background-color: {colors['background'].name()};
                }}

                QMainWindow::separator {{
                    background-color: {colors['border'].name()};
                    width: 1px;
                    height: 1px;
                }}
            """,

            'table': f"""
                QTableView {{
                    background-color: {colors['background'].name()};
                    alternate-background-color: {colors['surface'].name()};
                    gridline-color: {colors['border'].name()};
                    selection-background-color: {colors['selection'].name()};
                    selection-color: {colors['selection_text'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QTableView::item {{
                    padding: 4px;
                    border-right: 1px solid {colors['border'].name()};
                }}

                QTableView::item:selected {{
                    background-color: {colors['selection'].name()};
                    color: {colors['selection_text'].name()};
                }}

                QTableView::item:hover {{
                    background-color: {colors['hover'].name()};
                }}

                QHeaderView::section {{
                    background-color: {colors['panel'].name()};
                    color: {colors['foreground'].name()};
                    padding: 6px;
                    border: 1px solid {colors['border'].name()};
                    font-weight: bold;
                }}

                QHeaderView::section:hover {{
                    background-color: {colors['hover'].name()};
                }}
            """,

            'button': f"""
                QPushButton {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 6px 12px;
                }}

                QPushButton:hover {{
                    background-color: {colors['hover'].name()};
                    border-color: {colors['selection'].name()};
                }}

                QPushButton:pressed {{
                    background-color: {colors['pressed'].name()};
                }}

                QPushButton:disabled {{
                    background-color: {colors['panel'].name()};
                    border-color: {colors['disabled_text'].name()};
                    color: {colors['disabled_text'].name()};
                }}
            """,

            'combo_box': f"""
                QComboBox {{
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 4px 8px;
                    min-width: 80px;
                }}

                QComboBox:hover {{
                    border-color: {colors['selection'].name()};
                }}

                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                }}

                QComboBox QAbstractItemView {{
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                    selection-background-color: {colors['selection'].name()};
                }}
            """,

            'line_edit': f"""
                QLineEdit {{
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 4px 8px;
                }}

                QLineEdit:focus {{
                    border-color: {colors['selection'].name()};
                }}

                QLineEdit:disabled {{
                    background-color: {colors['surface'].name()};
                    color: {colors['disabled_text'].name()};
                }}
            """,

            'group_box': f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    margin-top: 6px;
                    padding-top: 6px;
                }}

                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px 0 4px;
                    background-color: {colors['background'].name()};
                }}
            """,

            'frame': f"""
                QFrame[frameShape="4"] {{
                    border: 1px solid {colors['border'].name()};
                }}

                QFrame[frameShape="5"] {{
                    border: 1px solid {colors['border'].name()};
                }}
            """,

            'menu': f"""
                QMenuBar {{
                    background-color: {colors['surface'].name()};
                    border-bottom: 1px solid {colors['border'].name()};
                }}

                QMenuBar::item {{
                    background: transparent;
                    padding: 6px 8px;
                }}

                QMenuBar::item:selected {{
                    background-color: {colors['hover'].name()};
                }}

                QMenu {{
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QMenu::item {{
                    padding: 6px 24px 6px 8px;
                }}

                QMenu::item:selected {{
                    background-color: {colors['selection'].name()};
                    color: {colors['selection_text'].name()};
                }}

                QMenu::separator {{
                    height: 1px;
                    background-color: {colors['border'].name()};
                    margin: 2px 0px;
                }}
            """,

            'toolbar': f"""
                QToolBar {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                    spacing: 2px;
                }}

                QToolBar::separator {{
                    background-color: {colors['border'].name()};
                    width: 1px;
                    margin: 2px;
                }}
            """,

            'status_bar': f"""
                QStatusBar {{
                    background-color: {colors['surface'].name()};
                    border-top: 1px solid {colors['border'].name()};
                }}

                QStatusBar::item {{
                    border: none;
                }}
            """,

            'progress_bar': f"""
                QProgressBar {{
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    text-align: center;
                }}

                QProgressBar::chunk {{
                    background-color: {colors['selection'].name()};
                    border-radius: 2px;
                }}
            """,

            'checkbox': f"""
                QCheckBox::indicator {{
                    width: 13px;
                    height: 13px;
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QCheckBox::indicator:checked {{
                    background-color: {colors['selection'].name()};
                }}

                QCheckBox::indicator:hover {{
                    border-color: {colors['selection'].name()};
                }}
            """,

            'spinbox': f"""
                QSpinBox {{
                    background-color: {colors['background'].name()};
                    border: 1px solid {colors['border'].name()};
                    border-radius: 3px;
                    padding: 2px;
                }}

                QSpinBox:focus {{
                    border-color: {colors['selection'].name()};
                }}

                QSpinBox::up-button, QSpinBox::down-button {{
                    background-color: {colors['surface'].name()};
                    border: 1px solid {colors['border'].name()};
                }}

                QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                    background-color: {colors['hover'].name()};
                }}
            """,
        }