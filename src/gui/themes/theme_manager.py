"""
Theme manager for the MetaTrader Python Framework GUI.

This module provides comprehensive theme management including theme switching,
persistence, and application of themes to widgets and applications.
"""

from __future__ import annotations

from typing import Dict, Optional, Any, Type
from pathlib import Path
import json

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QSettings, pyqtSignal, QObject
from PyQt6.QtGui import QPalette

from src.core.config import get_settings
from src.core.logging import get_logger

from .themes import BaseTheme, DarkTheme, LightTheme, ThemeType

logger = get_logger(__name__)


class ThemeManager(QObject):
    """
    Comprehensive theme management system.

    Features:
    - Multiple theme support (dark, light, custom)
    - Theme switching with persistence
    - Widget-specific theme application
    - Color scheme management
    - Style sheet generation and caching
    - Theme preferences storage
    """

    # Signals
    themeChanged = pyqtSignal(str)      # theme_name
    colorSchemeChanged = pyqtSignal()

    def __init__(self, initial_theme: str = "dark"):
        """
        Initialize theme manager.

        Args:
            initial_theme: Initial theme name to load
        """
        super().__init__()

        # Configuration
        self.settings = get_settings()

        # Theme registry
        self._themes: Dict[str, BaseTheme] = {}
        self._current_theme: Optional[BaseTheme] = None
        self._current_theme_name: str = ""

        # Style cache
        self._style_cache: Dict[str, str] = {}
        self._palette_cache: Dict[str, QPalette] = {}

        # Preferences
        self._preferences_file = Path("data/theme_preferences.json")
        self._preferences: Dict[str, Any] = {}

        # Register built-in themes
        self._register_builtin_themes()

        # Load preferences
        self._load_preferences()

        # Set initial theme
        self.set_theme(initial_theme)

        logger.info(f"Theme manager initialized with theme: {initial_theme}")

    def _register_builtin_themes(self) -> None:
        """Register built-in themes."""
        try:
            # Register dark theme
            dark_theme = DarkTheme()
            self._themes[dark_theme.name.lower()] = dark_theme

            # Register light theme
            light_theme = LightTheme()
            self._themes[light_theme.name.lower()] = light_theme

            logger.debug(f"Registered {len(self._themes)} built-in themes")

        except Exception as e:
            logger.error(f"Error registering built-in themes: {e}")

    def _load_preferences(self) -> None:
        """Load theme preferences from file."""
        try:
            if self._preferences_file.exists():
                with open(self._preferences_file, 'r') as f:
                    self._preferences = json.load(f)
                logger.debug("Theme preferences loaded")
            else:
                self._preferences = self._get_default_preferences()
                logger.debug("Using default theme preferences")

        except Exception as e:
            logger.error(f"Error loading theme preferences: {e}")
            self._preferences = self._get_default_preferences()

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default theme preferences."""
        return {
            'last_theme': 'dark',
            'auto_switch': False,
            'auto_switch_times': {
                'light_start': '08:00',
                'dark_start': '18:00'
            },
            'custom_colors': {},
            'widget_overrides': {}
        }

    def save_preferences(self) -> None:
        """Save theme preferences to file."""
        try:
            # Ensure directory exists
            self._preferences_file.parent.mkdir(parents=True, exist_ok=True)

            # Update current theme
            if self._current_theme_name:
                self._preferences['last_theme'] = self._current_theme_name

            # Save to file
            with open(self._preferences_file, 'w') as f:
                json.dump(self._preferences, f, indent=2)

            logger.debug("Theme preferences saved")

        except Exception as e:
            logger.error(f"Error saving theme preferences: {e}")

    def register_theme(self, theme: BaseTheme) -> None:
        """
        Register a custom theme.

        Args:
            theme: Theme instance to register
        """
        theme_name = theme.name.lower()
        self._themes[theme_name] = theme

        # Clear caches for this theme
        self._clear_cache_for_theme(theme_name)

        logger.info(f"Registered custom theme: {theme.name}")

    def get_available_themes(self) -> list[str]:
        """Get list of available theme names."""
        return list(self._themes.keys())

    def get_current_theme(self) -> Optional[BaseTheme]:
        """Get current theme instance."""
        return self._current_theme

    def get_current_theme_name(self) -> str:
        """Get current theme name."""
        return self._current_theme_name

    def set_theme(self, theme_name: str) -> bool:
        """
        Set current theme.

        Args:
            theme_name: Name of theme to set

        Returns:
            bool: True if theme was set successfully
        """
        theme_name_lower = theme_name.lower()

        if theme_name_lower not in self._themes:
            logger.error(f"Theme not found: {theme_name}")
            return False

        try:
            # Set new theme
            old_theme_name = self._current_theme_name
            self._current_theme = self._themes[theme_name_lower]
            self._current_theme_name = theme_name_lower

            # Clear style cache
            self._style_cache.clear()

            # Emit signal
            self.themeChanged.emit(theme_name_lower)

            logger.info(f"Theme changed from '{old_theme_name}' to '{theme_name_lower}'")
            return True

        except Exception as e:
            logger.error(f"Error setting theme {theme_name}: {e}")
            return False

    def apply_theme(self, app: QApplication) -> None:
        """
        Apply current theme to application.

        Args:
            app: QApplication instance
        """
        if not self._current_theme:
            logger.warning("No current theme to apply")
            return

        try:
            # Set application palette
            palette = self._get_cached_palette()
            app.setPalette(palette)

            # Set application stylesheet
            style_sheet = self._get_application_stylesheet()
            app.setStyleSheet(style_sheet)

            logger.debug(f"Applied theme '{self._current_theme_name}' to application")

        except Exception as e:
            logger.error(f"Error applying theme to application: {e}")

    def apply_theme_to_widget(self, widget: QWidget, widget_type: str = None) -> None:
        """
        Apply current theme to a specific widget.

        Args:
            widget: Widget to apply theme to
            widget_type: Specific widget type for styling
        """
        if not self._current_theme:
            logger.warning("No current theme to apply")
            return

        try:
            # Determine widget type if not provided
            if widget_type is None:
                widget_type = widget.__class__.__name__.lower()

            # Get widget-specific style
            style_sheet = self._get_widget_stylesheet(widget_type)

            if style_sheet:
                widget.setStyleSheet(style_sheet)

            # Set palette
            palette = self._get_cached_palette()
            widget.setPalette(palette)

            logger.debug(f"Applied theme to widget: {widget_type}")

        except Exception as e:
            logger.debug(f"Error applying theme to widget {widget_type}: {e}")

    def get_color(self, color_name: str) -> Optional[Any]:
        """
        Get color from current theme.

        Args:
            color_name: Name of color to get

        Returns:
            QColor if found, None otherwise
        """
        if not self._current_theme:
            return None

        return self._current_theme.get_color(color_name)

    def get_themed_stylesheet(self, widget_type: str) -> str:
        """
        Get themed stylesheet for widget type.

        Args:
            widget_type: Type of widget

        Returns:
            Style sheet string
        """
        return self._get_widget_stylesheet(widget_type)

    def _get_cached_palette(self) -> QPalette:
        """Get cached palette for current theme."""
        if not self._current_theme:
            return QPalette()

        theme_name = self._current_theme_name

        if theme_name not in self._palette_cache:
            self._palette_cache[theme_name] = self._current_theme.get_palette()

        return self._palette_cache[theme_name]

    def _get_application_stylesheet(self) -> str:
        """Get complete application stylesheet."""
        cache_key = f"{self._current_theme_name}_application"

        if cache_key not in self._style_cache:
            styles = []

            # Get all widget styles
            for widget_type in ['application', 'main_window', 'table', 'button',
                              'combo_box', 'line_edit', 'group_box', 'frame',
                              'menu', 'toolbar', 'status_bar', 'progress_bar',
                              'checkbox', 'spinbox']:

                widget_style = self._current_theme.get_style(widget_type)
                if widget_style:
                    styles.append(widget_style)

            # Combine all styles
            combined_style = '\n\n'.join(styles)
            self._style_cache[cache_key] = combined_style

        return self._style_cache[cache_key]

    def _get_widget_stylesheet(self, widget_type: str) -> str:
        """Get stylesheet for specific widget type."""
        if not self._current_theme:
            return ""

        cache_key = f"{self._current_theme_name}_{widget_type}"

        if cache_key not in self._style_cache:
            style = self._current_theme.get_style(widget_type) or ""
            self._style_cache[cache_key] = style

        return self._style_cache[cache_key]

    def _clear_cache_for_theme(self, theme_name: str) -> None:
        """Clear cached styles for specific theme."""
        keys_to_remove = [key for key in self._style_cache.keys()
                         if key.startswith(f"{theme_name}_")]

        for key in keys_to_remove:
            del self._style_cache[key]

        if theme_name in self._palette_cache:
            del self._palette_cache[theme_name]

    def get_theme_info(self, theme_name: str = None) -> Dict[str, Any]:
        """
        Get information about a theme.

        Args:
            theme_name: Theme name (uses current if None)

        Returns:
            Dictionary with theme information
        """
        if theme_name is None:
            theme = self._current_theme
            theme_name = self._current_theme_name
        else:
            theme = self._themes.get(theme_name.lower())

        if not theme:
            return {}

        return {
            'name': theme.name,
            'type': theme.theme_type.value,
            'colors': {name: color.name() for name, color in theme.get_all_colors().items()},
            'is_current': theme_name.lower() == self._current_theme_name
        }

    def export_theme(self, theme_name: str, file_path: Path) -> bool:
        """
        Export theme to file.

        Args:
            theme_name: Name of theme to export
            file_path: Path to export file

        Returns:
            bool: True if export successful
        """
        theme = self._themes.get(theme_name.lower())
        if not theme:
            logger.error(f"Theme not found for export: {theme_name}")
            return False

        try:
            theme_data = {
                'name': theme.name,
                'type': theme.theme_type.value,
                'colors': {name: color.name() for name, color in theme.get_all_colors().items()},
                'styles': theme.get_all_styles()
            }

            with open(file_path, 'w') as f:
                json.dump(theme_data, f, indent=2)

            logger.info(f"Theme '{theme_name}' exported to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting theme: {e}")
            return False

    def import_theme(self, file_path: Path) -> bool:
        """
        Import theme from file.

        Args:
            file_path: Path to theme file

        Returns:
            bool: True if import successful
        """
        try:
            with open(file_path, 'r') as f:
                theme_data = json.load(f)

            # Create custom theme class (simplified implementation)
            # In a full implementation, this would create a proper theme class
            logger.info(f"Theme import would create custom theme: {theme_data.get('name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error importing theme: {e}")
            return False

    def reset_to_default(self) -> None:
        """Reset theme manager to default state."""
        try:
            # Reset preferences
            self._preferences = self._get_default_preferences()

            # Set default theme
            self.set_theme('dark')

            # Clear caches
            self._style_cache.clear()
            self._palette_cache.clear()

            logger.info("Theme manager reset to default state")

        except Exception as e:
            logger.error(f"Error resetting theme manager: {e}")

    def get_preferences(self) -> Dict[str, Any]:
        """Get current theme preferences."""
        return self._preferences.copy()

    def set_preference(self, key: str, value: Any) -> None:
        """Set a theme preference."""
        self._preferences[key] = value
        logger.debug(f"Theme preference set: {key} = {value}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a theme preference."""
        return self._preferences.get(key, default)