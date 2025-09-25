"""
Localization manager for the MetaTrader Python Framework GUI.

This module provides comprehensive localization management including
language switching, translation loading, and persistence of language preferences.
"""

from __future__ import annotations

from typing import Dict, Optional, Any, List
from pathlib import Path
import json

from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal

from src.core.config import get_settings
from src.core.logging import get_logger

from .translator import Translator, HUNGARIAN_TRANSLATIONS, ENGLISH_TRANSLATIONS

logger = None  # Will be initialized after LoggerFactory setup


def _ensure_logger():
    """Ensure logger is initialized."""
    global logger
    if logger is None:
        logger = get_logger(__name__)


class LocalizationManager(QObject):
    """
    Comprehensive localization management system.

    Features:
    - Multiple language support (Hungarian, English)
    - Dynamic language switching
    - Translation persistence
    - Custom translation loading
    - Translation validation
    - Fallback language support
    """

    # Signals
    languageChanged = pyqtSignal(str)       # language_code
    translationsLoaded = pyqtSignal(str)    # language_code

    def __init__(self, initial_language: str = "hu"):
        """
        Initialize localization manager.

        Args:
            initial_language: Initial language code
        """
        super().__init__()

        # Configuration
        self.settings = get_settings()

        # Language management
        self._current_language = ""
        self._available_languages = {
            "en": "English",
            "hu": "Magyar"
        }
        self._fallback_language = "en"

        # Translator management
        self._current_translator: Optional[Translator] = None
        self._translators: Dict[str, Translator] = {}

        # Translation storage
        self._builtin_translations = {
            "en": ENGLISH_TRANSLATIONS,
            "hu": HUNGARIAN_TRANSLATIONS
        }

        # Preferences
        self._preferences_file = Path("data/localization_preferences.json")
        self._preferences: Dict[str, Any] = {}

        # Load preferences and create translations directory
        self._load_preferences()
        self._create_translation_files()

        # Set initial language
        self.set_language(initial_language)

        _ensure_logger()
        logger.info(f"Localization manager initialized with language: {initial_language}")

    def _load_preferences(self) -> None:
        """Load localization preferences."""
        try:
            if self._preferences_file.exists():
                with open(self._preferences_file, 'r', encoding='utf-8') as f:
                    self._preferences = json.load(f)
                logger.debug("Localization preferences loaded")
            else:
                self._preferences = self._get_default_preferences()
                logger.debug("Using default localization preferences")

        except Exception as e:
            logger.error(f"Error loading localization preferences: {e}")
            self._preferences = self._get_default_preferences()

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default localization preferences."""
        return {
            'last_language': 'hu',
            'auto_detect': False,
            'fallback_language': 'en',
            'custom_translations': {}
        }

    def _create_translation_files(self) -> None:
        """Create translation files from built-in translations."""
        try:
            # Create localization directory
            localization_dir = Path("localization")
            localization_dir.mkdir(exist_ok=True)

            # Create translation files for each language
            for lang_code, translations in self._builtin_translations.items():
                file_path = localization_dir / f"{lang_code}.json"

                # Only create if doesn't exist
                if not file_path.exists():
                    # Convert flat dictionary to nested structure
                    nested_translations = self._create_nested_structure(translations)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(nested_translations, f, ensure_ascii=False, indent=2)

                    logger.debug(f"Created translation file: {file_path}")

        except Exception as e:
            logger.error(f"Error creating translation files: {e}")

    def _create_nested_structure(self, flat_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Convert flat translation dictionary to nested structure.

        Args:
            flat_dict: Flat dictionary with dot-separated keys

        Returns:
            Nested dictionary structure
        """
        nested = {}

        for key, value in flat_dict.items():
            parts = key.split('.')
            current = nested

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return nested

    def save_preferences(self) -> None:
        """Save localization preferences to file."""
        try:
            # Ensure directory exists
            self._preferences_file.parent.mkdir(parents=True, exist_ok=True)

            # Update current language
            self._preferences['last_language'] = self._current_language

            # Save to file
            with open(self._preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self._preferences, f, ensure_ascii=False, indent=2)

            logger.debug("Localization preferences saved")

        except Exception as e:
            logger.error(f"Error saving localization preferences: {e}")

    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages."""
        return self._available_languages.copy()

    def get_current_language(self) -> str:
        """Get current language code."""
        return self._current_language

    def get_language_name(self, language_code: str = None) -> str:
        """
        Get human-readable language name.

        Args:
            language_code: Language code (uses current if None)

        Returns:
            Language name
        """
        if language_code is None:
            language_code = self._current_language

        return self._available_languages.get(language_code, language_code)

    def set_language(self, language_code: str) -> bool:
        """
        Set current language.

        Args:
            language_code: Language code to set

        Returns:
            bool: True if language was set successfully
        """
        if language_code not in self._available_languages:
            logger.error(f"Language not available: {language_code}")
            return False

        try:
            # Create translator for language
            translator = self._get_or_create_translator(language_code)

            if translator:
                old_language = self._current_language
                self._current_language = language_code
                self._current_translator = translator

                # Emit signal
                self.languageChanged.emit(language_code)

                logger.info(f"Language changed from '{old_language}' to '{language_code}'")
                return True
            else:
                logger.error(f"Failed to create translator for language: {language_code}")
                return False

        except Exception as e:
            logger.error(f"Error setting language {language_code}: {e}")
            return False

    def _get_or_create_translator(self, language_code: str) -> Optional[Translator]:
        """
        Get or create translator for language.

        Args:
            language_code: Language code

        Returns:
            Translator instance or None
        """
        if language_code not in self._translators:
            try:
                translator = Translator(language_code)
                self._translators[language_code] = translator
                self.translationsLoaded.emit(language_code)
                logger.debug(f"Created translator for language: {language_code}")
            except Exception as e:
                logger.error(f"Error creating translator for {language_code}: {e}")
                return None

        return self._translators[language_code]

    def get_translator(self) -> Optional[Translator]:
        """Get current translator instance."""
        return self._current_translator

    def translate(self, key: str, context: str = None) -> str:
        """
        Translate text using current translator.

        Args:
            key: Translation key
            context: Translation context

        Returns:
            Translated text
        """
        if self._current_translator:
            return self._current_translator.translate(context or "", key)
        return key

    def add_language(self, language_code: str, language_name: str) -> None:
        """
        Add new language support.

        Args:
            language_code: Language code (e.g., 'de')
            language_name: Language display name (e.g., 'Deutsch')
        """
        self._available_languages[language_code] = language_name
        logger.info(f"Added language support: {language_code} ({language_name})")

    def remove_language(self, language_code: str) -> bool:
        """
        Remove language support.

        Args:
            language_code: Language code to remove

        Returns:
            bool: True if removed successfully
        """
        if language_code == self._fallback_language:
            logger.error(f"Cannot remove fallback language: {language_code}")
            return False

        if language_code == self._current_language:
            # Switch to fallback before removing
            self.set_language(self._fallback_language)

        # Remove from available languages
        self._available_languages.pop(language_code, None)

        # Remove translator
        self._translators.pop(language_code, None)

        logger.info(f"Removed language support: {language_code}")
        return True

    def reload_translations(self, language_code: str = None) -> bool:
        """
        Reload translations for language.

        Args:
            language_code: Language to reload (current if None)

        Returns:
            bool: True if reloaded successfully
        """
        if language_code is None:
            language_code = self._current_language

        try:
            # Remove existing translator
            if language_code in self._translators:
                del self._translators[language_code]

            # Create new translator
            translator = self._get_or_create_translator(language_code)

            if translator and language_code == self._current_language:
                self._current_translator = translator

            logger.info(f"Reloaded translations for language: {language_code}")
            return True

        except Exception as e:
            logger.error(f"Error reloading translations for {language_code}: {e}")
            return False

    def get_translation_stats(self, language_code: str = None) -> Dict[str, Any]:
        """
        Get translation statistics.

        Args:
            language_code: Language code (current if None)

        Returns:
            Dictionary with translation statistics
        """
        if language_code is None:
            language_code = self._current_language

        translator = self._translators.get(language_code)
        if not translator:
            return {}

        return {
            'language_code': language_code,
            'language_name': self.get_language_name(language_code),
            'translation_count': translator.get_translation_count(),
            'is_current': language_code == self._current_language,
            'has_fallback': language_code != self._fallback_language
        }

    def validate_translations(self, language_code: str) -> List[str]:
        """
        Validate translations for missing keys.

        Args:
            language_code: Language code to validate

        Returns:
            List of missing translation keys
        """
        missing_keys = []

        try:
            # Get base language translations (fallback)
            base_translations = self._builtin_translations.get(self._fallback_language, {})
            target_translations = self._builtin_translations.get(language_code, {})

            # Find missing keys
            for key in base_translations.keys():
                if key not in target_translations:
                    missing_keys.append(key)

            logger.info(f"Translation validation for {language_code}: {len(missing_keys)} missing keys")

        except Exception as e:
            logger.error(f"Error validating translations for {language_code}: {e}")

        return missing_keys

    def export_translations(self, language_code: str, file_path: Path) -> bool:
        """
        Export translations to file.

        Args:
            language_code: Language to export
            file_path: Export file path

        Returns:
            bool: True if export successful
        """
        try:
            translations = self._builtin_translations.get(language_code, {})
            if not translations:
                logger.error(f"No translations found for language: {language_code}")
                return False

            # Create nested structure for export
            nested_translations = self._create_nested_structure(translations)

            # Add metadata
            export_data = {
                'metadata': {
                    'language_code': language_code,
                    'language_name': self.get_language_name(language_code),
                    'translation_count': len(translations),
                    'export_version': '1.0'
                },
                'translations': nested_translations
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported translations for {language_code} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting translations: {e}")
            return False

    def import_translations(self, file_path: Path) -> bool:
        """
        Import translations from file.

        Args:
            file_path: Import file path

        Returns:
            bool: True if import successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            translations = data.get('translations', {})

            language_code = metadata.get('language_code')
            language_name = metadata.get('language_name')

            if not language_code or not translations:
                logger.error("Invalid translation file format")
                return False

            # Flatten translations
            flat_translations = {}
            self._flatten_translations(translations, flat_translations)

            # Add to built-in translations
            self._builtin_translations[language_code] = flat_translations

            # Add language if not exists
            if language_code not in self._available_languages:
                self.add_language(language_code, language_name or language_code)

            # Reload translator if it exists
            if language_code in self._translators:
                self.reload_translations(language_code)

            logger.info(f"Imported {len(flat_translations)} translations for {language_code}")
            return True

        except Exception as e:
            logger.error(f"Error importing translations: {e}")
            return False

    def _flatten_translations(self, data: Dict[str, Any], result: Dict[str, str], prefix: str = "") -> None:
        """
        Flatten nested translations dictionary.

        Args:
            data: Nested translation data
            result: Result dictionary to populate
            prefix: Key prefix
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_translations(value, result, full_key)
            else:
                result[full_key] = str(value)

    def get_preferences(self) -> Dict[str, Any]:
        """Get localization preferences."""
        return self._preferences.copy()

    def set_preference(self, key: str, value: Any) -> None:
        """Set localization preference."""
        self._preferences[key] = value
        logger.debug(f"Localization preference set: {key} = {value}")

    def reset_to_default(self) -> None:
        """Reset localization manager to default state."""
        try:
            # Reset preferences
            self._preferences = self._get_default_preferences()

            # Set default language
            self.set_language('hu')

            # Clear translator cache
            self._translators.clear()

            logger.info("Localization manager reset to default state")

        except Exception as e:
            logger.error(f"Error resetting localization manager: {e}")

    def get_missing_translations(self) -> Dict[str, List[str]]:
        """Get missing translations for all languages."""
        missing = {}

        for lang_code in self._available_languages.keys():
            if lang_code != self._fallback_language:
                missing[lang_code] = self.validate_translations(lang_code)

        return missing


# Global localization helper functions

_localization_manager: Optional[LocalizationManager] = None


def get_localization_manager() -> LocalizationManager:
    """Get global localization manager instance."""
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
    return _localization_manager


def set_global_language(language_code: str) -> bool:
    """Set global application language."""
    manager = get_localization_manager()
    return manager.set_language(language_code)


def tr(text: str, context: str = None) -> str:
    """
    Translate text using global localization manager.

    Args:
        text: Text to translate
        context: Translation context

    Returns:
        Translated text
    """
    manager = get_localization_manager()
    return manager.translate(text, context)