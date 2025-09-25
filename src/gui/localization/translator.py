"""
Translation system for the MetaTrader Python Framework.

This module provides PyQt6 translator functionality with support for
Hungarian and English languages, including dynamic translation updates.
"""

from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path
import json

from PyQt6.QtCore import QTranslator, QCoreApplication

from src.core.logging import get_logger

logger = None  # Will be initialized after LoggerFactory setup


def _ensure_logger():
    """Ensure logger is initialized."""
    global logger
    if logger is None:
        logger = get_logger(__name__)


class Translator(QTranslator):
    """
    Custom translator class for the MetaTrader Framework.

    Provides translation functionality with JSON-based translation files
    and support for dynamic language switching.
    """

    def __init__(self, language_code: str = "en"):
        """
        Initialize translator.

        Args:
            language_code: Language code (e.g., 'en', 'hu')
        """
        super().__init__()

        self.language_code = language_code
        self._translations: Dict[str, str] = {}
        self._fallback_translations: Dict[str, str] = {}

        # Load translations
        self._load_translations()

    def _load_translations(self) -> None:
        """Load translations from JSON files."""
        try:
            # Load primary language translations
            primary_file = Path(f"localization/{self.language_code}.json")
            self._translations = self._load_translation_file(primary_file)

            # Load fallback (English) if primary is not English
            if self.language_code != "en":
                fallback_file = Path("localization/en.json")
                self._fallback_translations = self._load_translation_file(fallback_file)

            _ensure_logger()
            logger.info(f"Loaded {len(self._translations)} translations for '{self.language_code}'")

        except Exception as e:
            logger.error(f"Error loading translations for '{self.language_code}': {e}")

    def _load_translation_file(self, file_path: Path) -> Dict[str, str]:
        """
        Load translations from a JSON file.

        Args:
            file_path: Path to translation file

        Returns:
            Dictionary of translations
        """
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Flatten nested dictionaries
                    return self._flatten_translations(data)
            else:
                logger.warning(f"Translation file not found: {file_path}")
                return {}

        except Exception as e:
            logger.error(f"Error loading translation file {file_path}: {e}")
            return {}

    def _flatten_translations(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """
        Flatten nested translation dictionaries.

        Args:
            data: Translation data
            prefix: Key prefix for nested items

        Returns:
            Flattened dictionary
        """
        result = {}

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten_translations(value, full_key))
            else:
                result[full_key] = str(value)

        return result

    def translate(self, context: str, source_text: str, disambiguation: str = None, n: int = -1) -> str:
        """
        Translate text using loaded translations.

        Args:
            context: Translation context
            source_text: Source text to translate
            disambiguation: Disambiguation string
            n: Number for pluralization

        Returns:
            Translated text
        """
        # Create translation key
        if context:
            key = f"{context}.{source_text}"
        else:
            key = source_text

        # Try primary translations
        if key in self._translations:
            return self._translations[key]

        # Try context-less key
        if source_text in self._translations:
            return self._translations[source_text]

        # Try fallback translations
        if key in self._fallback_translations:
            return self._fallback_translations[key]

        if source_text in self._fallback_translations:
            return self._fallback_translations[source_text]

        # Return original text if no translation found
        logger.debug(f"No translation found for: {key}")
        return source_text

    def isEmpty(self) -> bool:
        """Check if translator is empty."""
        return len(self._translations) == 0 and len(self._fallback_translations) == 0

    def get_translation_count(self) -> int:
        """Get number of loaded translations."""
        return len(self._translations)

    def has_translation(self, key: str) -> bool:
        """Check if translation exists for key."""
        return key in self._translations or key in self._fallback_translations

    def get_language_code(self) -> str:
        """Get current language code."""
        return self.language_code


# Translation helper functions

def tr(source_text: str, context: str = None) -> str:
    """
    Translate text using the current application translator.

    Args:
        source_text: Text to translate
        context: Translation context

    Returns:
        Translated text
    """
    app = QCoreApplication.instance()
    if app:
        return app.translate(context or "", source_text)
    return source_text


def tr_n(source_text: str, n: int, context: str = None) -> str:
    """
    Translate text with pluralization.

    Args:
        source_text: Text to translate
        n: Number for pluralization
        context: Translation context

    Returns:
        Translated text
    """
    app = QCoreApplication.instance()
    if app:
        return app.translate(context or "", source_text, "", n)
    return source_text


# Predefined translations for Hungarian language
HUNGARIAN_TRANSLATIONS = {
    # Application general
    "Application": "Alkalmazás",
    "File": "Fájl",
    "Edit": "Szerkesztés",
    "View": "Nézet",
    "Tools": "Eszközök",
    "Help": "Súgó",
    "Exit": "Kilépés",
    "About": "Névjegy",
    "Settings": "Beállítások",
    "Preferences": "Preferenciák",

    # Window and UI
    "Window": "Ablak",
    "Close": "Bezárás",
    "Minimize": "Kis méret",
    "Maximize": "Nagy méret",
    "Restore": "Visszaállítás",
    "OK": "OK",
    "Cancel": "Mégse",
    "Apply": "Alkalmaz",
    "Yes": "Igen",
    "No": "Nem",
    "Save": "Mentés",
    "Load": "Betöltés",
    "Delete": "Törlés",
    "Remove": "Eltávolítás",
    "Add": "Hozzáadás",
    "Edit": "Szerkesztés",
    "Refresh": "Frissítés",
    "Update": "Frissítés",

    # Themes
    "Theme": "Téma",
    "Dark": "Sötét",
    "Light": "Világos",
    "Language": "Nyelv",
    "English": "Angol",
    "Magyar": "Magyar",

    # Symbol management
    "Symbol": "Szimbólum",
    "Symbols": "Szimbólumok",
    "Symbol Management": "Szimbólum kezelés",
    "Symbol Selection": "Szimbólum választás",
    "Active Symbols": "Aktív szimbólumok",
    "Add Symbol": "Szimbólum hozzáadása",
    "Remove Symbol": "Szimbólum eltávolítása",
    "Remove Selected": "Kijelölt eltávolítása",
    "Search": "Keresés",
    "Enter symbol or search term...": "Szimbólum vagy keresési kifejezés...",
    "Symbol Info": "Szimbólum információ",
    "Up": "Fel",
    "Down": "Le",
    "Move selected symbol up": "Kijelölt szimbólum felfelé",
    "Move selected symbol down": "Kijelölt szimbólum lefelé",
    "Total": "Összesen",
    "Active": "Aktív",
    "Priority": "Prioritás",
    "Description": "Leírás",
    "Type": "Típus",
    "Status": "Státusz",
    "Tradeable": "Kereskedhető",

    # Market data
    "Market Data": "Piaci adatok",
    "Bid": "Vétel",
    "Ask": "Eladás",
    "Spread": "Spread",
    "Spread (Pips)": "Spread (pip)",
    "Change %": "Változás %",
    "Volume": "Volumen",
    "Positions": "Pozíciók",
    "Last Update": "Utolsó frissítés",
    "Auto-resize columns": "Oszlopok automatikus méretezése",
    "Show milliseconds": "Ezredmásodpercek mutatása",
    "Decimal places": "Tizedesjegyek",
    "Highlight changes": "Változások kiemelése",
    "Refresh Data": "Adatok frissítése",

    # Trends
    "Trend": "Trend",
    "Trend M1": "Trend M1",
    "Trend M5": "Trend M5",
    "Trend M15": "Trend M15",
    "Trend H1": "Trend H1",
    "Trend H4": "Trend H4",
    "Trend D1": "Trend D1",

    # Connection and status
    "Connected": "Kapcsolódva",
    "Disconnected": "Lekapcsolva",
    "Connection": "Kapcsolat",
    "MT5": "MT5",
    "Status": "Státusz",
    "Ready": "Kész",
    "Loading": "Betöltés",
    "Error": "Hiba",
    "Warning": "Figyelmeztetés",
    "Information": "Információ",

    # Performance
    "Performance": "Teljesítmény",
    "Memory": "Memória",
    "Updates/sec": "Frissítés/mp",
    "Rate": "Sebesség",
    "Good": "Jó",
    "Normal": "Normál",
    "High": "Magas",
    "Low": "Alacsony",

    # Trading
    "Buy": "Vásárlás",
    "Sell": "Eladás",
    "Long": "Long",
    "Short": "Short",
    "Open": "Nyitás",
    "Close": "Zárás",
    "Profit": "Profit",
    "Loss": "Veszteség",
    "P&L": "P&L",

    # Time periods
    "Minute": "Perc",
    "Hour": "Óra",
    "Day": "Nap",
    "Week": "Hét",
    "Month": "Hónap",

    # Messages
    "Please select or enter a symbol": "Kérjük, válasszon vagy írjon be egy szimbólumot",
    "Failed to add symbol": "Nem sikerült hozzáadni a szimbólumot",
    "Symbol added successfully": "Szimbólum sikeresen hozzáadva",
    "Symbol removed successfully": "Szimbólum sikeresen eltávolítva",
    "No symbols selected": "Nincs kijelölt szimbólum",
    "Are you sure?": "Biztos benne?",
    "This action cannot be undone": "Ez a művelet nem vonható vissza",
    "Operation completed successfully": "Művelet sikeresen befejezve",
    "Operation failed": "Művelet sikertelen",
    "Please wait...": "Kérjük várjon...",
    "Processing...": "Feldolgozás...",
    "Done": "Kész",

    # Context menu
    "Symbol Details": "Szimbólum részletei",
    "Open Chart": "Grafikon megnyitása",
    "Copy": "Másolás",
    "Paste": "Beillesztés",
    "Select All": "Mind kijelölése",
    "Clear": "Törlés",

    # Errors
    "Connection Error": "Kapcsolódási hiba",
    "Data Error": "Adat hiba",
    "System Error": "Rendszer hiba",
    "Network Error": "Hálózati hiba",
    "Timeout Error": "Időtúllépési hiba",
    "Authentication Error": "Hitelesítési hiba",
    "Permission Error": "Jogosultsági hiba",
    "Not Found": "Nem található",
    "Invalid Data": "Érvénytelen adat",
    "Operation Cancelled": "Művelet megszakítva",

    # Tooltips and help
    "Connected to MetaTrader 5": "Kapcsolódva a MetaTrader 5-höz",
    "Disconnected from MetaTrader 5": "Lekapcsolódva a MetaTrader 5-ről",
    "Refresh available symbols from MT5": "Elérhető szimbólumok frissítése MT5-ből",
    "Active symbols / Total symbols": "Aktív szimbólumok / Összes szimbólum",
    "Data updates per second": "Adatfrissítések másodpercenként",
    "Memory usage": "Memóriahasználat",
    "Show detailed symbol information": "Részletes szimbólum információk mutatása",

    # About dialog
    "About MetaTrader Framework": "A MetaTrader Framework-ről",
    "Version": "Verzió",
    "A comprehensive framework for MetaTrader 5 integration": "Átfogó keretrendszer a MetaTrader 5 integrációhoz",
    "Built with PyQt6 and modern Python technologies": "PyQt6-tal és modern Python technológiákkal készült"
}

ENGLISH_TRANSLATIONS = {
    # This serves as the base/fallback language
    # Keys are the same as values for English
    key: key for key in HUNGARIAN_TRANSLATIONS.keys()
}