"""
Core service logic for Pattern Service
Technical indicators and pattern recognition
"""

import logging
from .indicator_engine import IndicatorEngine
from .pattern_detector import PatternDetector
from .pattern_scanner import PatternScanner

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Manages pattern recognition and technical indicators"""

    def __init__(self, data_service_url: str = "http://localhost:5001"):
        """
        Initialize Pattern Analyzer

        Args:
            data_service_url: URL of the data service
        """
        self.indicator_engine = IndicatorEngine()
        self.pattern_detector = PatternDetector()
        self.pattern_scanner = PatternScanner(data_service_url=data_service_url)
        logger.info("Pattern Analyzer initialized")

    def get_indicator_engine(self):
        """Get the indicator engine instance"""
        return self.indicator_engine

    def get_pattern_detector(self):
        """Get the pattern detector instance"""
        return self.pattern_detector

    def get_pattern_scanner(self):
        """Get the pattern scanner instance"""
        return self.pattern_scanner
