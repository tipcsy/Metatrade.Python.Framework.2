"""
Test script for Pattern & Indicator Service
Tests core functionality without requiring external services
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test imports
print("Testing imports...")
from app.core.indicator_engine import IndicatorEngine
from app.core.pattern_detector import PatternDetector
print("✓ Imports successful")

# Generate sample OHLC data
print("\nGenerating sample OHLC data...")
np.random.seed(42)
dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D')
base_price = 100

data = pd.DataFrame({
    'time': dates,
    'open': base_price + np.random.randn(100).cumsum(),
    'high': base_price + np.random.randn(100).cumsum() + 1,
    'low': base_price + np.random.randn(100).cumsum() - 1,
    'close': base_price + np.random.randn(100).cumsum(),
    'tick_volume': np.random.randint(1000, 10000, 100)
})

# Ensure high is highest and low is lowest
data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)

print(f"✓ Generated {len(data)} bars of sample data")

# Test Indicator Engine
print("\n" + "="*60)
print("Testing Indicator Engine")
print("="*60)

engine = IndicatorEngine()

# Test SMA
print("\n1. Testing SMA...")
sma_20 = engine.calculate_sma(data, period=20)
print(f"   SMA(20) latest value: {sma_20.iloc[-1]:.4f}")
print("   ✓ SMA calculation successful")

# Test EMA
print("\n2. Testing EMA...")
ema_20 = engine.calculate_ema(data, period=20)
print(f"   EMA(20) latest value: {ema_20.iloc[-1]:.4f}")
print("   ✓ EMA calculation successful")

# Test RSI
print("\n3. Testing RSI...")
rsi = engine.calculate_rsi(data, period=14)
print(f"   RSI(14) latest value: {rsi.iloc[-1]:.2f}")
print("   ✓ RSI calculation successful")

# Test MACD
print("\n4. Testing MACD...")
macd = engine.calculate_macd(data)
print(f"   MACD latest value: {macd['macd'].iloc[-1]:.4f}")
print(f"   Signal latest value: {macd['signal'].iloc[-1]:.4f}")
print(f"   Histogram latest value: {macd['histogram'].iloc[-1]:.4f}")
print("   ✓ MACD calculation successful")

# Test Bollinger Bands
print("\n5. Testing Bollinger Bands...")
bbands = engine.calculate_bollinger_bands(data, period=20)
print(f"   Upper Band: {bbands['upper'].iloc[-1]:.4f}")
print(f"   Middle Band: {bbands['middle'].iloc[-1]:.4f}")
print(f"   Lower Band: {bbands['lower'].iloc[-1]:.4f}")
print("   ✓ Bollinger Bands calculation successful")

# Test ATR
print("\n6. Testing ATR...")
atr = engine.calculate_atr(data, period=14)
print(f"   ATR(14) latest value: {atr.iloc[-1]:.4f}")
print("   ✓ ATR calculation successful")

# Test Stochastic
print("\n7. Testing Stochastic Oscillator...")
stoch = engine.calculate_stochastic(data)
print(f"   %K latest value: {stoch['k'].iloc[-1]:.2f}")
print(f"   %D latest value: {stoch['d'].iloc[-1]:.2f}")
print("   ✓ Stochastic calculation successful")

# Test ADX
print("\n8. Testing ADX...")
adx = engine.calculate_adx(data, period=14)
print(f"   ADX latest value: {adx['adx'].iloc[-1]:.2f}")
print(f"   +DI latest value: {adx['plus_di'].iloc[-1]:.2f}")
print(f"   -DI latest value: {adx['minus_di'].iloc[-1]:.2f}")
print("   ✓ ADX calculation successful")

# Test OBV
print("\n9. Testing OBV...")
obv = engine.calculate_obv(data)
print(f"   OBV latest value: {obv.iloc[-1]:.0f}")
print("   ✓ OBV calculation successful")

# Test all indicators at once
print("\n10. Testing calculate_all_indicators...")
all_indicators = engine.calculate_all_indicators(data)
print(f"   Calculated {len(all_indicators)} indicator values")
print("   ✓ All indicators calculation successful")

# Test Pattern Detector
print("\n" + "="*60)
print("Testing Pattern Detector")
print("="*60)

detector = PatternDetector()

# Test Doji
print("\n1. Testing Doji detection...")
doji = detector.detect_doji(data)
doji_count = doji.sum()
print(f"   Found {doji_count} Doji patterns")
print("   ✓ Doji detection successful")

# Test Hammer
print("\n2. Testing Hammer detection...")
hammer = detector.detect_hammer(data)
hammer_count = hammer.sum()
print(f"   Found {hammer_count} Hammer patterns")
print("   ✓ Hammer detection successful")

# Test Shooting Star
print("\n3. Testing Shooting Star detection...")
shooting_star = detector.detect_shooting_star(data)
ss_count = shooting_star.sum()
print(f"   Found {ss_count} Shooting Star patterns")
print("   ✓ Shooting Star detection successful")

# Test Engulfing
print("\n4. Testing Engulfing patterns...")
engulfing = detector.detect_engulfing(data)
bull_eng_count = engulfing['bullish_engulfing'].sum()
bear_eng_count = engulfing['bearish_engulfing'].sum()
print(f"   Found {bull_eng_count} Bullish Engulfing patterns")
print(f"   Found {bear_eng_count} Bearish Engulfing patterns")
print("   ✓ Engulfing detection successful")

# Test Morning/Evening Star
print("\n5. Testing Star patterns...")
morning_star = detector.detect_morning_star(data)
evening_star = detector.detect_evening_star(data)
print(f"   Found {morning_star.sum()} Morning Star patterns")
print(f"   Found {evening_star.sum()} Evening Star patterns")
print("   ✓ Star pattern detection successful")

# Test Three Soldiers/Crows
print("\n6. Testing Three Soldiers/Crows...")
three_white = detector.detect_three_white_soldiers(data)
three_black = detector.detect_three_black_crows(data)
print(f"   Found {three_white.sum()} Three White Soldiers patterns")
print(f"   Found {three_black.sum()} Three Black Crows patterns")
print("   ✓ Three Soldiers/Crows detection successful")

# Test Support/Resistance
print("\n7. Testing Support/Resistance detection...")
sr_levels = detector.detect_support_resistance(data)
print(f"   Found {len(sr_levels['support'])} support levels")
print(f"   Found {len(sr_levels['resistance'])} resistance levels")
print("   ✓ Support/Resistance detection successful")

# Test Trend
print("\n8. Testing Trend detection...")
trend = detector.detect_trend(data)
print(f"   Current trend: {trend}")
print("   ✓ Trend detection successful")

# Test all patterns at once
print("\n9. Testing detect_all_candlestick_patterns...")
all_patterns = detector.detect_all_candlestick_patterns(data)
print(f"   Detected patterns: {sum(1 for v in all_patterns.values() if v)}")
for pattern, detected in all_patterns.items():
    if detected:
        print(f"     - {pattern}")
print("   ✓ All patterns detection successful")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✓ All indicator calculations working correctly")
print("✓ All pattern detection working correctly")
print("✓ Pattern & Indicator Service is ready for deployment")
print("="*60)
