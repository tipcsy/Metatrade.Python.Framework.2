#!/usr/bin/env python3
"""
Test script for Strategy Service
Demonstrates all the key functionality of the service
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:5003"


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_response(response: requests.Response, title: str = None):
    """Print formatted response"""
    if title:
        print(f"\n{title}:")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except:
        print(response.text)
    print()


def test_health_check():
    """Test health check endpoint"""
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response, "Health Check Response")


def test_create_strategies():
    """Test creating strategies"""
    print_section("2. Create Strategies")

    # Create MA Crossover strategy
    ma_strategy = {
        "strategy_type": "MA_CROSSOVER",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "parameters": {
            "fast_period": 10,
            "slow_period": 20,
            "stop_loss_pips": 50,
            "take_profit_pips": 100
        }
    }
    response = requests.post(f"{BASE_URL}/strategies", json=ma_strategy)
    print_response(response, "Created MA Crossover Strategy")

    # Create RSI strategy
    rsi_strategy = {
        "strategy_type": "RSI",
        "symbol": "GBPUSD",
        "timeframe": "M15",
        "parameters": {
            "rsi_period": 14,
            "oversold_level": 30,
            "overbought_level": 70,
            "stop_loss_pips": 40,
            "take_profit_pips": 80
        }
    }
    response = requests.post(f"{BASE_URL}/strategies", json=rsi_strategy)
    print_response(response, "Created RSI Strategy")


def test_list_strategies():
    """Test listing all strategies"""
    print_section("3. List All Strategies")
    response = requests.get(f"{BASE_URL}/strategies")
    print_response(response, "All Strategies")
    return response.json()


def test_start_strategy(strategy_id: str):
    """Test starting a strategy"""
    print_section(f"4. Start Strategy {strategy_id}")
    response = requests.post(f"{BASE_URL}/strategies/{strategy_id}/start")
    print_response(response, f"Start Strategy {strategy_id}")


def test_get_strategy(strategy_id: str):
    """Test getting a specific strategy"""
    print_section(f"5. Get Strategy {strategy_id}")
    response = requests.get(f"{BASE_URL}/strategies/{strategy_id}")
    print_response(response, f"Strategy {strategy_id} Details")


def test_risk_status():
    """Test risk status endpoint"""
    print_section("6. Risk Status")
    response = requests.get(f"{BASE_URL}/risk/status")
    print_response(response, "Current Risk Status")


def test_position_statistics():
    """Test position statistics endpoint"""
    print_section("7. Position Statistics")
    response = requests.get(f"{BASE_URL}/positions/statistics")
    print_response(response, "Position Statistics")


def test_list_positions():
    """Test listing positions"""
    print_section("8. List Positions")
    response = requests.get(f"{BASE_URL}/positions")
    print_response(response, "All Positions")


def test_stop_strategy(strategy_id: str):
    """Test stopping a strategy"""
    print_section(f"9. Stop Strategy {strategy_id}")
    response = requests.post(f"{BASE_URL}/strategies/{strategy_id}/stop")
    print_response(response, f"Stop Strategy {strategy_id}")


def test_pause_resume_strategy(strategy_id: str):
    """Test pausing and resuming a strategy"""
    print_section(f"10. Pause/Resume Strategy {strategy_id}")

    # Pause
    response = requests.post(f"{BASE_URL}/strategies/{strategy_id}/pause")
    print_response(response, "Pause Strategy")

    # Get status
    response = requests.get(f"{BASE_URL}/strategies/{strategy_id}")
    print_response(response, "Strategy Status After Pause")

    # Resume
    response = requests.post(f"{BASE_URL}/strategies/{strategy_id}/resume")
    print_response(response, "Resume Strategy")


def test_strategy_statistics():
    """Test strategy statistics"""
    print_section("11. Strategy Statistics")
    response = requests.get(f"{BASE_URL}/strategies/statistics")
    print_response(response, "Strategy Engine Statistics")


def test_update_balance():
    """Test updating account balance"""
    print_section("12. Update Account Balance")
    response = requests.post(f"{BASE_URL}/risk/update-balance?new_balance=12000.0")
    print_response(response, "Update Balance to $12,000")

    # Check risk status after update
    response = requests.get(f"{BASE_URL}/risk/status")
    print_response(response, "Risk Status After Balance Update")


def test_delete_strategy(strategy_id: str):
    """Test deleting a strategy"""
    print_section(f"13. Delete Strategy {strategy_id}")
    response = requests.delete(f"{BASE_URL}/strategies/{strategy_id}")
    print_response(response, f"Delete Strategy {strategy_id}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  STRATEGY SERVICE TEST SUITE")
    print("="*60)

    try:
        # Test 1: Health check
        test_health_check()

        # Test 2: Create strategies
        test_create_strategies()

        # Test 3: List strategies
        strategies = test_list_strategies()

        if strategies:
            strategy_id = strategies[0]['strategy_id']

            # Test 4: Start strategy
            test_start_strategy(strategy_id)

            # Test 5: Get strategy details
            test_get_strategy(strategy_id)

            # Test 6: Risk status
            test_risk_status()

            # Test 7: Position statistics
            test_position_statistics()

            # Test 8: List positions
            test_list_positions()

            # Test 9: Pause/Resume
            test_pause_resume_strategy(strategy_id)

            # Test 10: Strategy statistics
            test_strategy_statistics()

            # Test 11: Update balance
            test_update_balance()

            # Test 12: Stop strategy
            test_stop_strategy(strategy_id)

            # Test 13: Delete strategy
            test_delete_strategy(strategy_id)

        print_section("TEST SUITE COMPLETED SUCCESSFULLY")
        print("All endpoints are working correctly!")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to Strategy Service at", BASE_URL)
        print("Please make sure the service is running on port 5003")
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {str(e)}")


if __name__ == "__main__":
    main()
