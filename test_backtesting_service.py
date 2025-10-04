#!/usr/bin/env python3
"""
Backtesting Service Test Script

Tests:
1. Health check
2. List available strategies
3. Start a backtest
4. Check backtest status
5. Retrieve backtest results
"""

import requests
import time
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:5006"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_result(test_name, success, message=""):
    status = "✓ PASS" if success else "✗ FAIL"
    color = "\033[0;32m" if success else "\033[0;31m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} - {test_name}")
    if message:
        print(f"      {message}")

def test_health_check():
    """Test 1: Health check"""
    print_header("Test 1: Health Check")

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        success = response.status_code == 200

        if success:
            data = response.json()
            print_result(
                "Health check",
                True,
                f"Service: {data.get('service')}, Status: {data.get('status')}"
            )
            return True
        else:
            print_result("Health check", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("Health check", False, str(e))
        return False

def test_list_strategies():
    """Test 2: List available strategies"""
    print_header("Test 2: List Available Strategies")

    try:
        response = requests.get(f"{BASE_URL}/strategies", timeout=5)
        success = response.status_code == 200

        if success:
            data = response.json()
            strategies = data.get('data', {}).get('strategies', [])
            print_result(
                "List strategies",
                True,
                f"Found {len(strategies)} strategies"
            )

            for strategy in strategies:
                print(f"      - {strategy['name']} ({strategy['type']})")

            return True
        else:
            print_result("List strategies", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("List strategies", False, str(e))
        return False

def test_metric_definitions():
    """Test 3: Get metric definitions"""
    print_header("Test 3: Get Metric Definitions")

    try:
        response = requests.get(f"{BASE_URL}/metrics/definitions", timeout=5)
        success = response.status_code == 200

        if success:
            data = response.json()
            metrics = data.get('data', {}).get('metrics', {})
            print_result(
                "Get metrics",
                True,
                f"Found {len(metrics)} performance metrics"
            )

            # Show a few metrics
            for i, (key, value) in enumerate(list(metrics.items())[:5]):
                print(f"      - {value['name']}: {value['description']}")

            return True
        else:
            print_result("Get metrics", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("Get metrics", False, str(e))
        return False

def test_start_backtest():
    """Test 4: Start a backtest"""
    print_header("Test 4: Start Backtest")

    # Calculate timestamps (last 30 days)
    now = datetime.now()
    from_time = int((now - timedelta(days=30)).timestamp())
    to_time = int(now.timestamp())

    backtest_request = {
        "strategy_type": "MA_CROSSOVER",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "from_time": from_time,
        "to_time": to_time,
        "initial_balance": 10000.0,
        "parameters": {
            "fast_period": 10,
            "slow_period": 30,
            "stop_loss_pips": 50.0,
            "take_profit_pips": 100.0,
            "position_size": 0.01
        },
        "commission": 0.0,
        "spread_pips": 1.0,
        "slippage_pips": 0.5,
        "position_size": 0.01
    }

    try:
        response = requests.post(
            f"{BASE_URL}/backtest/start",
            json=backtest_request,
            timeout=10
        )
        success = response.status_code == 200

        if success:
            data = response.json()
            backtest_id = data.get('data', {}).get('backtest_id')
            print_result(
                "Start backtest",
                True,
                f"Backtest ID: {backtest_id}"
            )
            return backtest_id
        else:
            print_result("Start backtest", False, f"HTTP {response.status_code}")
            if response.text:
                print(f"      Response: {response.text[:200]}")
            return None
    except Exception as e:
        print_result("Start backtest", False, str(e))
        return None

def test_backtest_status(backtest_id):
    """Test 5: Check backtest status"""
    print_header("Test 5: Check Backtest Status")

    if not backtest_id:
        print_result("Check status", False, "No backtest ID provided")
        return False

    try:
        response = requests.get(
            f"{BASE_URL}/backtest/{backtest_id}/status",
            timeout=5
        )
        success = response.status_code == 200

        if success:
            data = response.json()
            status_info = data.get('data', {})
            print_result(
                "Check status",
                True,
                f"Status: {status_info.get('status')}, Progress: {status_info.get('progress', 0)}%"
            )
            return True
        else:
            print_result("Check status", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("Check status", False, str(e))
        return False

def test_backtest_result(backtest_id):
    """Test 6: Get backtest result"""
    print_header("Test 6: Get Backtest Result")

    if not backtest_id:
        print_result("Get result", False, "No backtest ID provided")
        return False

    # Wait a bit for backtest to complete
    print("      Waiting for backtest to complete...")
    time.sleep(3)

    try:
        response = requests.get(
            f"{BASE_URL}/backtest/{backtest_id}",
            timeout=5
        )
        success = response.status_code == 200

        if success:
            data = response.json()
            result = data.get('data', {})

            print_result(
                "Get result",
                True,
                f"Status: {result.get('status')}"
            )

            # Show performance if available
            performance = result.get('performance')
            if performance:
                print(f"\n      Performance Metrics:")
                print(f"      - Total Trades: {performance.get('total_trades', 0)}")
                print(f"      - Win Rate: {performance.get('win_rate', 0):.2f}%")
                print(f"      - Net Profit: ${performance.get('net_profit', 0):.2f}")
                print(f"      - Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
                print(f"      - Max Drawdown: {performance.get('max_drawdown', 0):.2f}%")

            return True
        else:
            print_result("Get result", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("Get result", False, str(e))
        return False

def test_list_backtests():
    """Test 7: List all backtests"""
    print_header("Test 7: List All Backtests")

    try:
        response = requests.get(
            f"{BASE_URL}/backtests?limit=10",
            timeout=5
        )
        success = response.status_code == 200

        if success:
            data = response.json()
            backtests = data.get('data', {}).get('backtests', [])
            print_result(
                "List backtests",
                True,
                f"Found {len(backtests)} backtests"
            )

            for bt in backtests[:3]:
                print(f"      - {bt.get('backtest_id')[:8]}... | {bt.get('strategy_type')} | {bt.get('status')}")

            return True
        else:
            print_result("List backtests", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("List backtests", False, str(e))
        return False

def main():
    print("\n🧪 Backtesting Service Test Suite")
    print("Starting tests at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    results = []

    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("List Strategies", test_list_strategies()))
    results.append(("Metric Definitions", test_metric_definitions()))

    backtest_id = test_start_backtest()
    results.append(("Start Backtest", backtest_id is not None))

    if backtest_id:
        results.append(("Check Status", test_backtest_status(backtest_id)))
        results.append(("Get Result", test_backtest_result(backtest_id)))

    results.append(("List Backtests", test_list_backtests()))

    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
