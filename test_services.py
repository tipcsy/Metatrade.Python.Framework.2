#!/usr/bin/env python3
"""
Service Health Check Test Script
Tests all microservices health endpoints
"""

import requests
import json
from typing import Dict, List

# Service configuration
SERVICES = [
    {"name": "Backend API", "port": 5000},
    {"name": "Data Service", "port": 5001},
    {"name": "MT5 Service", "port": 5002},
    {"name": "Pattern Service", "port": 5003},
    {"name": "Strategy Service", "port": 5004},
    {"name": "AI Service", "port": 5005},
    {"name": "Backtesting Service", "port": 5006},
]

def test_service_health(name: str, port: int) -> Dict:
    """
    Test a service health endpoint

    Args:
        name: Service name
        port: Service port

    Returns:
        Test result dictionary
    """
    url = f"http://localhost:{port}/health"

    try:
        response = requests.get(url, timeout=2)

        if response.status_code == 200:
            data = response.json()
            return {
                "name": name,
                "port": port,
                "status": "✅ HEALTHY",
                "response": data
            }
        else:
            return {
                "name": name,
                "port": port,
                "status": f"⚠️  ERROR (HTTP {response.status_code})",
                "response": None
            }

    except requests.exceptions.ConnectionError:
        return {
            "name": name,
            "port": port,
            "status": "❌ OFFLINE (Connection refused)",
            "response": None
        }
    except requests.exceptions.Timeout:
        return {
            "name": name,
            "port": port,
            "status": "⏱️  TIMEOUT",
            "response": None
        }
    except Exception as e:
        return {
            "name": name,
            "port": port,
            "status": f"❌ ERROR: {str(e)}",
            "response": None
        }

def main():
    """Main test function"""
    print("=" * 60)
    print("MT5 Trading Platform 2.0 - Service Health Check")
    print("=" * 60)
    print()

    results: List[Dict] = []

    for service in SERVICES:
        print(f"Testing {service['name']} (port {service['port']})...", end=" ")
        result = test_service_health(service['name'], service['port'])
        results.append(result)
        print(result['status'])

        if result['response']:
            print(f"  Response: {json.dumps(result['response'], indent=2)}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    healthy_count = sum(1 for r in results if "HEALTHY" in r['status'])
    offline_count = sum(1 for r in results if "OFFLINE" in r['status'])
    error_count = sum(1 for r in results if "ERROR" in r['status'])

    print(f"✅ Healthy: {healthy_count}")
    print(f"❌ Offline: {offline_count}")
    print(f"⚠️  Errors: {error_count}")
    print(f"📊 Total: {len(SERVICES)}")
    print()

    if healthy_count == len(SERVICES):
        print("🎉 All services are healthy!")
    elif offline_count == len(SERVICES):
        print("⚠️  All services are offline. Did you start them?")
        print("\nTo start a service:")
        print("  cd services/<service-name>")
        print("  python main.py")
    else:
        print("⚠️  Some services are not healthy. Check the output above.")

    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
