#!/usr/bin/env python3
"""
Test script for the monitoring setup
"""
import json
import os
import random
import sys
import time

import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API endpoint
API_URL = "http://localhost:8000"


def test_single_prediction():
    """Test a single prediction with monitoring"""
    print("Testing single prediction...")

    # Sample bid data
    bid_data = {
        "auction_id": "test_auction_001",
        "bidder_tendency": random.uniform(0, 1),
        "bidding_ratio": random.uniform(0, 1),
        "successive_outbidding": random.uniform(0, 1),
        "last_bidding": random.uniform(0, 1),
        "auction_bids": random.uniform(0, 1),
        "starting_price_average": random.uniform(0, 1),
        "early_bidding": random.uniform(0, 1),
        "winning_ratio": random.uniform(0, 1),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=bid_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Probability: {result['probability']:.3f}")
            print(f"   Drift Score: {result['drift_score']:.3f}")
            print(f"   Performance Score: {result['performance_score']:.3f}")
            print(f"   Is Shill Bid: {result['is_shill_bid']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction with monitoring"""
    print("\nTesting batch prediction...")

    # Generate multiple bid data
    bid_data_list = []
    for i in range(3):
        bid_data = {
            "auction_id": f"test_auction_{i:03d}",
            "bidder_tendency": random.uniform(0, 1),
            "bidding_ratio": random.uniform(0, 1),
            "successive_outbidding": random.uniform(0, 1),
            "last_bidding": random.uniform(0, 1),
            "auction_bids": random.uniform(0, 1),
            "starting_price_average": random.uniform(0, 1),
            "early_bidding": random.uniform(0, 1),
            "winning_ratio": random.uniform(0, 1),
        }
        bid_data_list.append(bid_data)

    try:
        response = requests.post(f"{API_URL}/predict_batch", json=bid_data_list)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Batch prediction successful! {len(results)} predictions")
            for i, result in enumerate(results):
                print(
                    f"   Prediction {i+1}: {result['prediction']} (prob: {result['probability']:.3f}, drift: {result['drift_score']:.3f}, perf: {result['performance_score']:.3f})"
                )
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_metrics():
    """Test Prometheus metrics endpoint"""
    print("\nTesting metrics endpoint...")

    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics = response.text
            print("✅ Metrics endpoint working!")
            print("   Available metrics:")
            for line in metrics.split("\n"):
                if line.startswith("# HELP") or line.startswith("# TYPE"):
                    continue
                if "evidently_" in line and not line.startswith("#"):
                    print(f"   {line.split()[0]}")
            return True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_health():
    """Test health endpoint"""
    print("\nTesting health endpoint...")

    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {health['status']}")
            print(f"   Model loaded: {health['model_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting monitoring tests...")
    print("=" * 50)

    tests = [test_health, test_single_prediction, test_batch_prediction, test_metrics]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Monitoring setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the setup.")


if __name__ == "__main__":
    main()
