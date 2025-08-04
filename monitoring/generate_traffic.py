#!/usr/bin/env python3
"""
Generate continuous traffic to populate the monitoring dashboard
"""
import json
import os
import random
import sys
import time

import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "http://localhost:8000"


def generate_bid_data():
    """Generate random bid data"""
    return {
        "auction_id": f"auction_{random.randint(1000, 9999)}",
        "bidder_tendency": random.uniform(0, 1),
        "bidding_ratio": random.uniform(0, 1),
        "successive_outbidding": random.uniform(0, 1),
        "last_bidding": random.uniform(0, 1),
        "auction_bids": random.uniform(0, 1),
        "starting_price_average": random.uniform(0, 1),
        "early_bidding": random.uniform(0, 1),
        "winning_ratio": random.uniform(0, 1),
    }


def make_prediction():
    """Make a single prediction"""
    try:
        bid_data = generate_bid_data()
        response = requests.post(f"{API_URL}/predict", json=bid_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(
                f"✅ Prediction: {result['prediction']} (prob: {result['probability']:.3f}, drift: {result['drift_score']:.3f})"
            )
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def make_batch_prediction():
    """Make a batch prediction"""
    try:
        bid_data_list = [generate_bid_data() for _ in range(random.randint(2, 5))]
        response = requests.post(
            f"{API_URL}/predict_batch", json=bid_data_list, timeout=5
        )
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Batch: {len(results)} predictions")
            for i, result in enumerate(results):
                print(
                    f"   {i+1}: {result['prediction']} (prob: {result['probability']:.3f}, drift: {result['drift_score']:.3f})"
                )
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Batch request failed: {e}")
        return False


def main():
    """Generate continuous traffic"""
    print("🚀 Starting traffic generation for monitoring dashboard...")
    print("📊 Open Grafana at http://localhost:3000 (admin/admin) to see the dashboard")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)

    prediction_count = 0
    batch_count = 0

    try:
        while True:
            # Randomly choose between single and batch predictions
            if random.random() < 0.7:  # 70% single predictions
                if make_prediction():
                    prediction_count += 1
            else:
                if make_batch_prediction():
                    batch_count += 1

            # Print summary every 10 predictions
            total = prediction_count + batch_count
            if total % 10 == 0:
                print(
                    f"\n📈 Summary: {prediction_count} single predictions, {batch_count} batch predictions"
                )
                print("=" * 60)

            # Random delay between 1-3 seconds
            time.sleep(random.uniform(1, 3))

    except KeyboardInterrupt:
        print(f"\n🛑 Traffic generation stopped!")
        print(
            f"📊 Total: {prediction_count} single predictions, {batch_count} batch predictions"
        )
        print(
            "🌐 Check Grafana at http://localhost:3000 to see your monitoring dashboard!"
        )


if __name__ == "__main__":
    main()
