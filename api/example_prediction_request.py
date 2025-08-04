import requests

# API endpoint
url = "http://localhost:8000/predict"

# Test data to send to the API. Note that auction_id is a string, while the others
# are integers.
test_data = {
    "auction_id": "732",
    "bidder_tendency": 0.5,
    "bidding_ratio": 0.3,
    "successive_outbidding": 0.2,
    "last_bidding": 0.1,
    "auction_bids": 10.0,
    "starting_price_average": 100.0,
    "early_bidding": 0.4,
    "winning_ratio": 0.6,
}

# Send request
response = requests.post(url, json=test_data)

# Print results
print("Status Code:", response.status_code)
print("Response:", response.json())
