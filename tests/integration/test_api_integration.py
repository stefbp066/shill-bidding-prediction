import time

import pytest
from fastapi.testclient import TestClient

from api.main_simple import app


class TestAPIEndpoints:
    """Integration tests for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)

    @pytest.fixture
    def sample_bid_data(self):
        """Sample bid data for testing"""
        return {
            "auction_id": "test_auction_123",
            "bidder_tendency": 0.5,
            "bidding_ratio": 0.6,
            "successive_outbidding": 0.4,
            "last_bidding": 0.3,
            "auction_bids": 0.7,
            "starting_price_average": 0.8,
            "early_bidding": 0.2,
            "winning_ratio": 0.9,
        }

    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["model_loaded"] is True

    def test_predict_endpoint_success(self, client, sample_bid_data):
        """Test successful prediction request"""
        response = client.post("/predict", json=sample_bid_data)

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "prediction" in data
        assert "probability" in data
        assert "is_shill_bid" in data
        assert "drift_score" in data
        assert "performance_score" in data

        # Check data types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], float)
        assert isinstance(data["is_shill_bid"], bool)
        assert isinstance(data["drift_score"], float)
        assert isinstance(data["performance_score"], float)

        # Check value ranges
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert 0.0 <= data["drift_score"] <= 1.0
        assert 0.0 <= data["performance_score"] <= 1.0

    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction with invalid data"""
        invalid_data = {
            "auction_id": "test",
            "bidder_tendency": 1.5,  # Invalid: > 1
            "bidding_ratio": 0.6,
            "successive_outbidding": 0.4,
            "last_bidding": 0.3,
            "auction_bids": 0.7,
            "starting_price_average": 0.8,
            "early_bidding": 0.2,
            "winning_ratio": 0.9,
        }

        response = client.post("/predict", json=invalid_data)

        # Should return 422 (Unprocessable Entity) for validation errors
        assert response.status_code == 422

    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "auction_id": "test",
            "bidder_tendency": 0.5,
            # Missing other required fields
        }

        response = client.post("/predict", json=incomplete_data)

        assert response.status_code == 422

    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint"""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        # Check that metrics are returned
        metrics_text = response.text
        assert "evidently_prediction_count_total" in metrics_text
        assert "evidently_prediction_duration_seconds" in metrics_text
        assert "evidently_data_drift_score" in metrics_text
        assert "evidently_model_performance_score" in metrics_text

    def test_multiple_predictions(self, client, sample_bid_data):
        """Test multiple predictions to ensure consistency"""
        predictions = []

        for i in range(5):
            response = client.post("/predict", json=sample_bid_data)
            assert response.status_code == 200
            predictions.append(response.json())

        # Check that all predictions have the same structure
        for pred in predictions:
            assert "prediction" in pred
            assert "probability" in pred
            assert "is_shill_bid" in pred
            assert "drift_score" in pred
            assert "performance_score" in pred


class TestAPIPerformance:
    """Integration tests for API performance"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_prediction_response_time(self, client):
        """Test that predictions return within reasonable time"""
        sample_data = {
            "auction_id": "perf_test",
            "bidder_tendency": 0.5,
            "bidding_ratio": 0.6,
            "successive_outbidding": 0.4,
            "last_bidding": 0.3,
            "auction_bids": 0.7,
            "starting_price_average": 0.8,
            "early_bidding": 0.2,
            "winning_ratio": 0.9,
        }

        start_time = time.time()
        response = client.post("/predict", json=sample_data)
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should complete within 1 second

    def test_concurrent_predictions(self, client):
        """Test handling of concurrent prediction requests"""
        import queue
        import threading

        sample_data = {
            "auction_id": "concurrent_test",
            "bidder_tendency": 0.5,
            "bidding_ratio": 0.6,
            "successive_outbidding": 0.4,
            "last_bidding": 0.3,
            "auction_bids": 0.7,
            "starting_price_average": 0.8,
            "early_bidding": 0.2,
            "winning_ratio": 0.9,
        }

        results = queue.Queue()

        def make_prediction():
            response = client.post("/predict", json=sample_data)
            results.put(response.status_code)

        # Start 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all requests succeeded
        while not results.empty():
            status_code = results.get()
            assert status_code == 200


class TestAPIErrorHandling:
    """Integration tests for API error handling"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post("/predict", data="invalid json")

        assert response.status_code == 422

    def test_empty_request_body(self, client):
        """Test handling of empty request body"""
        response = client.post("/predict", json={})

        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client):
        """Test handling of nonexistent endpoints"""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test handling of unsupported HTTP methods"""
        response = client.put("/predict")

        assert response.status_code == 405


class TestAPIMetrics:
    """Integration tests for API metrics"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_metrics_after_predictions(self, client):
        """Test that metrics are updated after predictions"""
        # Get initial metrics
        initial_metrics = client.get("/metrics").text

        # Make a prediction
        sample_data = {
            "auction_id": "metrics_test",
            "bidder_tendency": 0.5,
            "bidding_ratio": 0.6,
            "successive_outbidding": 0.4,
            "last_bidding": 0.3,
            "auction_bids": 0.7,
            "starting_price_average": 0.8,
            "early_bidding": 0.2,
            "winning_ratio": 0.9,
        }

        response = client.post("/predict", json=sample_data)
        assert response.status_code == 200

        # Get metrics after prediction
        updated_metrics = client.get("/metrics").text

        # Check that prediction counter increased
        initial_count = initial_metrics.count("evidently_prediction_count_total")
        updated_count = updated_metrics.count("evidently_prediction_count_total")

        # The metrics should be different after a prediction
        assert (
            initial_count != updated_count
            or "evidently_prediction_count_total 1" in updated_metrics
        )


if __name__ == "__main__":
    pytest.main([__file__])
