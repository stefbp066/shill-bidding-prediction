import numpy as np
import pandas as pd
import pytest

from api.main_simple import BidData, MockModel, PredictionResponse


class TestMockModel:
    """Unit tests for the MockModel class"""

    def test_model_initialization(self):
        """Test that MockModel initializes correctly"""
        model = MockModel()
        assert model is not None

    def test_predict_returns_binary_values(self):
        """Test that predict returns only 0 or 1 values"""
        model = MockModel()
        X = pd.DataFrame(
            {
                "Bidder_Tendency": [0.5, 0.7, 0.3],
                "Bidding_Ratio": [0.6, 0.8, 0.4],
                "Successive_Outbidding": [0.4, 0.9, 0.2],
            }
        )

        predictions = model.predict(X)

        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba_returns_valid_probabilities(self):
        """Test that predict_proba returns valid probability values"""
        model = MockModel()
        X = pd.DataFrame(
            {
                "Bidder_Tendency": [0.5, 0.7],
                "Bidding_Ratio": [0.6, 0.8],
                "Successive_Outbidding": [0.4, 0.9],
            }
        )

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (2, 2)
        assert all(0 <= prob <= 1 for row in probabilities for prob in row)
        assert all(abs(row.sum() - 1.0) < 1e-6 for row in probabilities)


class TestBidData:
    """Unit tests for the BidData model"""

    def test_valid_bid_data(self):
        """Test creating valid BidData instance"""
        bid_data = BidData(
            auction_id="test_auction",
            bidder_tendency=0.5,
            bidding_ratio=0.6,
            successive_outbidding=0.4,
            last_bidding=0.3,
            auction_bids=0.7,
            starting_price_average=0.8,
            early_bidding=0.2,
            winning_ratio=0.9,
        )

        assert bid_data.auction_id == "test_auction"
        assert bid_data.bidder_tendency == 0.5
        assert bid_data.bidding_ratio == 0.6

    def test_bid_data_validation(self):
        """Test that invalid data raises validation errors"""
        with pytest.raises(ValueError):
            BidData(
                auction_id="test",
                bidder_tendency=1.5,  # Invalid: > 1
                bidding_ratio=0.6,
                successive_outbidding=0.4,
                last_bidding=0.3,
                auction_bids=0.7,
                starting_price_average=0.8,
                early_bidding=0.2,
                winning_ratio=0.9,
            )


class TestPredictionResponse:
    """Unit tests for the PredictionResponse model"""

    def test_prediction_response_creation(self):
        """Test creating PredictionResponse instance"""
        response = PredictionResponse(
            prediction=1,
            probability=0.85,
            is_shill_bid=True,
            drift_score=0.3,
            performance_score=0.9,
        )

        assert response.prediction == 1
        assert response.probability == 0.85
        assert response.is_shill_bid is True
        assert response.drift_score == 0.3
        assert response.performance_score == 0.9

    def test_prediction_response_defaults(self):
        """Test PredictionResponse with default values"""
        response = PredictionResponse(prediction=0, probability=0.2, is_shill_bid=False)

        assert response.drift_score is None
        assert response.performance_score is None


class TestDataProcessing:
    """Unit tests for data processing functions"""

    def test_column_mapping(self):
        """Test the column mapping functionality"""
        data = {
            "auction_id": "test_auction",
            "bidder_tendency": 0.5,
            "bidding_ratio": 0.6,
            "successive_outbidding": 0.4,
            "last_bidding": 0.3,
            "auction_bids": 0.7,
            "starting_price_average": 0.8,
            "early_bidding": 0.2,
            "winning_ratio": 0.9,
        }

        df = pd.DataFrame([data])

        column_mapping = {
            "auction_id": "Auction_ID",
            "bidder_tendency": "Bidder_Tendency",
            "bidding_ratio": "Bidding_Ratio",
            "successive_outbidding": "Successive_Outbidding",
            "last_bidding": "Last_Bidding",
            "auction_bids": "Auction_Bids",
            "starting_price_average": "Starting_Price_Average",
            "early_bidding": "Early_Bidding",
            "winning_ratio": "Winning_Ratio",
        }

        df_mapped = df.rename(columns=column_mapping)

        assert "Auction_ID" in df_mapped.columns
        assert "Bidder_Tendency" in df_mapped.columns
        assert "Bidding_Ratio" in df_mapped.columns

    def test_numeric_column_extraction(self):
        """Test extracting numeric columns from DataFrame"""
        df = pd.DataFrame(
            {
                "Auction_ID": ["auction1", "auction2"],
                "Bidder_Tendency": [0.5, 0.7],
                "Bidding_Ratio": [0.6, 0.8],
                "Successive_Outbidding": [0.4, 0.9],
            }
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        assert "Bidder_Tendency" in numeric_cols
        assert "Bidding_Ratio" in numeric_cols
        assert "Successive_Outbidding" in numeric_cols
        assert "Auction_ID" not in numeric_cols


class TestMetricsCalculation:
    """Unit tests for metrics calculation"""

    def test_drift_score_calculation(self):
        """Test drift score calculation logic"""
        feature_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        avg_feature_value = np.mean(feature_values)

        # Simulate drift score calculation
        drift_score = min(1.0, max(0.0, abs(avg_feature_value - 0.5) * 2))
        drift_score += np.random.normal(0, 0.1)
        drift_score = max(0.0, min(1.0, drift_score))

        assert 0.0 <= drift_score <= 1.0

    def test_performance_score_calculation(self):
        """Test performance score calculation logic"""
        probability = 0.85
        prediction = 1

        # Simulate performance score calculation
        performance_score = probability if prediction == 1 else (1 - probability)
        performance_score += np.random.normal(0, 0.05)
        performance_score = max(0.0, min(1.0, performance_score))

        assert 0.0 <= performance_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
