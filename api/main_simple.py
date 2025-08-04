# api/main_simple.py
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "evidently_prediction_count_total", "Total number of predictions"
)
PREDICTION_ERRORS = Counter(
    "evidently_prediction_errors_total", "Total number of prediction errors"
)
PREDICTION_DURATION = Histogram(
    "evidently_prediction_duration_seconds", "Prediction duration in seconds"
)
DRIFT_SCORE = Gauge("evidently_data_drift_score", "Data drift score")
MODEL_PERFORMANCE = Gauge(
    "evidently_model_performance_score", "Model performance score"
)

app = FastAPI(title="Shill Bidding Prediction API (Simple)")


class BidData(BaseModel):
    auction_id: str
    bidder_tendency: float = Field(
        ..., ge=0.0, le=1.0, description="Bidder tendency score between 0 and 1"
    )
    bidding_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Bidding ratio between 0 and 1"
    )
    successive_outbidding: float = Field(
        ..., ge=0.0, le=1.0, description="Successive outbidding score between 0 and 1"
    )
    last_bidding: float = Field(
        ..., ge=0.0, le=1.0, description="Last bidding score between 0 and 1"
    )
    auction_bids: float = Field(
        ..., ge=0.0, le=1.0, description="Auction bids score between 0 and 1"
    )
    starting_price_average: float = Field(
        ..., ge=0.0, le=1.0, description="Starting price average between 0 and 1"
    )
    early_bidding: float = Field(
        ..., ge=0.0, le=1.0, description="Early bidding score between 0 and 1"
    )
    winning_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Winning ratio between 0 and 1"
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_shill_bid: bool
    drift_score: float = None
    performance_score: float = None
    monitoring_info: dict[str, Any] = None


# Mock model for testing
class MockModel:
    def predict(self, X):
        # Simple mock prediction based on features
        return np.random.choice([0, 1], size=len(X))

    def predict_proba(self, X):
        # Mock probabilities
        probs = np.random.random(len(X))
        return np.column_stack([1 - probs, probs])


# Initialize mock model
model = MockModel()


@app.post("/predict", response_model=PredictionResponse)
async def predict_shill_bidding(bid_data: BidData):
    """Predict if a bid is shill bidding"""
    start_time = time.time()

    try:
        # Convert to DataFrame
        df = pd.DataFrame([bid_data.model_dump()])

        # Convert column names to match training data format
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
        df = df.rename(columns=column_mapping)

        # Get only numeric columns for prediction
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of positive class

        # Update Prometheus metrics
        PREDICTION_COUNTER.inc()
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        # Calculate realistic drift and performance scores
        # Simulate drift based on feature values
        feature_values = list(bid_data.model_dump().values())[1:]  # Exclude auction_id
        avg_feature_value = np.mean(
            [v for v in feature_values if isinstance(v, (int, float))]
        )

        # Drift score based on how far features are from expected range
        drift_score = min(1.0, max(0.0, abs(avg_feature_value - 0.5) * 2))

        # Performance score based on prediction confidence
        performance_score = probability if prediction == 1 else (1 - probability)

        # Add some realistic variation
        drift_score += np.random.normal(0, 0.1)
        performance_score += np.random.normal(0, 0.05)

        # Clamp values
        drift_score = max(0.0, min(1.0, drift_score))
        performance_score = max(0.0, min(1.0, performance_score))

        # Update Prometheus gauges
        DRIFT_SCORE.set(drift_score)
        MODEL_PERFORMANCE.set(performance_score)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            is_shill_bid=bool(prediction == 1),
            drift_score=drift_score,
            performance_score=performance_score,
            monitoring_info={
                "drift_score": drift_score,
                "performance_score": performance_score,
                "prediction": int(prediction),
                "probability": float(probability),
                "timestamp": pd.Timestamp.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(bid_data_list: list[BidData]):
    """Predict multiple bids at once"""
    start_time = time.time()

    try:
        # Convert to DataFrame
        df = pd.DataFrame([bid.dict() for bid in bid_data_list])

        # Convert column names
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
        df = df.rename(columns=column_mapping)

        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Update Prometheus metrics
        PREDICTION_COUNTER.inc(len(predictions))
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Calculate drift and performance for each prediction
            bid_data = bid_data_list[i]
            feature_values = list(bid_data.dict().values())[1:]  # Exclude auction_id
            avg_feature_value = np.mean(
                [v for v in feature_values if isinstance(v, (int, float))]
            )

            # Drift score
            drift_score = min(1.0, max(0.0, abs(avg_feature_value - 0.5) * 2))
            drift_score += np.random.normal(0, 0.1)
            drift_score = max(0.0, min(1.0, drift_score))

            # Performance score
            performance_score = prob if pred == 1 else (1 - prob)
            performance_score += np.random.normal(0, 0.05)
            performance_score = max(0.0, min(1.0, performance_score))

            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    probability=float(prob),
                    is_shill_bid=bool(pred == 1),
                    drift_score=drift_score,
                    performance_score=performance_score,
                    monitoring_info={
                        "drift_score": drift_score,
                        "performance_score": performance_score,
                        "prediction": int(pred),
                        "probability": float(prob),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    },
                )
            )

        # Update overall metrics with averages
        avg_drift = np.mean([r.drift_score for r in results])
        avg_performance = np.mean([r.performance_score for r in results])
        DRIFT_SCORE.set(avg_drift)
        MODEL_PERFORMANCE.set(avg_performance)

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_name": "mock-model",
        "model_version": "1.0.0",
        "model_type": "mock",
        "monitoring": "enabled",
    }


@app.get("/features")
async def get_required_features():
    """Get the list of required features for prediction"""
    return {
        "required_features": [
            "auction_id",
            "bidder_tendency",
            "bidding_ratio",
            "successive_outbidding",
            "last_bidding",
            "auction_bids",
            "starting_price_average",
            "early_bidding",
            "winning_ratio",
        ],
        "feature_engineering": "mock_prediction",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=os.getenv("API_PORT", 8000))
