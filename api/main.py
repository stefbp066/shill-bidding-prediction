# api/main.py
import logging
import os
import sys
import time
from typing import Any

import mlflow
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
from pydantic import BaseModel

# Import monitoring
sys.path.append("..")
from monitoring.monitoring_evidently import get_monitor

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

app = FastAPI(title="Shill Bidding Prediction API")

# MLflow configuration - support environment variables for cloud deployment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_NAME = os.getenv("MODEL_NAME", "shill-bidding-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize monitoring
monitor = get_monitor()


# Load the trained model
def load_model():
    """Load the best model from MLflow model registry"""
    try:
        # Load the specified version of the registered model
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        logger.info(f"Attempting to load model from: {model_uri}")

        # Load the model and handle potential tuple return
        loaded_model = mlflow.lightgbm.load_model(model_uri)

        # Check if it's a tuple and extract the model
        if isinstance(loaded_model, tuple):
            # Handle different tuple formats: (model, conda_env) or 
            # (model, conda_env, code_paths)
            if len(loaded_model) >= 2:
                # The actual model is typically at index 1 in this format
                model = loaded_model[1]  # Extract the actual model
                logger.info(
                    f"Model loaded from tuple with {len(loaded_model)} elements"
                )
            else:
                raise ValueError("Tuple too short for model extraction")
        else:
            model = loaded_model
            logger.info("Model loaded directly")

        logger.info(f"Model loaded successfully from MLflow registry: {model_uri}")
        logger.info(f"Model type: {type(model)}")

        # Test the model has predict method
        if hasattr(model, "predict"):
            logger.info("Model has predict method - ready for inference")
        else:
            raise ValueError("Loaded model does not have predict method")

        return model

    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        # Fallback to local model file if MLflow fails
        try:
            logger.info("Attempting to load local model file...")
            model_path = (
                "../mlruns/108907814007183194/0d13c7c9352744c9a69c046d1d662ef3/"
                "artifacts/lightgbm_model"
            )
            model = mlflow.lightgbm.load_model(model_path)
            logger.info("Local model loaded successfully")
            return model
        except Exception as local_error:
            logger.error(f"Error loading local model: {local_error}")
            raise


# Load model at startup
model = load_model()

# Store auction statistics for feature engineering
auction_stats = {}


def load_auction_statistics():
    """Load pre-computed auction statistics for feature engineering"""
    global auction_stats
    try:
        import json

        with open("api/auction_stats.json") as f:
            auction_stats = json.load(f)
        logger.info(f"Auction statistics loaded for {len(auction_stats)} auctions")
    except Exception as e:
        logger.warning(f"Could not load auction statistics: {e}")
        auction_stats = {}


load_auction_statistics()


class BidData(BaseModel):
    auction_id: str
    bidder_tendency: float
    bidding_ratio: float
    successive_outbidding: float
    last_bidding: float
    auction_bids: float
    starting_price_average: float
    early_bidding: float
    winning_ratio: float
    # Add other required features


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_shill_bid: bool
    drift_score: float = None
    monitoring_info: dict[str, Any] = None


def apply_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as in training"""
    logger.info("Applying feature engineering to incoming data...")

    # List of features to standardize (same as training)
    auction_feats = [
        "Bidder_Tendency",
        "Bidding_Ratio",
        "Successive_Outbidding",
        "Last_Bidding",
        "Auction_Bids",
        "Starting_Price_Average",
        "Early_Bidding",
        "Winning_Ratio",
    ]

    # Check which features exist
    available_feats = [f for f in auction_feats if f in data.columns]

    if not available_feats:
        logger.warning("No auction features found for engineering")
        return data

    # Create a copy to avoid modifying original data
    df_engineered = data.copy()

    # For each auction in the incoming data
    for auction_id in data["Auction_ID"].unique():
        auction_data = data[data["Auction_ID"] == auction_id]

        # Use pre-computed statistics if available, otherwise compute from current data
        auction_id_str = str(auction_id)
        if auction_id_str in auction_stats:
            auction_means = auction_stats[auction_id_str]["means"]
            auction_stds = auction_stats[auction_id_str]["stds"]
            logger.info(f"Using pre-computed statistics for auction {auction_id}")
        else:
            # Compute auction statistics from current data
            auction_means = auction_data[available_feats].mean()
            auction_stds = auction_data[available_feats].std()
            auction_stds = auction_stds.replace(0, 1)
            logger.info(f"Computing statistics for new auction {auction_id}")

        # Apply z-score standardization
        for feat in available_feats:
            z_col = f"{feat}_z"
            if isinstance(auction_means, dict):
                mean_val = auction_means.get(feat, 0)
                std_val = auction_stds.get(feat, 1)
            else:
                mean_val = auction_means[feat]
                std_val = auction_stds[feat]

            df_engineered.loc[df_engineered["Auction_ID"] == auction_id, z_col] = (
                df_engineered.loc[df_engineered["Auction_ID"] == auction_id, feat]
                - mean_val
            ) / std_val

    # Keep both original features and z-scores (model expects both)
    logger.info(f"Feature engineering completed. Shape: {df_engineered.shape}")
    logger.info(f"Columns after engineering: {list(df_engineered.columns)}")

    # Add missing columns that the model expects
    expected_columns = [
        "Record_ID",
        "Auction_ID",
        "Bidder_Tendency",
        "Bidding_Ratio",
        "Successive_Outbidding",
        "Last_Bidding",
        "Auction_Bids",
        "Starting_Price_Average",
        "Early_Bidding",
        "Winning_Ratio",
        "Auction_Duration",
        "Bidder_Tendency_z",
        "Bidding_Ratio_z",
        "Successive_Outbidding_z",
        "Last_Bidding_z",
        "Auction_Bids_z",
        "Starting_Price_Average_z",
        "Early_Bidding_z",
        "Winning_Ratio_z",
    ]

    for col in expected_columns:
        if col not in df_engineered.columns:
            if col == "Record_ID":
                df_engineered[col] = 1  # Default value
            elif col == "Auction_Duration":
                df_engineered[col] = 0  # Default value
            else:
                df_engineered[col] = 0  # Default value for missing z-scores

    # Ensure columns are in the same order as training
    df_engineered = df_engineered.reindex(columns=expected_columns)

    logger.info(f"Final shape: {df_engineered.shape}")
    logger.info(f"Final columns: {list(df_engineered.columns)}")
    return df_engineered


@app.post("/predict", response_model=PredictionResponse)
async def predict_shill_bidding(bid_data: BidData):
    """Predict if a bid is shill bidding"""
    start_time = time.time()

    try:
        # Convert to DataFrame
        df = pd.DataFrame([bid_data.dict()])

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

        # Apply feature engineering
        df_engineered = apply_feature_engineering(df)

        # Get only numeric columns for prediction
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        X = df_engineered[numeric_cols]

        # Debug: Print what features we have
        logger.info(f"Features we have: {list(X.columns)}")
        logger.info(f"Number of features: {len(X.columns)}")

        # Temporarily disable shape check for testing
        prediction = model.predict(X, predict_disable_shape_check=True)[0]
        probability = model.predict_proba(X, predict_disable_shape_check=True)[0][
            1
        ]  # Probability of positive class

        # Update Prometheus metrics
        PREDICTION_COUNTER.inc()
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        # Monitor the prediction
        monitoring_result = monitor.monitor_prediction(
            features=bid_data.dict(),
            prediction=int(prediction),
            probability=float(probability),
        )

        # Update drift score metric
        DRIFT_SCORE.set(monitoring_result["drift_score"])

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            is_shill_bid=bool(prediction == 1),
            drift_score=monitoring_result["drift_score"],
            monitoring_info=monitoring_result,
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

        # Apply feature engineering
        df_engineered = apply_feature_engineering(df)

        # Get only numeric columns
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        X = df_engineered[numeric_cols]

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Update Prometheus metrics
        PREDICTION_COUNTER.inc(len(predictions))
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Monitor each prediction
            monitoring_result = monitor.monitor_prediction(
                features=bid_data_list[i].dict(),
                prediction=int(pred),
                probability=float(prob),
            )

            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    probability=float(prob),
                    is_shill_bid=bool(pred == 1),
                    drift_score=monitoring_result["drift_score"],
                    monitoring_info=monitoring_result,
                )
            )

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
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_type": "lightgbm",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "feature_engineering": "auction_level_z_scores",
            "class_balancing": "smote",
            "monitoring": "evidently_ai",
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        "feature_engineering": "auction_level_z_score_standardization",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=os.getenv("API_PORT", 8000))
