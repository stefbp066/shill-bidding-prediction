#!/usr/bin/env python3
"""
Script to register the best model in MLflow model registry for API deployment
"""

import logging
from datetime import datetime

import mlflow

from models import ShillBiddingModelTrainer

today_str = datetime.today().strftime("%Y%m%d")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("shill-bidding-prediction")


def register_best_model():
    """Register the best model in MLflow model registry"""
    try:
        # Initialize the trainer
        trainer = ShillBiddingModelTrainer()

        # Load and prepare data
        trainer.load_data()
        trainer.prepare_data()

        # Train models and get the best one
        trainer.train_baseline_models(balancing_method="smote")

        # Get the best model based on balanced accuracy
        best_model = trainer.save_best_model(metric="balanced_accuracy")

        # Register the model in MLflow model registry
        model_name = "shill-bidding-model"

        # Log the model to the registry
        with mlflow.start_run(run_name="model_registration"):
            # Log model parameters
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("feature_engineering", "auction_level_z_scores")
            mlflow.log_param("class_balancing", "smote")
            mlflow.log_param("metric_used", "balanced_accuracy")

            # Log the model
            mlflow.lightgbm.log_model(best_model, "lightgbm_model")

            # Register the model
            result = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/lightgbm_model",
                name=model_name,
            )

        logger.info(
            f"Model registered successfully: {result.name} (version {result.version})"
        )

        # Also save auction statistics for feature engineering
        import json

        auction_stats = {}

        # Get auction statistics from the processed data
        df_processed = trainer.feature_engineering(trainer.df)

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

        available_feats = [f for f in auction_feats if f in df_processed.columns]

        for auction_id in df_processed["Auction_ID"].unique():
            auction_data = df_processed[df_processed["Auction_ID"] == auction_id]
            # Convert numpy types to native Python types for JSON serialization
            auction_stats[str(auction_id)] = {
                "means": {
                    k: float(v)
                    for k, v in auction_data[available_feats].mean().to_dict().items()
                },
                "stds": {
                    k: float(v)
                    for k, v in auction_data[available_feats]
                    .std()
                    .replace(0, 1)
                    .to_dict()
                    .items()
                },
            }

        # Save auction statistics
        import os

        os.makedirs("api", exist_ok=True)
        with open("api/auction_stats.json", "w") as f:
            json.dump(auction_stats, f)

        logger.info("Auction statistics saved to api/auction_stats.json")

        return result

    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise


if __name__ == "__main__":
    register_best_model()
