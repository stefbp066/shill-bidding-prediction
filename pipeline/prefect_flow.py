import logging
import warnings

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from prefect import flow, task
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow configuration
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("shill-bidding-prediction")


@task
def load_data(path: str) -> pd.DataFrame:
    """Load the shill bidding dataset"""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


@task
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply auction-level feature engineering for shill bidding detection"""
    logger.info("Applying auction-level feature engineering...")

    # List of features to standardize within each auction
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

    # Check which features exist in the dataset and are numeric
    available_feats = []
    for feat in auction_feats:
        if feat in df.columns:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(df[feat]):
                available_feats.append(feat)
            else:
                logger.warning(f"Skipping non-numeric feature: {feat}")

    logger.info(f"Available numeric auction features: {available_feats}")

    if not available_feats:
        logger.warning(
            "No numeric auction features found in dataset. Returning original dataframe."
        )
        return df

    # Compute per-auction mean & std
    auction_agg = df.groupby("Auction_ID")[available_feats].agg(["mean", "std"])
    auction_agg.columns = ["_".join(col) for col in auction_agg.columns]
    auction_agg = auction_agg.reset_index()
    auction_agg = auction_agg.fillna(1)

    # Merge back to main df
    df_aug = df.merge(auction_agg, on="Auction_ID", how="left")

    # Standardize per auction: (value - auction mean) / auction std
    for feat in available_feats:
        mean_col = f"{feat}_mean"
        std_col = f"{feat}_std"
        z_col = f"{feat}_z"

        # Avoid division by zero
        df_aug[std_col] = df_aug[std_col].replace(0, 1)
        df_aug[z_col] = (df_aug[feat] - df_aug[mean_col]) / df_aug[std_col]

    # Drop raw means/stds if you only want z-scores
    drop_cols = [f"{f}_mean" for f in available_feats] + [
        f"{f}_std" for f in available_feats
    ]
    df_auction_aug = df_aug.drop(columns=drop_cols)

    logger.info(
        f"Feature engineering completed. Original shape: {df.shape}, New shape: {df_auction_aug.shape}"
    )
    logger.info(f"Added {len(available_feats)} z-score features")

    return df_auction_aug


@task
def prepare_data(df: pd.DataFrame, target_col="Class", test_size=0.2, random_state=42):
    """Prepare data for training with train-test split"""
    logger.info("Preparing data for training...")

    # Clean data - remove non-numeric columns and handle any remaining issues
    logger.info("Cleaning data for model training...")

    # Get only numeric columns for features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Available numeric columns: {numeric_columns}")

    # Ensure target column is included
    if target_col not in numeric_columns:
        logger.warning(
            f"Target column '{target_col}' is not numeric. Converting to numeric..."
        )
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Separate features and target
    feature_columns = [col for col in numeric_columns if col != target_col]
    X = df[feature_columns]
    y = df[target_col]

    # Handle any remaining NaN values
    X = X.fillna(0)  # Fill NaN with 0 for features
    y = y.fillna(0)  # Fill NaN with 0 for target

    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


@task
def apply_class_balancing(X_train, y_train, method="smote", random_state=42):
    """Apply class balancing techniques for imbalanced datasets"""
    logger.info(f"Applying class balancing with method: {method}")

    # Check class distribution
    class_counts = np.bincount(y_train)
    logger.info(
        f"Original class distribution: {dict(zip(range(len(class_counts)), class_counts))}"
    )

    if method == "none":
        logger.info("No class balancing applied")
        return X_train, y_train

    try:
        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=random_state)
        elif method == "random_under":
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == "smoteenn":
            sampler = SMOTEENN(random_state=random_state)
        elif method == "smotetomek":
            sampler = SMOTETomek(random_state=random_state)
        else:
            logger.warning(f"Unknown balancing method: {method}. Using SMOTE.")
            sampler = SMOTE(random_state=random_state)

        # Apply balancing
        X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)

        # Check new class distribution
        balanced_class_counts = np.bincount(y_balanced)
        logger.info(
            f"Balanced class distribution: {dict(zip(range(len(balanced_class_counts)), balanced_class_counts))}"
        )

        return X_balanced, y_balanced

    except Exception as e:
        logger.error(f"Error in class balancing: {e}")
        logger.info("Returning original data without balancing")
        return X_train, y_train


@task
def calculate_imbalanced_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics for imbalanced classification"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # AUC metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate additional metrics for imbalanced data
    tn, fp, fn, tp = cm.ravel()

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Sensitivity (True Positive Rate / Recall for positive class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2

    # Geometric mean
    geometric_mean = np.sqrt(sensitivity * specificity)

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "auc_roc": auc_roc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "geometric_mean": geometric_mean,
        "precision_class_0": precision_per_class[0],
        "precision_class_1": precision_per_class[1],
        "recall_class_0": recall_per_class[0],
        "recall_class_1": recall_per_class[1],
        "f1_class_0": f1_per_class[0],
        "f1_class_1": f1_per_class[1],
    }

    return metrics, cm


@task
def train_lightgbm_model(X_train, y_train, balancing_method="smote"):
    """Train LightGBM model with the exact pipeline from models.py"""
    logger.info("Training LightGBM model...")

    # Apply class balancing
    X_train_balanced, y_train_balanced = apply_class_balancing(
        X_train, y_train, method=balancing_method
    )

    # Create LightGBM model with same parameters as in models.py
    model = lgb.LGBMClassifier(
        random_state=42, verbose=-1, n_estimators=100, learning_rate=0.1, max_depth=6
    )

    # Train the model
    model.fit(X_train_balanced, y_train_balanced)

    logger.info("LightGBM model training completed")
    return model


@task
def evaluate_lightgbm_model(model, X_test, y_test):
    """Evaluate LightGBM model with comprehensive metrics"""
    logger.info("Evaluating LightGBM model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate comprehensive metrics
    metrics, cm = calculate_imbalanced_metrics(y_test, y_pred, y_pred_proba)

    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Print key metrics
    logger.info(f"LightGBM Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_weighted']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")

    return metrics, cm


@task
def log_model_to_mlflow(model, metrics):
    """Log the trained model to MLflow"""
    # Remove the with mlflow.start_run(): wrapper
    # Just log directly to current run
    mlflow.log_param("model_type", "lightgbm")
    mlflow.lightgbm.log_model(model, "lightgbm_model")
    return mlflow.active_run().info.run_id


@flow(name="shillbidding_lightgbm_pipeline")
def main_pipeline(data_path: str = "data/Shill Bidding Dataset.csv"):
    """Main Prefect flow for LightGBM training pipeline"""
    logger.info("Starting Shill Bidding LightGBM Training Pipeline")

    # Load data
    df = load_data(data_path)

    # Apply feature engineering
    df_processed = feature_engineering(df)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df_processed)

    # Train LightGBM model
    model = train_lightgbm_model(X_train, y_train, balancing_method="smote")

    # Evaluate model
    metrics, cm = evaluate_lightgbm_model(model, X_test, y_test)

    # Log model to MLflow
    run_id = log_model_to_mlflow(model, metrics)

    logger.info("Pipeline completed successfully!")
    logger.info(f"Best model logged with run_id: {run_id}")

    return {
        "model": model,
        "metrics": metrics,
        "run_id": run_id,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    # Run without Prefect server for testing
    try:
        result = main_pipeline()
        print(f"Pipeline completed with run_id: {result['run_id']}")
        print(f"Best balanced accuracy: {result['metrics']['balanced_accuracy']:.4f}")
    except Exception as e:
        print(f"Error running with Prefect: {e}")
        print("Running as regular Python script...")

        # Run as regular Python functions
        logger.info("Starting Shill Bidding LightGBM Training Pipeline")

        # Load data
        df = load_data("data/Shill Bidding Dataset.csv")

        # Apply feature engineering
        df_processed = feature_engineering(df)

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df_processed)

        # Train LightGBM model
        model = train_lightgbm_model(X_train, y_train, balancing_method="smote")

        # Evaluate model
        metrics, cm = evaluate_lightgbm_model(model, X_test, y_test)

        # Log model to MLflow
        run_id = log_model_to_mlflow(model, metrics)

        logger.info("Pipeline completed successfully!")
        logger.info(f"Best model logged with run_id: {run_id}")
        print(f"Pipeline completed with run_id: {run_id}")
        print(f"Best balanced accuracy: {metrics['balanced_accuracy']:.4f}")
