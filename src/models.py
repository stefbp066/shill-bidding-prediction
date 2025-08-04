import logging
import warnings

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from mlflow import MlflowClient
from optuna.integration import MLflowCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/model_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# MLflow configuration
EXPERIMENT_NAME = "shill-bidding-prediction"
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)


class ShillBiddingModelTrainer:
    def __init__(self, data_path="data/Shill Bidding Dataset.csv"):
        """
        Initialize the model trainer with data path and MLflow setup
        """
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

        logger.info(f"Initialized ShillBiddingModelTrainer with data path: {data_path}")

    def load_data(self):
        """
        Load and prepare the dataset
        """
        logger.info("Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def feature_engineering(self, df):
        """
        Auction-level feature engineering for shill bidding detection
        """
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

    def prepare_data(self, target_col="Class", test_size=0.2, random_state=42):
        """
        Prepare data for training with train-test split
        """
        logger.info("Preparing data for training...")

        # Apply feature engineering
        df_processed = self.feature_engineering(self.df)

        # Clean data - remove non-numeric columns and handle any remaining issues
        logger.info("Cleaning data for model training...")

        # Get only numeric columns for features
        numeric_columns = df_processed.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        logger.info(f"Available numeric columns: {numeric_columns}")

        # Ensure target column is included
        if target_col not in numeric_columns:
            logger.warning(
                f"Target column '{target_col}' is not numeric. Converting to numeric..."
            )
            df_processed[target_col] = pd.to_numeric(
                df_processed[target_col], errors="coerce"
            )

        # Separate features and target
        feature_columns = [col for col in numeric_columns if col != target_col]
        X = df_processed[feature_columns]
        y = df_processed[target_col]

        # Handle any remaining NaN values
        X = X.fillna(0)  # Fill NaN with 0 for features
        y = y.fillna(0)  # Fill NaN with 0 for target

        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Feature columns: {list(X.columns)}")

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(
            f"Data split completed. Train: {self.X_train.shape}, Test: {self.X_test.shape}"
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def cross_validation_split(self, n_splits=5, random_state=42):
        """
        Cross-validation split function using StratifiedKFold
        """
        logger.info(f"Setting up {n_splits}-fold cross-validation...")

        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        logger.info("Cross-validation setup completed")
        return cv

    def apply_class_balancing(self, X, y, method="smote", random_state=42):
        """
        Apply class balancing techniques for imbalanced datasets

        Args:
            X: Feature matrix
            y: Target variable
            method: Balancing method ('smote', 'adasyn', 'random_under', 'smoteenn', 'smotetomek', 'none')
            random_state: Random seed
        """
        logger.info(f"Applying class balancing with method: {method}")

        # Check class distribution
        class_counts = np.bincount(y)
        logger.info(
            f"Original class distribution: {dict(zip(range(len(class_counts)), class_counts))}"
        )

        if method == "none":
            logger.info("No class balancing applied")
            return X, y

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
            X_balanced, y_balanced = sampler.fit_resample(X, y)

            # Check new class distribution
            balanced_class_counts = np.bincount(y_balanced)
            logger.info(
                f"Balanced class distribution: {dict(zip(range(len(balanced_class_counts)), balanced_class_counts))}"
            )

            return X_balanced, y_balanced

        except Exception as e:
            logger.error(f"Error in class balancing: {e}")
            logger.info("Returning original data without balancing")
            return X, y

    def calculate_imbalanced_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate comprehensive metrics for imbalanced classification
        """
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

    def get_models(self):
        """
        Define the models to train
        """
        models = {
            "logistic_regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]},
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
            },
            "lightgbm": {
                "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
            },
            "xgboost": {
                "model": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
            },
        }

        logger.info(f"Defined {len(models)} models for training")
        return models

    def train_baseline_models(self, balancing_method="smote"):
        """
        Train baseline models with class balancing and comprehensive metrics
        """
        logger.info("Starting baseline model training...")

        models = self.get_models()
        cv = self.cross_validation_split()

        for name, model_config in models.items():
            logger.info(f"Training {name}...")

            with mlflow.start_run(run_name=f"baseline_{name}"):
                # Log parameters
                mlflow.log_param("model_type", name)
                mlflow.log_param("cv_folds", cv.n_splits)
                mlflow.log_param("feature_engineering", "auction_level_z_scores")
                mlflow.log_param("class_balancing", balancing_method)

                # Apply class balancing
                X_train_balanced, y_train_balanced = self.apply_class_balancing(
                    self.X_train, self.y_train, method=balancing_method
                )

                # Train model
                model = model_config["model"]
                model.fit(X_train_balanced, y_train_balanced)

                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

                # Calculate comprehensive metrics
                metrics, cm = self.calculate_imbalanced_metrics(
                    self.y_test, y_pred, y_pred_proba
                )

                # Log all metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model
                mlflow.sklearn.log_model(model, f"{name}_model")

                # Store results
                self.models[name] = model
                self.results[name] = {
                    **metrics,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                    "confusion_matrix": cm,
                }

                logger.info(
                    f"{name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc_roc']:.4f}"
                )
                logger.info(
                    f"{name} - F1: {metrics['f1_weighted']:.4f}, Balanced Acc: {metrics['balanced_accuracy']:.4f}"
                )

        logger.info("Baseline model training completed")
        return self.results

    def hyperparameter_optimization(self, model_name, n_trials=50):
        """
        Hyperparameter optimization using Optuna
        """
        logger.info(f"Starting hyperparameter optimization for {model_name}...")

        def objective(trial):
            # Define hyperparameter search space based on model type
            if model_name == "logistic_regression":
                C = trial.suggest_float("C", 0.01, 100, log=True)
                penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
                model = LogisticRegression(
                    C=C, penalty=penalty, random_state=42, max_iter=1000
                )

            elif model_name == "random_forest":
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth = trial.suggest_int("max_depth", 5, 30)
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )

            elif model_name == "lightgbm":
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                learning_rate = trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                )
                max_depth = trial.suggest_int("max_depth", 3, 10)
                model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                    verbose=-1,
                )

            elif model_name == "xgboost":
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                learning_rate = trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                )
                max_depth = trial.suggest_int("max_depth", 3, 10)
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                    eval_metric="logloss",
                )

            # Cross-validation score
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, cv=5, scoring="roc_auc"
            )
            return cv_scores.mean()

        # Set up Optuna with MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="cv_score"
        )

        study = optuna.create_study(
            direction="maximize", study_name=f"{model_name}_optimization"
        )

        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

        logger.info(f"Hyperparameter optimization completed for {model_name}")
        return study

    def train_optimized_models(self, n_trials=50):
        """
        Train models with hyperparameter optimization
        """
        logger.info("Starting optimized model training...")

        models = self.get_models()

        for name in models.keys():
            logger.info(f"Optimizing {name}...")

            # Run hyperparameter optimization
            study = self.hyperparameter_optimization(name, n_trials)

            # Train with best parameters
            with mlflow.start_run(run_name=f"optimized_{name}"):
                # Get best parameters
                best_params = study.best_params
                logger.info(f"Best parameters for {name}: {best_params}")

                # Create model with best parameters
                if name == "logistic_regression":
                    model = LogisticRegression(
                        **best_params, random_state=42, max_iter=1000
                    )
                elif name == "random_forest":
                    model = RandomForestClassifier(**best_params, random_state=42)
                elif name == "lightgbm":
                    model = lgb.LGBMClassifier(
                        **best_params, random_state=42, verbose=-1
                    )
                elif name == "xgboost":
                    model = xgb.XGBClassifier(
                        **best_params, random_state=42, eval_metric="logloss"
                    )

                # Train model
                model.fit(self.X_train, self.y_train)

                # Predictions and metrics
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

                accuracy = accuracy_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba)

                # Log everything
                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("auc", auc)
                mlflow.log_metric("best_cv_score", study.best_value)
                mlflow.sklearn.log_model(model, f"optimized_{name}_model")

                logger.info(
                    f"Optimized {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}"
                )

        logger.info("Optimized model training completed")

    def evaluate_models(self):
        """
        Comprehensive model evaluation
        """
        logger.info("Evaluating all trained models...")

        evaluation_results = {}

        for name, result in self.results.items():
            logger.info(f"Evaluating {name}...")

            # Classification report
            report = classification_report(self.y_test, result["predictions"])
            logger.info(f"{name} Classification Report:\n{report}")

            # Confusion matrix
            cm = result.get(
                "confusion_matrix", confusion_matrix(self.y_test, result["predictions"])
            )
            logger.info(f"{name} Confusion Matrix:\n{cm}")

            evaluation_results[name] = {
                "classification_report": report,
                "confusion_matrix": cm,
                "accuracy": result.get("accuracy", 0),
                "auc_roc": result.get("auc_roc", 0),
                "balanced_accuracy": result.get("balanced_accuracy", 0),
                "f1_weighted": result.get("f1_weighted", 0),
                "sensitivity": result.get("sensitivity", 0),
                "specificity": result.get("specificity", 0),
            }

        return evaluation_results

    def save_best_model(self, metric="balanced_accuracy"):
        """
        Save the best performing model based on specified metric
        For imbalanced datasets, balanced_accuracy is recommended
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return None

        # Find best model
        best_model_name = max(
            self.results.keys(), key=lambda x: self.results[x].get(metric, 0)
        )
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name].get(metric, 0)

        logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.4f}")

        # Log additional metrics for the best model
        if metric in self.results[best_model_name]:
            logger.info(f"Best model additional metrics:")
            for key, value in self.results[best_model_name].items():
                if isinstance(value, (int, float)) and key != metric:
                    logger.info(f"  {key}: {value:.4f}")

        # Save model (you can implement your preferred saving method)
        # Example: joblib.dump(best_model, f"best_{best_model_name}.pkl")

        return best_model_name, best_model, best_score


def main():
    """
    Main function to run the complete training pipeline with class balancing
    """
    logger.info("Starting Shill Bidding Prediction Model Training Pipeline")

    # Initialize trainer
    trainer = ShillBiddingModelTrainer()

    # Load data
    trainer.load_data()

    # Prepare data
    trainer.prepare_data()

    # Test different balancing methods
    balancing_methods = ["none", "smote", "adasyn", "smoteenn"]

    all_results = {}

    for method in balancing_methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with balancing method: {method}")
        logger.info(f"{'='*50}")

        # Train baseline models with current balancing method
        baseline_results = trainer.train_baseline_models(balancing_method=method)
        all_results[method] = baseline_results

        # Evaluate models
        evaluation_results = trainer.evaluate_models()

        # Save best model for this method
        best_model_info = trainer.save_best_model(metric="balanced_accuracy")

        logger.info(f"Completed training with {method} balancing")

    # Find overall best model across all methods
    logger.info(f"\n{'='*50}")
    logger.info("COMPARING ALL BALANCING METHODS")
    logger.info(f"{'='*50}")

    best_overall_score = 0
    best_overall_method = None
    best_overall_model = None

    for method, results in all_results.items():
        for model_name, result in results.items():
            # Use balanced accuracy as primary metric for imbalanced data
            score = result.get("balanced_accuracy", 0)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_method = method
                best_overall_model = model_name

    logger.info(
        f"Best overall model: {best_overall_model} with {best_overall_method} balancing"
    )
    logger.info(f"Best balanced accuracy: {best_overall_score:.4f}")

    logger.info("Training pipeline completed successfully")

    return trainer, all_results, best_overall_method, best_overall_model


if __name__ == "__main__":
    trainer, all_results, best_method, best_model = main()
