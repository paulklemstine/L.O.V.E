# core/learning_module.py
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles data ingestion from various sources (APIs, files, streams)."""

    def fetch_data(self, source: str, source_type: str = "api") -> Dict[str, Any]:
        """
        Fetch data from the specified source.

        Args:
            source: URL for API or file path
            source_type: Type of source ('api', 'file', 'stream')

        Returns:
            Dictionary containing the fetched data

        Raises:
            ValueError: If source type is invalid or data cannot be fetched
        """
        try:
            if source_type == "api":
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                return response.json()
            elif source_type == "file":
                with open(source, "r") as f:
                    if source.endswith(".json"):
                        return json.load(f)
                    elif source.endswith(".csv"):
                        return pd.read_csv(source).to_dict("records")
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {str(e)}")
            raise ValueError(f"Failed to fetch data: {str(e)}")

    def validate_data(
        self, data: Dict[str, Any], schema: Optional[Dict] = None
    ) -> bool:
        """
        Validate the structure and content of the data.

        Args:
            data: Data to validate
            schema: Optional schema to validate against

        Returns:
            True if data is valid, False otherwise
        """
        if not data:
            return False

        if schema:
            # Basic schema validation
            for key, value_type in schema.items():
                if key not in data:
                    return False
                if not isinstance(data[key], value_type):
                    return False

        return True


class DataPreprocessor:
    """Handles data cleaning, normalization, and feature extraction."""

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.

        Args:
            data: Raw input data

        Returns:
            Cleaned DataFrame
        """
        try:
            # Handle missing values
            data = data.dropna()

            # Remove outliers using IQR method
            for col in data.select_dtypes(include=[np.number]).columns:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

            return data
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.

        Args:
            data: Input data

        Returns:
            Normalized DataFrame
        """
        try:
            scaler = StandardScaler()
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
            return data
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from the data.

        Args:
            data: Input data

        Returns:
            DataFrame with additional features
        """
        try:
            # Example feature extraction - add more as needed
            if "timestamp" in data.columns:
                data["hour"] = pd.to_datetime(data["timestamp"]).dt.hour
                data["day_of_week"] = pd.to_datetime(data["timestamp"]).dt.dayofweek

            return data
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise


class LearningEngine:
    """Handles machine learning model training and prediction."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the learning engine.

        Args:
            model_path: Path to load a pre-trained model from
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train_model(self, data: pd.DataFrame, target_col: str) -> None:
        """
        Train or update the machine learning model.

        Args:
            data: Training data
            target_col: Name of the target column
        """
        try:
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")

            X = data.drop(columns=[target_col])
            y = data[target_col]

            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")

            self.is_trained = True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            data: Input data for prediction

        Returns:
            Array of predictions

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_model() first.")

        try:
            data_scaled = self.scaler.transform(data)
            return self.model.predict(data_scaled)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model's performance.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            Dictionary of performance metrics
        """
        try:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)

            return {
                "accuracy": accuracy_score(y_test, y_pred),
                # Add more metrics as needed
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, model_path: str) -> None:
        """Save the trained model to disk."""
        try:
            joblib.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "is_trained": self.is_trained,
                },
                model_path,
            )
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model from disk."""
        try:
            loaded = joblib.load(model_path)
            self.model = loaded["model"]
            self.scaler = loaded["scaler"]
            self.is_trained = loaded["is_trained"]
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


class RealTimeAnalyzer:
    """Main class that orchestrates the real-time data analysis pipeline."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the real-time analyzer.

        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.ingestor = DataIngestor()
        self.preprocessor = DataPreprocessor()
        self.engine = LearningEngine(model_path)

    def process_stream(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Process a stream of data through the entire pipeline.

        Args:
            data: Input data
            target_col: Name of the target column

        Returns:
            Dictionary containing predictions and metrics
        """
        try:
            # Preprocess data
            cleaned_data = self.preprocessor.clean_data(data)
            normalized_data = self.preprocessor.normalize(cleaned_data)
            processed_data = self.preprocessor.extract_features(normalized_data)

            # Train or update model
            self.engine.train_model(processed_data, target_col)

            # Make predictions
            predictions = self.engine.predict(processed_data.drop(columns=[target_col]))

            return {
                "predictions": predictions.tolist(),
                "status": "success",
                "processed_records": len(processed_data),
            }
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            return {"status": "error", "error": str(e)}

    def log_insights(
        self, results: Dict[str, Any], log_file: str = "insights.log"
    ) -> None:
        """
        Log the analysis results to a file.

        Args:
            results: Results from process_stream
            log_file: Path to the log file
        """
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(results) + "\n")
        except Exception as e:
            logger.error(f"Error logging insights: {str(e)}")
            raise
