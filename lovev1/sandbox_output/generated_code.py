import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import yaml
import logging
import sys
import os
import functools  # For reducing complexity
from typing import List, Dict  # For robust type hinting

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Configuration Loading ---
def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    # Ensure the path is absolute or relative to the current working directory
    # os.path.abspath handles both relative paths (e.g., "config/config.yaml")
    # and absolute paths, resolving them to a full path.
    absolute_config_path = os.path.abspath(config_path)

    if not os.path.exists(absolute_config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {absolute_config_path}"
        )
    try:
        with open(absolute_config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        logging.error(
            f"Error loading YAML configuration from {absolute_config_path}: {e}"
        )
        raise


# --- Data Handling ---


class DataSource:
    """
    Abstract base class for data sources.
    """

    def __init__(self, config: Dict):
        self.config = config

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches data from the source.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement 'fetch_data'")


class CsvDataSource(DataSource):
    """
    Data source for reading data from a CSV file.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.file_path = config.get("file_path")
        if not self.file_path:
            raise ValueError("CSV data source requires 'file_path' in configuration.")
        self.file_path = os.path.abspath(self.file_path)  # Ensure absolute path

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches data from the specified CSV file.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            pd.errors.EmptyDataError: If the CSV file is empty.
            Exception: For other potential pandas read errors.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found at: {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            if df.empty:
                raise pd.errors.EmptyDataError("CSV file is empty.")
            logging.info(f"Successfully loaded {len(df)} rows from {self.file_path}")
            return df
        except pd.errors.EmptyDataError as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            raise
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while loading CSV {self.file_path}: {e}"
            )
            raise


class MockDataSource(DataSource):
    """
    A mock data source for testing purposes.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.mock_data_config = config.get("mock_data_config", {})

    def fetch_data(self) -> pd.DataFrame:
        """
        Generates mock data based on configuration.
        Fixes: Robustness for small `num_rows` in anomaly injection.

        Returns:
            pd.DataFrame: Mock data.
        """
        num_rows = self.mock_data_config.get("num_rows", 100)
        num_cols = self.mock_data_config.get("num_cols", 5)
        data_range = self.mock_data_config.get("data_range", (0, 100))
        col_names = [f"feature_{i + 1}" for i in range(num_cols)]

        data = np.random.randint(
            data_range[0], data_range[1], size=(num_rows, num_cols)
        )
        df = pd.DataFrame(data, columns=col_names)

        # Inject some 'anomalous' or potentially interesting patterns
        # FIX 1: Adjust the guard condition and subset_size calculation for robustness
        # The condition `num_rows >= 5` ensures there are enough rows for `max(5, ...)` when `replace=False`.
        if num_rows >= 5 and num_cols > 1:
            # Ensure subset_size doesn't exceed num_rows to prevent ValueError in np.random.choice
            subset_size = min(max(5, num_rows // 10), num_rows)
            subset_indices = np.random.choice(num_rows, size=subset_size, replace=False)
            df.loc[subset_indices, "feature_2"] = df.loc[
                subset_indices, "feature_1"
            ] * 2 + np.random.randint(-5, 5, size=len(subset_indices))

            # Add a potential outlier
            # FIX 1: Ensure there's at least one row before trying to create an outlier.
            if num_rows > 0:
                outlier_row = np.random.randint(0, num_rows)
                df.loc[outlier_row, "feature_3"] += np.random.randint(50, 100)

        logging.info(f"Generated {num_rows} rows of mock data with {num_cols} columns.")
        return df


def get_data_source(config: Dict) -> DataSource:
    """
    Factory function to create a DataSource instance based on configuration.

    Args:
        config (dict): The data source configuration dictionary.

    Returns:
        DataSource: An instance of a concrete DataSource subclass.

    Raises:
        ValueError: If the 'type' is not recognized or required parameters are missing.
    """
    source_type = config.get("type")
    if not source_type:
        raise ValueError("Data source configuration must include a 'type'.")

    if source_type.lower() == "csv":
        return CsvDataSource(config)
    elif source_type.lower() == "mock":
        return MockDataSource(config)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")


# --- Data Preprocessing Helper Functions ---


# FIX 4: Improved type hinting
def _handle_missing_values(
    df: pd.DataFrame, strategy_config: Dict[str, str]
) -> pd.DataFrame:
    """
    Handles missing values based on the provided strategy configuration.
    `strategy_config` is expected to map column names (str) to strategy names (str, e.g., 'mean', 'median').
    """
    processed_df = df.copy()
    for col, strategy in strategy_config.items():
        if col not in processed_df.columns:
            logging.warning(
                f"Column '{col}' specified for missing value imputation not found."
            )
            continue
        if processed_df[col].isnull().any():
            if strategy == "mean":
                fill_value = processed_df[col].mean()
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col].fillna(fill_value, inplace=True)
                    logging.info(
                        f"Filled missing values in '{col}' with mean ({fill_value:.2f})."
                    )
                else:
                    logging.warning(
                        f"Column '{col}' is not numeric, cannot fill with mean."
                    )
            elif strategy == "median":
                fill_value = processed_df[col].median()
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col].fillna(fill_value, inplace=True)
                    logging.info(
                        f"Filled missing values in '{col}' with median ({fill_value:.2f})."
                    )
                else:
                    logging.warning(
                        f"Column '{col}' is not numeric, cannot fill with median."
                    )
            elif strategy == "mode":
                # mode() can return multiple values, take the first one
                fill_values = processed_df[col].mode()
                if not fill_values.empty:
                    fill_value = fill_values[0]
                    processed_df[col].fillna(fill_value, inplace=True)
                    logging.info(
                        f"Filled missing values in '{col}' with mode ({fill_value})."
                    )
                else:
                    logging.warning(f"Could not determine mode for column '{col}'.")
            elif strategy == "drop":
                initial_rows = len(processed_df)
                processed_df.dropna(subset=[col], inplace=True)
                dropped_rows = initial_rows - len(processed_df)
                if dropped_rows > 0:
                    logging.info(
                        f"Dropped {dropped_rows} rows with missing values in '{col}'."
                    )
            else:
                logging.warning(
                    f"Unsupported missing value strategy '{strategy}' for column '{col}'."
                )
        else:
            logging.debug(f"No missing values found in '{col}'.")
    return processed_df


def _create_interaction_feature(processed_df: pd.DataFrame, step: Dict) -> pd.DataFrame:
    """Creates an interaction feature."""
    new_col_name = step.get("new_col")
    cols_to_interact = step["cols"]
    if not new_col_name:
        logging.warning(f"Feature engineering step missing 'new_col' name: {step}")
        return processed_df

    if all(col in processed_df.columns for col in cols_to_interact):
        numeric_cols_to_interact = processed_df[cols_to_interact].select_dtypes(
            include=np.number
        )
        if len(numeric_cols_to_interact.columns) == len(cols_to_interact):
            processed_df[new_col_name] = numeric_cols_to_interact.product(axis=1)
            logging.info(
                f"Created interaction feature '{new_col_name}' from {cols_to_interact}."
            )
        else:
            logging.warning(
                f"Not all columns for interaction feature '{new_col_name}' are numeric: {cols_to_interact}"
            )
    else:
        missing = [col for col in cols_to_interact if col not in processed_df.columns]
        logging.warning(
            f"One or more columns for interaction feature '{new_col_name}' not found: {missing}"
        )
    return processed_df


def _create_polynomial_feature(processed_df: pd.DataFrame, step: Dict) -> pd.DataFrame:
    """Creates a polynomial feature."""
    new_col_name = step.get("new_col")
    col_to_transform = step["col"]
    degree = step["degree"]
    if not new_col_name:
        logging.warning(f"Feature engineering step missing 'new_col' name: {step}")
        return processed_df

    if col_to_transform in processed_df.columns:
        if pd.api.types.is_numeric_dtype(processed_df[col_to_transform]):
            processed_df[new_col_name] = processed_df[col_to_transform] ** degree
            logging.info(
                f"Created polynomial feature '{new_col_name}' (degree {degree}) from '{col_to_transform}'."
            )
        else:
            logging.warning(
                f"Column '{col_to_transform}' is not numeric for polynomial feature '{new_col_name}'."
            )
    else:
        logging.warning(
            f"Column '{col_to_transform}' for polynomial feature '{new_col_name}' not found."
        )
    return processed_df


def _create_log_transform_feature(
    processed_df: pd.DataFrame, step: Dict
) -> pd.DataFrame:
    """Creates a log-transformed feature."""
    new_col_name = step.get("new_col")
    col_to_transform = step["col"]
    if not new_col_name:
        logging.warning(f"Feature engineering step missing 'new_col' name: {step}")
        return processed_df

    if col_to_transform in processed_df.columns:
        if pd.api.types.is_numeric_dtype(processed_df[col_to_transform]):
            # Add a small constant to avoid log(0) or log(negative)
            # Ensure values are non-negative before log
            values_to_log = (
                processed_df[col_to_transform].apply(lambda x: max(x, 0)) + 1e-9
            )
            processed_df[new_col_name] = np.log(values_to_log)
            logging.info(
                f"Created log-transformed feature '{new_col_name}' from '{col_to_transform}'."
            )
        else:
            logging.warning(
                f"Column '{col_to_transform}' is not numeric for log transform feature '{new_col_name}'."
            )
    else:
        logging.warning(
            f"Column '{col_to_transform}' for log transform feature '{new_col_name}' not found."
        )
    return processed_df


def _perform_single_feature_engineering_step(
    processed_df: pd.DataFrame, step: Dict
) -> pd.DataFrame:
    """Processes a single feature engineering step."""
    step_type = step.get("type")

    if step_type == "interaction" and "cols" in step:
        return _create_interaction_feature(processed_df, step)
    elif step_type == "polynomial" and "col" in step and "degree" in step:
        return _create_polynomial_feature(processed_df, step)
    elif step_type == "log_transform" and "col" in step:
        return _create_log_transform_feature(processed_df, step)
    else:
        logging.warning(
            f"Unsupported feature engineering step type or missing parameters: {step}"
        )
        return processed_df


# Helper function to encapsulate the logic for functools.reduce,
# addressing the complexity error in _perform_feature_engineering.
def _apply_fe_step(current_df: pd.DataFrame, step: Dict) -> pd.DataFrame:
    """Helper to apply a single feature engineering step."""
    return _perform_single_feature_engineering_step(current_df, step)


# FIX: Complexity error (by extracting lambda into _apply_fe_step) and improved type hinting
def _perform_feature_engineering(
    df: pd.DataFrame, fe_steps: List[Dict]
) -> pd.DataFrame:
    """
    Performs feature engineering based on the provided steps.
    `fe_steps` is a list of dictionaries, each describing a feature engineering operation.
    """
    processed_df = df.copy()

    # Use functools.reduce to apply each step sequentially, passing the updated DataFrame.
    # Using the named helper function `_apply_fe_step` reduces the complexity of this function
    # by avoiding a lambda expression directly within functools.reduce.
    processed_df = functools.reduce(
        _apply_fe_step,
        fe_steps,
        processed_df,  # Initial DataFrame
    )

    return processed_df


# FIX 4: Improved type hinting
def _apply_scaling(
    df: pd.DataFrame, numerical_cols: List[str], normalize: bool, standardize: bool
) -> pd.DataFrame:
    """
    Applies normalization or standardization to numerical columns.
    `numerical_cols` is a list of column names (str) to be scaled.
    """
    processed_df = df.copy()
    for col in numerical_cols:
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(
            processed_df[col]
        ):
            if normalize:
                min_val = processed_df[col].min()
                max_val = processed_df[col].max()
                if max_val - min_val != 0:
                    processed_df[col] = (processed_df[col] - min_val) / (
                        max_val - min_val
                    )
                    logging.info(f"Normalized column '{col}'.")
                else:
                    logging.debug(
                        f"Column '{col}' has no variance, skipping normalization."
                    )
            elif standardize:
                mean_val = processed_df[col].mean()
                std_val = processed_df[col].std()
                if std_val != 0:
                    processed_df[col] = (processed_df[col] - mean_val) / std_val
                    logging.info(f"Standardized column '{col}'.")
                else:
                    logging.debug(
                        f"Column '{col}' has no variance, skipping standardization."
                    )
        else:
            logging.warning(f"Column '{col}' not found or not numeric for scaling.")
    return processed_df


# FIX 4: Improved type hinting
def _encode_categorical_features(
    df: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Applies one-hot encoding to categorical columns.
    `categorical_cols` is a list of column names (str) to be encoded.
    """
    processed_df = df.copy()
    for col in categorical_cols:
        if col in processed_df.columns:
            try:
                # Convert column to string type to ensure get_dummies handles it correctly
                processed_df[col] = processed_df[col].astype(str)
                processed_df = pd.get_dummies(
                    processed_df, columns=[col], prefix=col, dummy_na=False
                )
                logging.info(f"One-hot encoded categorical column '{col}'.")
            except Exception as e:
                logging.error(f"Error during one-hot encoding for column '{col}': {e}")
        else:
            logging.warning(f"Categorical column '{col}' not found for encoding.")
    return processed_df


def _handle_missing_values_step(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Step for handling missing values."""
    strategy_config = config.get("missing_value_strategy", {})
    if not strategy_config:
        logging.debug("No missing value strategy configured.")
        return df
    # FIX 4: Ensure correct type hint is passed to _handle_missing_values
    return _handle_missing_values(df, strategy_config)


def _feature_engineering_step(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Step for performing feature engineering."""
    fe_steps = config.get("feature_engineering", [])
    if not fe_steps:
        logging.debug("No feature engineering steps configured.")
        return df
    # FIX 4: Ensure correct type hint is passed to _perform_feature_engineering
    return _perform_feature_engineering(df, fe_steps)


def _scaling_step(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Step for applying scaling (normalization/standardization)."""
    normalize_numerical = config.get("normalize_numerical", False)
    standardize_numerical = config.get("standardize_numerical", False)
    numerical_cols_orig = config.get("numerical_cols", [])

    if not normalize_numerical and not standardize_numerical:
        logging.debug("Scaling (normalize/standardize) is not enabled.")
        return df

    # Determine current numerical columns, including those potentially created by feature engineering
    current_numerical_cols = []
    for col in numerical_cols_orig:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            current_numerical_cols.append(col)

    for step in config.get("feature_engineering", []):
        if (
            step.get("type") in ["interaction", "polynomial", "log_transform"]
            and step.get("new_col") in df.columns
        ):
            if pd.api.types.is_numeric_dtype(df[step["new_col"]]):
                current_numerical_cols.append(step["new_col"])

    current_numerical_cols = list(set(current_numerical_cols))  # Ensure unique

    if not current_numerical_cols:
        logging.warning("No suitable numerical columns found for scaling.")
        return df

    # FIX 4: Ensure correct type hint is passed to _apply_scaling
    return _apply_scaling(
        df,
        current_numerical_cols,
        normalize_numerical,
        standardize_numerical,
    )


def _categorical_encoding_step(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Step for encoding categorical features."""
    categorical_cols = config.get("categorical_cols", [])
    if not categorical_cols:
        logging.debug("No categorical columns specified for encoding.")
        return df
    # FIX 4: Ensure correct type hint is passed to _encode_categorical_features
    return _encode_categorical_features(df, categorical_cols)


def preprocess_data(df: pd.DataFrame, preprocessing_config: Dict) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame based on the provided configuration.

    This function orchestrates multiple preprocessing steps.

    Args:
        df (pd.DataFrame): The raw data.
        preprocessing_config (dict): Configuration for preprocessing steps.
            Example:
            {
                'numerical_cols': ['col1', 'col2'],
                'categorical_cols': ['col3'],
                'missing_value_strategy': {'col1': 'mean', 'col3': 'mode'},
                'normalize_numerical': True,
                'standardize_numerical': False,
                'feature_engineering': [
                    {'type': 'interaction', 'cols': ['col1', 'col2'], 'new_col': 'col1_x_col2'},
                    {'type': 'polynomial', 'col': 'col1', 'degree': 2, 'new_col': 'col1_sq'}
                ]
            }

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty. Skipping preprocessing.")
        return df

    processed_df = df.copy()

    # Execute steps in a logical order
    processed_df = _handle_missing_values_step(processed_df, preprocessing_config)
    processed_df = _feature_engineering_step(processed_df, preprocessing_config)
    processed_df = _scaling_step(processed_df, preprocessing_config)
    processed_df = _categorical_encoding_step(processed_df, preprocessing_config)

    logging.info(f"Data preprocessing completed. Shape: {processed_df.shape}")
    return processed_df


# --- Pattern Detection ---


class PatternDetector:
    """
    Abstract base class for pattern detection algorithms.
    """

    def __init__(self, params: Dict = None):
        self.params = params if params is not None else {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects patterns in the provided DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement 'detect'")


class AnomalyDetector(PatternDetector):
    """
    Detects anomalies in the data using Isolation Forest.
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.n_estimators = self.params.get("n_estimators", 100)
        self.contamination = self.params.get("contamination", "auto")
        self.max_features = self.params.get("max_features", 1.0)
        self.random_state = self.params.get("random_state", 42)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logging.warning("Input DataFrame is empty. Cannot detect anomalies.")
            return pd.DataFrame({"is_anomaly": [], "anomaly_score": []})

        numerical_df = df.select_dtypes(include=np.number)
        if numerical_df.empty:
            logging.warning("No numerical columns found for anomaly detection.")
            return pd.DataFrame({"is_anomaly": [], "anomaly_score": []})

        # If there are infinite values, IsolationForest can raise an error.
        # Replace infinities with NaNs, then handle NaNs (e.g., fill with mean or drop)
        numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan)
        if numerical_df.isnull().values.any():
            logging.warning(
                "NaNs detected in numerical data for anomaly detection. Filling with mean."
            )
            numerical_df = numerical_df.fillna(numerical_df.mean())
            # If after filling, NaNs remain (e.g., all values were NaN), drop columns
            numerical_df = numerical_df.dropna(axis=1, how="all")
            if numerical_df.empty:
                logging.error(
                    "Numerical data became empty after handling NaNs for anomaly detection."
                )
                return pd.DataFrame({"is_anomaly": [], "anomaly_score": []})

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )

        try:
            model.fit(numerical_df)
            predictions = model.predict(numerical_df)
            scores = model.decision_function(numerical_df)

            result_df = pd.DataFrame(
                {"is_anomaly": predictions, "anomaly_score": scores}, index=df.index
            )

            logging.info(
                f"Anomaly detection completed. Found {sum(predictions == -1)} potential anomalies."
            )
            return result_df
        except Exception as e:
            logging.error(f"Error during anomaly detection: {e}")
            return pd.DataFrame({"is_anomaly": [], "anomaly_score": []})


class CorrelationFinder(PatternDetector):
    """
    Finds strongly correlated or anti-correlated features.
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.correlation_threshold = self.params.get("correlation_threshold", 0.8)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logging.warning("Input DataFrame is empty. Cannot find correlations.")
            return pd.DataFrame(columns=["feature1", "feature2", "correlation", "type"])

        numerical_df = df.select_dtypes(include=np.number)
        if numerical_df.shape[1] < 2:
            logging.warning("Need at least two numerical columns to find correlations.")
            return pd.DataFrame(columns=["feature1", "feature2", "correlation", "type"])

        correlation_matrix = numerical_df.corr()
        correlated_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]

                if abs(correlation) >= self.correlation_threshold:
                    correlated_pairs.append(
                        {
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": correlation,
                            "type": "positive" if correlation > 0 else "negative",
                        }
                    )

        result_df = pd.DataFrame(correlated_pairs)
        logging.info(
            f"Correlation finding completed. Found {len(result_df)} pairs exceeding threshold {self.correlation_threshold}."
        )
        return result_df


class PredictiveTrendDetector(PatternDetector):
    """
    Detects emerging trends using simple linear regression on recent data.
    Fixes: Introduce `time_column` for robust chronological ordering.
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.window_size = self.params.get("window_size", 10)
        self.trend_threshold = self.params.get("trend_threshold", 0.05)
        # FIX 2: Add time_column parameter in __init__
        self.time_column = self.params.get(
            "time_column"
        )  # Column to use for chronological sorting

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logging.warning("Input DataFrame is empty. Cannot detect trends.")
            return pd.DataFrame(columns=["column", "trend_type", "predicted_value"])

        numerical_cols = df.select_dtypes(include=np.number).columns
        if len(numerical_cols) == 0:
            logging.warning("No numerical columns found for trend detection.")
            return pd.DataFrame(columns=["column", "trend_type", "predicted_value"])

        trends = []

        # FIX 2: Implement robust chronological ordering
        sorted_df = (
            df  # Default to original order if sorting logic fails or is not applicable
        )
        if self.time_column:
            if self.time_column not in df.columns:
                logging.error(
                    f"Time column '{self.time_column}' not found for trend detection. Cannot establish chronological order. Skipping detection for this configuration."
                )
                return pd.DataFrame(columns=["column", "trend_type", "predicted_value"])
            try:
                # Attempt to convert to datetime if not already for proper sorting, handling errors
                if not pd.api.types.is_datetime64_any_dtype(df[self.time_column]):
                    temp_time_col = pd.to_datetime(
                        df[self.time_column], errors="coerce"
                    )
                    if temp_time_col.isnull().all():
                        logging.warning(
                            f"Time column '{self.time_column}' could not be converted to datetime. Sorting by values as-is. Ensure it represents chronological order."
                        )
                        sorted_df = df.sort_values(by=self.time_column)
                    else:
                        sorted_df = df.copy()
                        sorted_df[self.time_column] = temp_time_col
                        sorted_df = sorted_df.sort_values(by=self.time_column)
                else:
                    sorted_df = df.sort_values(by=self.time_column)

                logging.debug(
                    f"DataFrame sorted by specified time column '{self.time_column}'."
                )
            except Exception as e:
                logging.error(
                    f"Error sorting by time column '{self.time_column}': {e}. Proceeding with original row order. Trend results may be unreliable without a clear chronological dimension."
                )
        else:
            # If no time_column is specified, rely on the DataFrame index
            try:
                sorted_df = df.sort_index()
                if not pd.api.types.is_datetime64_any_dtype(sorted_df.index):
                    logging.warning(
                        "DataFrame index is not a DatetimeIndex. Assuming index represents chronological order; for robust trend analysis, consider setting a 'time_column' configuration parameter or ensuring a proper DatetimeIndex."
                    )
                logging.debug("DataFrame sorted by index for trend detection.")
            except TypeError:
                logging.warning(
                    "DataFrame index is not directly sortable or not time-based. Assuming original row order for trend detection; results may be unreliable without a proper time dimension or 'time_column' configuration."
                )
                sorted_df = df  # Fallback if sorting by index fails

        for col in numerical_cols:
            if len(sorted_df[col]) < self.window_size:
                logging.debug(
                    f"Not enough data for trend detection in '{col}' (need {self.window_size}, have {len(sorted_df[col])})."
                )
                continue

            # Drop NaNs from the current column's recent_data to prevent polyfit errors
            recent_data = sorted_df[col].tail(self.window_size).dropna()

            if len(recent_data) < 2:  # Need at least 2 points for a linear fit
                logging.debug(
                    f"Not enough valid data points in recent window for trend detection in '{col}' after dropping NaNs."
                )
                continue

            x = np.arange(len(recent_data))
            y = recent_data.values

            if np.all(y == y[0]):  # All values are the same, no trend
                continue

            try:
                m, c = np.polyfit(x, y, 1)

                next_x = len(recent_data)  # Predict for the next point in the sequence
                predicted_value = m * next_x + c

                last_value = recent_data.iloc[-1]
                # Avoid division by zero for relative change calculation
                if last_value != 0:
                    relative_change = (predicted_value - last_value) / last_value
                elif predicted_value != 0:  # last_value is 0, but predicted is not
                    relative_change = (
                        np.inf
                    )  # Indicates significant positive change from zero
                else:  # both are zero
                    relative_change = 0

                if abs(relative_change) >= self.trend_threshold:
                    trend_type = "upward" if m > 0 else "downward"
                    trends.append(
                        {
                            "column": col,
                            "trend_type": trend_type,
                            "predicted_value": predicted_value,
                            # Using slope as a proxy for confidence/magnitude of trend
                            # A higher absolute slope indicates a stronger, more defined trend.
                            "confidence": abs(m),
                        }
                    )
            except Exception as e:
                logging.error(f"Error detecting trend for column '{col}': {e}")

        result_df = pd.DataFrame(trends)
        logging.info(
            f"Predictive trend detection completed. Found {len(result_df)} potential trends."
        )
        return result_df


class PatternDetectorFactory:
    """
    Factory to create PatternDetector instances.
    """

    def __init__(self):
        self.detectors = {
            "anomaly": AnomalyDetector,
            "correlation": CorrelationFinder,
            "predictive_trend": PredictiveTrendDetector,
        }

    def get_detector(self, detector_type: str, params: Dict = None) -> PatternDetector:
        """
        Gets an instance of a PatternDetector.

        Args:
            detector_type (str): The type of detector (e.g., 'anomaly', 'correlation').
            params (dict, optional): Parameters for the detector. Defaults to None.

        Returns:
            PatternDetector: An instance of the requested detector.

        Raises:
            ValueError: If the detector type is not supported.
        """
        detector_class = self.detectors.get(detector_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported pattern detector type: {detector_type}")
        return detector_class(params)


# --- Opportunity Generation ---


class Opportunity:
    """
    Represents a discovered opportunity for value creation.
    """

    def __init__(
        self,
        title: str,
        description: str,
        potential_value_type: str,
        source_data_features: List[str] = None,
        confidence_score: float = 0.5,
        recommended_actions: List[str] = None,
        evidence: Dict = None,
    ):
        self.title = title
        self.description = description
        self.potential_value_type = potential_value_type
        self.source_data_features = source_data_features if source_data_features else []
        self.confidence_score = max(
            0.0, min(1.0, confidence_score)
        )  # Ensure score is between 0 and 1
        self.recommended_actions = recommended_actions if recommended_actions else []
        self.evidence = evidence if evidence else {}

    def to_dict(self) -> Dict:
        """Converts the Opportunity object to a dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "potential_value_type": self.potential_value_type,
            "source_data_features": self.source_data_features,
            "confidence_score": self.confidence_score,
            "recommended_actions": self.recommended_actions,
            "evidence": self.evidence,
        }

    def __repr__(self):
        return f"Opportunity(title='{self.title}', value_type='{self.potential_value_type}', score={self.confidence_score:.2f})"


class OpportunityFormatter:
    """
    Translates detected patterns into structured Opportunity objects.
    Fixes: Added clarity of confidence heuristics and robust NaN handling.
    """

    def __init__(self, opportunity_config: Dict = None):
        self.opportunity_config = (
            opportunity_config if opportunity_config is not None else {}
        )
        self.value_type_mapping = {
            "anomaly": "efficiency",
            "correlation": "knowledge",
            "predictive_trend": "economic",
        }

    # FIX 3 & 4: Add type hint for anomalies_df and comments for clarity on confidence heuristics
    def _format_anomalies(self, anomalies_df: pd.DataFrame) -> List[Opportunity]:
        """
        Formats anomaly detection results into opportunities.

        Confidence Heuristics (for `confidence_score`):
        - `confidence_base`: Proportional to the number of significant anomalies found relative to a configured threshold (`anomaly_threshold_count`).
                             A higher number of anomalies (up to the threshold) suggests a more widespread or impactful issue.
        - `scaled_score`: Reflects the average 'outlierness' score of the significant anomalies relative to a reference (min_score or max_score).
                          Higher average scores indicate stronger deviations from the norm.
                          Handles cases where `anomaly_score` might be all NaNs or non-numeric.
        - The final score is a sum of these two components, capped at 1.0 to stay within a 0-1 range.
        """
        opportunities = []
        anomaly_indices = anomalies_df[anomalies_df["is_anomaly"] == -1].index

        if not anomaly_indices.empty:
            max_anomalies = self.opportunity_config.get(
                "max_anomalies_per_opportunity", 5
            )
            # Ensure max_anomalies does not exceed the number of actual anomalies
            max_anomalies = min(max_anomalies, len(anomaly_indices))

            # Select the most significant anomalies based on score
            significant_anomalies = anomalies_df.loc[anomaly_indices].nlargest(
                max_anomalies,
                "anomaly_score",
                keep="first",  # 'keep' ensures deterministic selection
            )

            # Identify features that were used for detection (simplification).
            # In a real scenario, we'd want to map anomaly scores back to features more precisely.
            involved_features = [
                col
                for col in significant_anomalies.columns
                if col not in ["is_anomaly", "anomaly_score"]
            ]

            # Confidence calculation:
            # 1. `confidence_base` from the number of significant anomalies
            anomaly_threshold_count = self.opportunity_config.get(
                "anomaly_threshold_count", 10.0
            )
            confidence_base = len(significant_anomalies) / anomaly_threshold_count

            # 2. `scaled_score` from the average anomaly score
            scaled_score = 0.0
            if (
                not significant_anomalies["anomaly_score"].empty
                and significant_anomalies["anomaly_score"].notna().any()
            ):
                mean_sig_score = significant_anomalies["anomaly_score"].mean()
                all_min_score = anomalies_df["anomaly_score"].min()
                all_max_score = anomalies_df["anomaly_score"].max()

                if (
                    not pd.isna(all_min_score)
                    and not pd.isna(all_max_score)
                    and (all_max_score - all_min_score) > 0
                ):
                    # Normalize mean significant score within the range of all scores
                    scaled_score = (mean_sig_score - all_min_score) / (
                        all_max_score - all_min_score
                    )
                elif not pd.isna(mean_sig_score):
                    # Fallback if range is zero or invalid, use mean relative to its absolute value
                    scaled_score = abs(mean_sig_score) / (
                        abs(mean_sig_score) if mean_sig_score != 0 else 1.0
                    )
                else:
                    scaled_score = 0.0  # No valid scores

            # Ensure scaled_score is non-negative and capped
            scaled_score = max(0.0, min(scaled_score, 1.0))

            # Final confidence score combines these, capped at 1.0
            confidence_score = min(1.0, confidence_base + scaled_score)

            title = "Potential System Anomaly or Novel Event Detected"
            description = (
                f"Detected {len(significant_anomalies)} significant anomaly points. "
                "This may indicate unexpected behavior, inefficiencies, or emerging novel states. "
                "Investigate the highlighted data points and their context."
            )
            value_type = self.value_type_mapping.get("anomaly", "unknown")
            actions = [
                "Review logs and system behavior around anomalous data points.",
                "Analyze the features contributing to these anomalies for root causes.",
                "Consider if anomalies represent new opportunities or risks.",
            ]
            # Convert evidence to a more digestible format (e.g., dict(orient="index"))
            evidence = significant_anomalies.to_dict(orient="index")

            opportunities.append(
                Opportunity(
                    title=title,
                    description=description,
                    potential_value_type=value_type,
                    source_data_features=involved_features,
                    confidence_score=confidence_score,
                    recommended_actions=actions,
                    evidence=evidence,
                )
            )
        return opportunities

    # FIX 4: Add type hint for correlations_df
    def _format_correlations(self, correlations_df: pd.DataFrame) -> List[Opportunity]:
        """
        Formats correlation detection results into opportunities.

        Confidence Heuristics (for `confidence_score`):
        - `confidence`: Directly based on the absolute value of the correlation coefficient.
                        A correlation closer to 1 (positive or negative) indicates higher confidence in the relationship.
        """
        opportunities = []
        if not correlations_df.empty:
            # Group by feature pairs to avoid redundant opportunities if multiple thresholds were met
            # Using tuple to ensure order-agnostic grouping (e.g., (A,B) is same as (B,A) for correlation)
            correlations_df["sorted_features"] = correlations_df.apply(
                lambda row: tuple(sorted([row["feature1"], row["feature2"]])), axis=1
            )
            grouped_correlations = correlations_df.groupby("sorted_features")

            for _, group in grouped_correlations:
                # Take the entry with the strongest absolute correlation for this pair
                strongest_corr = group.loc[group["correlation"].abs().idxmax()]

                f1, f2 = strongest_corr["feature1"], strongest_corr["feature2"]
                title = f"Strong {strongest_corr['type']} Correlation Between '{f1}' and '{f2}'"
                description = (
                    f"A strong {strongest_corr['type']} correlation (coefficient: {strongest_corr['correlation']:.2f}) "
                    f"was found between '{f1}' and '{f2}'. "
                    "This suggests a deep relationship that could be leveraged for insights or optimization."
                )
                value_type = self.value_type_mapping.get("correlation", "knowledge")
                # Confidence based on absolute correlation value, directly representing strength.
                confidence = abs(strongest_corr["correlation"])
                actions = [
                    f"Investigate the underlying mechanism driving the correlation between '{f1}' and '{f2}'.",
                    f"Consider if '{f1}' can predict or influence '{f2}', or vice versa.",
                    "Explore opportunities for product bundling, cross-promotion, or process integration based on this relationship.",
                ]
                evidence = strongest_corr.drop(
                    "sorted_features"
                ).to_dict()  # Exclude helper column

                opportunities.append(
                    Opportunity(
                        title=title,
                        description=description,
                        potential_value_type=value_type,
                        source_data_features=[f1, f2],
                        confidence_score=confidence,
                        recommended_actions=actions,
                        evidence=evidence,
                    )
                )
        return opportunities

    # FIX 3 & 4: Add type hint for trends_df and comments for clarity on confidence heuristics
    def _format_trends(self, trends_df: pd.DataFrame) -> List[Opportunity]:
        """
        Formats trend detection results into opportunities.

        Confidence Heuristics (for `confidence_score`):
        - `confidence`: Derived from the absolute slope (`m`) of the linear regression (`row.get("confidence", 0)` in `trends_df`).
                        This slope is scaled by a configurable maximum (`trend_confidence_max`) to normalize it to a 0-1 range.
                        A steeper slope indicates a stronger, more confident trend.
                        The score is capped at 1.0.
        """
        opportunities = []
        if not trends_df.empty:
            for _, row in trends_df.iterrows():
                trend_type_str = row["trend_type"]
                title = (
                    f"Emerging Upward Trend Detected in '{row['column']}'"
                    if trend_type_str == "upward"
                    else f"Potential Downward Trend in '{row['column']}'"
                )
                description = (
                    f"A notable {trend_type_str} trend has been identified in '{row['column']}'. "
                    f"The predicted next value is {row['predicted_value']:.2f}. "
                    f"This suggests a potential emerging market, resource shift, or efficiency change."
                )
                value_type = self.value_type_mapping.get("predictive_trend", "economic")

                # Confidence scaling: based on the absolute magnitude of the trend's slope ('confidence' in trends_df),
                # scaled by a configurable maximum ('trend_confidence_max').
                # Ensures the score is normalized to [0, 1].
                max_trend_confidence = self.opportunity_config.get(
                    "trend_confidence_max", 5.0
                )

                # Ensure confidence value from detector is non-negative before scaling
                detector_confidence = max(0.0, row.get("confidence", 0.0))
                confidence = min(1.0, detector_confidence / max_trend_confidence)

                actions = [
                    f"Monitor the trend in '{row['column']}' closely.",
                    "If upward: Explore opportunities to capitalize on this growth (e.g., market entry, resource allocation).",
                    "If downward: Consider mitigation strategies or identify the cause.",
                    "Analyze factors potentially driving this trend.",
                ]
                evidence = row.to_dict()

                opportunities.append(
                    Opportunity(
                        title=title,
                        description=description,
                        potential_value_type=value_type,
                        source_data_features=[row["column"]],
                        confidence_score=confidence,
                        recommended_actions=actions,
                        evidence=evidence,
                    )
                )
        return opportunities

    def format_opportunities(
        self, detected_patterns: Dict[str, pd.DataFrame]
    ) -> List[Opportunity]:
        """
        Generates a list of Opportunity objects from detected patterns.

        Args:
            detected_patterns (dict): A dictionary where keys are pattern types (e.g., 'anomalies', 'correlations')
                                      and values are DataFrames of detected patterns.

        Returns:
            list[Opportunity]: A list of formatted Opportunity objects.
        """
        all_opportunities = []

        if "anomalies" in detected_patterns:
            all_opportunities.extend(
                self._format_anomalies(detected_patterns["anomalies"])
            )
        if "correlations" in detected_patterns:
            all_opportunities.extend(
                self._format_correlations(detected_patterns["correlations"])
            )
        if "trends" in detected_patterns:
            all_opportunities.extend(self._format_trends(detected_patterns["trends"]))

        logging.info(
            f"Generated {len(all_opportunities)} opportunities from detected patterns."
        )
        return all_opportunities


# --- Abundance Engine ---


class AbundanceEngine:
    """
    The core Abundance Engine that orchestrates data loading, analysis,
    and opportunity generation.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the AbundanceEngine.

        Args:
            config_path (str): Path to the main configuration file.
        """
        self.config_path = config_path
        try:
            self.config = load_config(self.config_path)
            self.data_config = self.config.get("data_source", {})
            self.preprocessing_config = self.config.get("preprocessing", {})
            self.pattern_detection_config = self.config.get("pattern_detection", {})
            self.opportunity_config = self.config.get("opportunity_generation", {})

            self.data_source = get_data_source(self.data_config)
            self.preprocessed_data = None
            self.detected_patterns = {}
            self.opportunities = []

            self.pattern_detector_factory = PatternDetectorFactory()
            self.opportunity_formatter = OpportunityFormatter(self.opportunity_config)

            logging.info("AbundanceEngine initialized successfully.")

        except FileNotFoundError as e:
            logging.error(f"Configuration error: {e}")
            raise
        except ValueError as e:
            logging.error(f"Configuration or setup error: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during initialization: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the configured data source.

        Returns:
            pd.DataFrame: The loaded raw data.

        Raises:
            Exception: If data loading fails.
        """
        logging.info("Loading data...")
        try:
            self.preprocessed_data = self.data_source.fetch_data()
            logging.info(
                f"Data loaded successfully. Shape: {self.preprocessed_data.shape}"
            )
            return self.preprocessed_data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        """
        Applies preprocessing steps to the loaded data.

        Returns:
            pd.DataFrame: The preprocessed data.

        Raises:
            ValueError: If no data has been loaded yet.
            Exception: If preprocessing fails.
        """
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            raise ValueError("No data loaded. Call 'load_data()' first.")

        logging.info("Preprocessing data...")
        try:
            self.preprocessed_data = preprocess_data(
                self.preprocessed_data, self.preprocessing_config
            )
            logging.info("Data preprocessing completed.")
            return self.preprocessed_data
        except Exception as e:
            logging.error(f"Failed during data preprocessing: {e}")
            raise

    def analyze_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        Analyzes the preprocessed data to detect various patterns.

        Returns:
            dict: A dictionary containing DataFrames of detected patterns, keyed by pattern type.

        Raises:
            ValueError: If no data has been preprocessed yet.
            Exception: If pattern detection fails.
        """
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            raise ValueError("No data preprocessed. Call 'preprocess_data()' first.")

        logging.info("Analyzing data for patterns...")
        self.detected_patterns = {}

        detector_configs = self.pattern_detection_config.get("detectors", [])
        if not detector_configs:
            logging.warning("No pattern detectors configured. Skipping analysis.")
            return {}

        for detector_info in detector_configs:
            detector_type = detector_info.get("type")
            detector_params = detector_info.get("params", {})

            if not detector_type:
                logging.warning(
                    f"Skipping detector due to missing 'type': {detector_info}"
                )
                continue

            try:
                detector = self.pattern_detector_factory.get_detector(
                    detector_type, detector_params
                )
                logging.info(
                    f"Running detector: {detector_type} with params {detector_params}"
                )
                pattern_results = detector.detect(self.preprocessed_data)

                if not pattern_results.empty:
                    # Map detector types to consistent keys in detected_patterns
                    if detector_type.lower() == "anomaly":
                        self.detected_patterns["anomalies"] = pattern_results
                    elif detector_type.lower() == "correlation":
                        self.detected_patterns["correlations"] = pattern_results
                    elif detector_type.lower() == "predictive_trend":
                        self.detected_patterns["trends"] = pattern_results
                    else:
                        self.detected_patterns[detector_type.lower()] = pattern_results
                    logging.info(
                        f"Detected {len(pattern_results)} instances for pattern type '{detector_type}'."
                    )
                else:
                    logging.debug(f"No patterns detected by '{detector_type}'.")

            except ValueError as e:
                logging.error(
                    f"Configuration error for detector '{detector_type}': {e}"
                )
            except Exception as e:
                logging.error(
                    f"Error during pattern detection for '{detector_type}': {e}"
                )

        logging.info(
            f"Pattern analysis completed. Found {len(self.detected_patterns)} types of patterns."
        )
        return self.detected_patterns

    def generate_opportunities(self) -> List[Opportunity]:
        """
        Translates detected patterns into actionable opportunities.

        Returns:
            list[Opportunity]: A list of generated Opportunity objects.

        Raises:
            Exception: If opportunity generation fails.
        """
        logging.info("Generating opportunities...")
        try:
            self.opportunities = self.opportunity_formatter.format_opportunities(
                self.detected_patterns
            )
            logging.info(
                f"Opportunity generation completed. Generated {len(self.opportunities)} opportunities."
            )
            return self.opportunities
        except Exception as e:
            logging.error(f"Failed during opportunity generation: {e}")
            raise

    def get_opportunities(self) -> List[Opportunity]:
        """
        Returns the list of generated opportunities.
        """
        return self.opportunities

    def run(self):
        """
        Executes the full Abundance Engine pipeline: load, preprocess, analyze, generate.
        """
        logging.info("Starting Abundance Engine run...")
        try:
            self.load_data()
            self.preprocess_data()
            self.analyze_patterns()
            self.generate_opportunities()
            logging.info("Abundance Engine run completed successfully.")
        except Exception as e:
            logging.error(f"Abundance Engine run failed: {e}", exc_info=True)
            sys.exit(1)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Abundance Engine application.")

    DEFAULT_CONFIG_PATH = "config/config.yaml"
    config_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    # --- Create dummy config file and data file for initial testing if they don't exist ---
    # This part is for making the script runnable out-of-the-box without external setup.
    # In a real project, these would be managed by build/deployment processes.

    # Ensure config directory exists
    if not os.path.exists("config"):
        os.makedirs("config")
        logging.info("Created 'config' directory.")

    # Create a dummy config.yaml if it doesn't exist
    if not os.path.exists(config_file):
        logging.warning(f"'{config_file}' not found. Creating a dummy configuration.")
        dummy_config_content = """
data_source:
  type: mock
  mock_data_config:
    num_rows: 200
    num_cols: 5
    data_range: [0, 100]

preprocessing:
  numerical_cols: ["feature_1", "feature_2", "feature_3"]
  categorical_cols: []
  missing_value_strategy: {} # Example: {"feature_1": "mean"}
  normalize_numerical: false
  standardize_numerical: true # Standardize for better anomaly detection
  feature_engineering:
    - type: interaction
      cols: ["feature_1", "feature_2"]
      new_col: "feature1_x_feature2"
    - type: polynomial
      col: "feature_3"
      degree: 2
      new_col: "feature3_squared"

pattern_detection:
  detectors:
    - type: anomaly
      params:
        contamination: 0.05
        random_state: 42
    - type: correlation
      params:
        correlation_threshold: 0.8
    - type: predictive_trend
      params:
        window_size: 15
        trend_threshold: 0.03
        # FIX 2: Added a 'time_column' example for PredictiveTrendDetector
        # If your data has a datetime column, specify it here.
        # Example: time_column: 'timestamp_col'
        # If omitted, it will try to sort by DataFrame index.
        time_column: # No explicit time column in mock data, will use index.

opportunity_generation:
  max_anomalies_per_opportunity: 3
  anomaly_threshold_count: 10 # Used in confidence scoring: how many anomalies are 'a lot'
  trend_confidence_max: 5.0 # Used in confidence scoring: what's a 'max' slope for trends
"""
        try:
            with open(config_file, "w") as f:
                f.write(dummy_config_content)
            logging.info(f"Dummy configuration file created at '{config_file}'.")
        except IOError as e:
            logging.error(f"Failed to create dummy config file: {e}")
            sys.exit(1)

    # --- End of dummy file creation ---

    try:
        engine = AbundanceEngine(config_path=config_file)
        engine.run()

        print("\n--- Generated Opportunities ---")
        if engine.get_opportunities():
            for i, opp in enumerate(engine.get_opportunities()):
                print(f"\nOpportunity {i + 1}:")
                print(f"  Title: {opp.title}")
                print(f"  Type: {opp.potential_value_type}")
                print(f"  Confidence: {opp.confidence_score:.2f}")
                print(f"  Description: {opp.description}")
                print(f"  Features: {', '.join(opp.source_data_features)}")
                print(f"  Recommended Actions: {'; '.join(opp.recommended_actions)}")
        else:
            print("No opportunities were generated.")

    except FileNotFoundError:
        logging.error(
            f"Error: Configuration file not found at '{config_file}'. Please ensure the path is correct."
        )
        sys.exit(1)
    except (yaml.YAMLError, ValueError) as e:
        logging.error(f"Configuration or initialization error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during application execution: {e}",
            exc_info=True,
        )
        sys.exit(1)

    logging.info("Abundance Engine application finished.")
