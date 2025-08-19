import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from io import BytesIO
import json
import joblib
import pickle

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from minio import Minio
from minio.error import S3Error

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec
import tempfile
import os

# MinIO Configuration
MINIO_CONFIG = {
    'endpoint': 'minio:9000',
    'access_key': 'admin',
    'secret_key': 'admin123',
    'secure': False
}

BUCKET_NAME = 'crypto-models'

class MLflowModelRegistry:
    def __init__(self, tracking_uri="http://mlflow-tracking"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.logger = logging.getLogger("mlflow_registry")
    
    def create_experiment_if_not_exists(self, experiment_name):
        """Create MLflow experiment if it doesn't exist"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.logger.info(f"Created experiment: {experiment_name} with ID: {experiment_id}")
                return experiment_id
            else:
                return experiment.experiment_id
        except Exception as e:
            self.logger.error(f"Error managing experiment {experiment_name}: {e}")
            raise
    
    def log_model_with_registry(self, model, model_type, symbol, features, target, 
                               metrics, feature_cols, dvc_version_id, model_path):
        """Log model to MLflow with full registry features"""
        
        experiment_name = f"crypto_models_{symbol}"
        experiment_id = self.create_experiment_if_not_exists(experiment_name)
        
        with mlflow.start_run(experiment_id=experiment_id, 
                             run_name=f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # 1. Log parameters
            if model_type == 'lightgbm':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'random_state': 42
                }
            elif model_type == 'xgboost':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            
            mlflow.log_params(params)
            
            # 2. Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # 3. Log feature importance (if available)
            if hasattr(model, 'feature_importance'):
                feature_importance = model.feature_importance()
                importance_dict = dict(zip(feature_cols, feature_importance))
                
                # Log top 10 most important features
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                for feat, importance in sorted_importance:
                    mlflow.log_metric(f"feature_importance_{feat}", importance)
            
            # 4. Create model signature
            signature = infer_signature(features, target)
            
            # 5. Log model with MLflow native logging
            model_info = None
            if model_type == 'lightgbm':
                model_info = mlflow.lightgbm.log_model(
                    lgb_model=model,
                    name="model",
                    signature=signature,
                    registered_model_name=f"crypto_{model_type}_{symbol}"
                )
            elif model_type == 'xgboost':
                model_info = mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model",
                    signature=signature,
                    registered_model_name=f"crypto_{model_type}_{symbol}"
                )
            
            # 6. Log additional artifacts and metadata
            self._log_model_artifacts(run.info.run_id, features, target, feature_cols, 
                                    dvc_version_id, symbol, model_type)
            
            # 7. Tag the run
            mlflow.set_tags({
                "model_type": model_type,
                "symbol": symbol,
                "data_version": dvc_version_id,
                "stage": "training",
                "framework": model_type,
                "data_lineage": f"dvc_version:{dvc_version_id}"
            })
            
            self.logger.info(f"Logged {model_type} model for {symbol} to MLflow")
            return run.info.run_id, model_info.model_uri if model_info else None
    
    def _log_model_artifacts(self, run_id, features, target, feature_cols, 
                           dvc_version_id, symbol, model_type):
        """Log additional artifacts to MLflow"""
        
        # 1. Feature statistics
        feature_stats = features.describe()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            feature_stats.to_json(f.name)
            mlflow.log_artifact(f.name, "feature_stats")
            os.unlink(f.name)
        
        # 2. Data lineage information
        lineage_info = {
            "dvc_version_id": dvc_version_id,
            "symbol": symbol,
            "model_type": model_type,
            "feature_count": len(feature_cols),
            "training_samples": len(features),
            "feature_columns": feature_cols,
            "data_shape": features.shape,
            "target_distribution": {
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max())
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(lineage_info, f, indent=2)
            mlflow.log_artifact(f.name, "data_lineage")
            os.unlink(f.name)
        
        # 3. Model validation plots (if needed)
        # You can add plotting code here
    
    def promote_model_to_staging(self, model_name, model_version=None):
        """Promote model to staging"""
        try:
            if model_version is None:
                # Get latest version
                latest_version = self.client.get_latest_versions(
                    model_name, stages=["None"]
                )[0]
                model_version = latest_version.version
            
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging"
            )
            
            self.logger.info(f"Promoted {model_name} version {model_version} to Staging")
            return model_version
            
        except Exception as e:
            self.logger.error(f"Failed to promote model {model_name}: {e}")
            raise
    
    def promote_model_to_production(self, model_name, model_version):
        """Promote model to production"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production"
            )
            
            self.logger.info(f"Promoted {model_name} version {model_version} to Production")
            return model_version
            
        except Exception as e:
            self.logger.error(f"Failed to promote model {model_name} to production: {e}")
            raise
    
    def add_model_description(self, model_name, model_version, description):
        """Add description to model version"""
        try:
            self.client.update_model_version(
                name=model_name,
                version=model_version,
                description=description
            )
            self.logger.info(f"Added description to {model_name} version {model_version}")
        except Exception as e:
            self.logger.error(f"Failed to add description: {e}")
    
    def compare_model_versions(self, model_name, version1, version2):
        """Compare two model versions"""
        try:
            v1_details = self.client.get_model_version(model_name, version1)
            v2_details = self.client.get_model_version(model_name, version2)
            
            # Get run details for metrics comparison
            v1_run = self.client.get_run(v1_details.run_id)
            v2_run = self.client.get_run(v2_details.run_id)
            
            comparison = {
                "model_name": model_name,
                "version_1": {
                    "version": version1,
                    "metrics": v1_run.data.metrics,
                    "stage": v1_details.current_stage,
                    "creation_time": v1_details.creation_timestamp
                },
                "version_2": {
                    "version": version2,
                    "metrics": v2_run.data.metrics,
                    "stage": v2_details.current_stage,
                    "creation_time": v2_details.creation_timestamp
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare model versions: {e}")
            raise
    
    def get_production_model(self, model_name):
        """Get current production model"""
        try:
            production_versions = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            
            if production_versions:
                return production_versions[0]
            else:
                self.logger.warning(f"No production version found for {model_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get production model {model_name}: {e}")
            raise
    
    def archive_model_version(self, model_name, model_version):
        """Archive a model version"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Archived"
            )
            
            self.logger.info(f"Archived {model_name} version {model_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to archive model version: {e}")
            raise

class ModelTrainingPipeline:
    def __init__(self, db_config):
        self.logger = logging.getLogger("model_training")
        self.db_config = db_config
        self.minio_client = self._get_minio_client()
        self._ensure_bucket_exists()
        self.models = {}
        self.scalers = {}
        # Add MLflow registry
        self.mlflow_registry = MLflowModelRegistry()

    
    def _get_minio_client(self):
        try:
            client = Minio(
                MINIO_CONFIG['endpoint'],
                access_key=MINIO_CONFIG['access_key'],
                secret_key=MINIO_CONFIG['secret_key'],
                secure=MINIO_CONFIG['secure']
            )
            self.logger.info(f"MinIO Client Initialized")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize MinIO Client: {str(e)}")
            raise

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            if not self.minio_client.bucket_exists(BUCKET_NAME):
                self.minio_client.make_bucket(BUCKET_NAME)
                self.logger.info(f"Created bucket: {BUCKET_NAME}")
            else:
                self.logger.info(f"Bucket {BUCKET_NAME} already exists")
        except S3Error as e:
            self.logger.error(f"Error with bucket operations: {str(e)}")
            raise

    def get_dvc_version_metadata(self, version_id):
        """Get DVC version metadata from database"""
        query = """
        SELECT dataset_name, file_path, data_schema, metadata
        FROM dvc_data_versions 
        WHERE version_id = %s
        """
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (version_id,))
                    result = cur.fetchone()
                    
                    if not result:
                        raise ValueError(f"DVC version {version_id} not found")
                    
                    dataset_name, file_path, schema, metadata = result
                    
            return {
                'dataset_name': dataset_name,
                'file_path': file_path,
                'schema': schema,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get DVC metadata for {version_id}: {e}")
            raise

    def load_features_from_dvc_fallback(self, version_id):
        """Load features using DVC metadata but falling back to MinIO if needed"""
        try:
            # Get version metadata from database
            version_info = self.get_dvc_version_metadata(version_id)
            metadata = version_info['metadata']
            
            # Try to load from original MinIO storage path if available
            if metadata and 'source_storage_path' in metadata:
                storage_path = metadata['source_storage_path']
                self.logger.info(f"Loading features for {version_id} from original MinIO path: {storage_path}")
                
                response = self.minio_client.get_object('crypto-features', storage_path)
                df = pd.read_parquet(BytesIO(response.read()))
                response.close()
                response.release_conn()
                
                return df, metadata
            else:
                # Fallback: try to construct storage path from version_id
                # version_id format: features_SYMBOL_timestamp_hash
                parts = version_id.split('_')
                if len(parts) >= 3 and parts[0] == 'features':
                    symbol = parts[1]
                    timestamp = parts[2]
                    date_partition = timestamp[:8]  # YYYYMMDD
                    storage_path = f"features/{date_partition}/{symbol}.parquet"
                    
                    self.logger.info(f"Trying fallback storage path: {storage_path}")
                    response = self.minio_client.get_object('crypto-features', storage_path)
                    df = pd.read_parquet(BytesIO(response.read()))
                    response.close()
                    response.release_conn()
                    
                    return df, metadata
                else:
                    raise ValueError(f"Cannot determine storage path for version {version_id}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load features for version {version_id}: {e}")
            raise

    def prepare_training_data(self, df, target_col='return_1h', feature_cols=None):
        """Prepare features and target for training with debugging"""
        
        self.logger.info(f"Input DataFrame shape: {df.shape}")
        self.logger.info(f"Input DataFrame columns: {list(df.columns)}")
        self.logger.info(f"Input DataFrame head:\n{df.head()}")
        
        # Check if target column exists
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found in DataFrame")
            self.logger.info(f"Available columns: {list(df.columns)}")
            # Fallback to creating return_1h if close_price exists
            if 'close_price' in df.columns:
                self.logger.info("Creating return_1h from close_price")
                df[target_col] = df['close_price'].pct_change()
            else:
                raise ValueError(f"Cannot create target column {target_col}")
        
        # Remove non-numeric columns and handle missing values
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.info(f"Numeric columns found: {len(numeric_cols)}")
            
            # Remove target column and time-related columns from features
            exclude_cols = [target_col, 'open_time'] + [col for col in numeric_cols if 'return' in col and col != target_col]
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            self.logger.info(f"Excluded columns: {exclude_cols}")
            self.logger.info(f"Feature columns selected: {len(feature_cols)}")
        
        # Check if we have enough features
        if len(feature_cols) == 0:
            raise ValueError("No feature columns available for training")
        
        # Create target variable (shifted for prediction)
        target = df[target_col].shift(-1)  # Predict next period return
        self.logger.info(f"Target created, non-null values: {target.notna().sum()}")
        
        # Align features and target
        features = df[feature_cols].iloc[:-1]  # Remove last row due to shift
        target = target.iloc[:-1]  # Remove last row due to shift
        
        self.logger.info(f"After alignment - Features shape: {features.shape}, Target shape: {target.shape}")
        
        # Check for missing values before filtering
        features_na_count = features.isna().sum().sum()
        target_na_count = target.isna().sum()
        self.logger.info(f"Missing values - Features: {features_na_count}, Target: {target_na_count}")
        
        # Check individual feature null counts
        null_counts = features.isnull().sum()
        high_null_features = null_counts[null_counts > len(features) * 0.5]  # More than 50% null
        if len(high_null_features) > 0:
            self.logger.warning(f"Features with >50% null values: {dict(high_null_features)}")
        
        # Remove rows with missing values
        valid_idx = features.notna().all(axis=1) & target.notna()
        self.logger.info(f"Valid rows before filtering: {len(features)}")
        self.logger.info(f"Valid rows after filtering: {valid_idx.sum()}")
        
        features = features[valid_idx]
        target = target[valid_idx]
        
        self.logger.info(f"Final shapes - Features: {features.shape}, Target: {target.shape}")
        
        # If still empty, try more lenient filtering
        if len(features) == 0:
            self.logger.warning("No valid rows found with strict filtering, trying lenient approach")
            
            # Allow some missing values (less than 20% per row)
            features = df[feature_cols].iloc[:-1]
            target = df[target_col].shift(-1).iloc[:-1]
            
            # Remove rows where target is null or more than 20% of features are null
            features_null_pct = features.isnull().sum(axis=1) / len(feature_cols)
            valid_idx = target.notna() & (features_null_pct < 0.2)
            
            features = features[valid_idx]
            target = target[valid_idx]
            
            # Fill remaining missing values with forward fill then median
            features = features.fillna(method='ffill').fillna(features.median())
            
            self.logger.info(f"Lenient filtering - Final shapes: Features: {features.shape}, Target: {target.shape}")
        
        if len(features) == 0:
            raise ValueError("No valid training data available after preprocessing")
        
        return features, target, feature_cols

    def train_lightgbm(self, X_train, y_train, X_val, y_val, params=None):
        """Train LightGBM model"""
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model

    def evaluate_model(self, model, X_test, y_test, model_type='lightgbm'):
        """Evaluate model performance"""
        if model_type == 'lightgbm':
            y_pred = model.predict(X_test)
        elif model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'directional_accuracy': np.mean(np.sign(y_pred) == np.sign(y_test))
        }
        
        return metrics, y_pred

    def cross_validate_model(self, features, target, model_type='lightgbm', n_splits=5):
        """Perform time series cross validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            self.logger.info(f"Training {model_type} fold {fold + 1}/{n_splits}")
            
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
            
            if model_type == 'lightgbm':
                model = self.train_lightgbm(X_train, y_train, X_val, y_val)
            elif model_type == 'xgboost':
                model = self.train_xgboost(X_train, y_train, X_val, y_val)
            
            metrics, _ = self.evaluate_model(model, X_val, y_val, model_type)
            cv_scores.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in cv_scores[0].keys():
            avg_metrics[f'{metric}_mean'] = np.mean([score[metric] for score in cv_scores])
            avg_metrics[f'{metric}_std'] = np.std([score[metric] for score in cv_scores])
        
        return avg_metrics, cv_scores

    def save_model_with_mlflow_registry(self, model, symbol, model_type, metrics, feature_cols, 
                                    source_dvc_version, features, target):
        """Save model to both MinIO and MLflow with full registry features"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save to MinIO (existing functionality)
        model_buffer = BytesIO()
        pickle.dump(model, model_buffer)
        model_buffer.seek(0)
        
        model_path = f"models/{symbol}/{model_type}/{timestamp}/model.pkl"
        self.minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=model_path,
            data=model_buffer,
            length=len(model_buffer.getvalue()),
            content_type='application/octet-stream'
        )
        
        # 2. Save metadata to MinIO
        metadata = {
            'symbol': symbol,
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_columns': feature_cols,
            'model_path': model_path,
            'source_dvc_version': source_dvc_version,
            'data_lineage': {
                'feature_version_id': source_dvc_version,
                'training_timestamp': timestamp
            }
        }
        
        metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode('utf-8'))
        metadata_path = f"models/{symbol}/{model_type}/{timestamp}/metadata.json"
        
        self.minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=metadata_path,
            data=metadata_buffer,
            length=len(metadata_buffer.getvalue()),
            content_type='application/json'
        )
        
        # 3. Log to MLflow with full registry features
        try:
            mlflow_run_id, mlflow_model_uri = self.mlflow_registry.log_model_with_registry(
                model=model,
                model_type=model_type,
                symbol=symbol,
                features=features,
                target=target,
                metrics=metrics,
                feature_cols=feature_cols,
                dvc_version_id=source_dvc_version,
                model_path=model_path
            )
            
            # Update metadata with MLflow info
            metadata['mlflow_run_id'] = mlflow_run_id
            metadata['mlflow_model_uri'] = mlflow_model_uri
            metadata['mlflow_registered_name'] = f"crypto_{model_type}_{symbol}"
            
            # Re-save updated metadata
            updated_metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode('utf-8'))
            self.minio_client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=metadata_path,
                data=updated_metadata_buffer,
                length=len(updated_metadata_buffer.getvalue()),
                content_type='application/json'
            )
            
            self.logger.info(f"Saved {model_type} model for {symbol} to MinIO and MLflow")
            return model_path, metadata_path, mlflow_run_id, mlflow_model_uri
            
        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {e}")
            # Still return MinIO paths even if MLflow fails
            return model_path, metadata_path, None, None

    # Update your train_symbol_models method to use the new save method
    def train_symbol_models(self, symbol, dvc_version_id):
        """Train all model types for a single symbol using DVC versioned features"""
        self.logger.info(f"Starting model training for {symbol} using DVC version {dvc_version_id}")
        
        # Load features from DVC (with fallback)
        df, feature_metadata = self.load_features_from_dvc_fallback(dvc_version_id)
        
        # Prepare training data
        features, target, feature_cols = self.prepare_training_data(df)
        
        # Split data (80/20 train/test)
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        results = {
            'symbol': symbol,
            'dvc_version_id': dvc_version_id,
            'training_timestamp': datetime.now().isoformat(),
            'data_shape': features.shape,
            'feature_metadata': feature_metadata,
            'models': {}
        }
        
        # Train LightGBM
        try:
            self.logger.info(f"Training LightGBM for {symbol}")
            lgb_model = self.train_lightgbm(X_train, y_train, X_test, y_test)
            lgb_metrics, _ = self.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
            
            # Cross validation
            lgb_cv_metrics, _ = self.cross_validate_model(features, target, 'lightgbm')
            lgb_metrics.update(lgb_cv_metrics)
            
            # Save model with MLflow registry
            lgb_model_path, lgb_metadata_path, lgb_mlflow_run, lgb_mlflow_uri = self.save_model_with_mlflow_registry(
                lgb_model, symbol, 'lightgbm', lgb_metrics, feature_cols, dvc_version_id, features, target
            )
            
            results['models']['lightgbm'] = {
                'metrics': lgb_metrics,
                'model_path': lgb_model_path,
                'metadata_path': lgb_metadata_path,
                'mlflow_run_id': lgb_mlflow_run,
                'mlflow_model_uri': lgb_mlflow_uri,
                'registered_name': f"crypto_lightgbm_{symbol}"
            }
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed for {symbol}: {e}")
            # Don't raise, continue with XGBoost
        
        # Train XGBoost
        try:
            self.logger.info(f"Training XGBoost for {symbol}")
            xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test)
            xgb_metrics, _ = self.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
            
            # Cross validation
            xgb_cv_metrics, _ = self.cross_validate_model(features, target, 'xgboost')
            xgb_metrics.update(xgb_cv_metrics)
            
            # Save model with MLflow registry
            xgb_model_path, xgb_metadata_path, xgb_mlflow_run, xgb_mlflow_uri = self.save_model_with_mlflow_registry(
                xgb_model, symbol, 'xgboost', xgb_metrics, feature_cols, dvc_version_id, features, target
            )
            
            results['models']['xgboost'] = {
                'metrics': xgb_metrics,
                'model_path': xgb_model_path,
                'metadata_path': xgb_metadata_path,
                'mlflow_run_id': xgb_mlflow_run,
                'mlflow_model_uri': xgb_mlflow_uri,
                'registered_name': f"crypto_xgboost_{symbol}"
            }
            
        except Exception as e:
            self.logger.error(f"XGBoost training failed for {symbol}: {e}")
        
        return results

    def run_training_pipeline(self, dvc_versioning_summary):
        """Run training pipeline using DVC versioned features"""
        training_results = {
            'pipeline_timestamp': datetime.now().isoformat(),
            'symbols_trained': [],
            'model_results': {},
            'dvc_lineage': dvc_versioning_summary
        }
        
        # Get feature version IDs from DVC versioning summary
        feature_version_ids = dvc_versioning_summary.get('feature_version_ids', [])
        
        if not feature_version_ids:
            raise ValueError("No DVC feature version IDs found in versioning summary")
        
        # Map version IDs to symbols (assuming version ID format: features_SYMBOL_timestamp_hash)
        for version_id in feature_version_ids:
            try:
                # Extract symbol from version ID
                symbol = version_id.split('_')[1]  # features_SYMBOL_timestamp_hash
                
                symbol_results = self.train_symbol_models(symbol, version_id)
                training_results['symbols_trained'].append(symbol)
                training_results['model_results'][symbol] = symbol_results
                
            except Exception as e:
                self.logger.error(f"Training failed for version {version_id}: {e}")
                continue
        
        self.logger.info(f"Training pipeline completed for {len(training_results['symbols_trained'])} symbols")
        return training_results

    def load_trained_model(self, model_path, model_type):
        """Load a trained model from MinIO"""
        try:
            response = self.minio_client.get_object(BUCKET_NAME, model_path)
            model_buffer = BytesIO(response.read())
            response.close()
            response.release_conn()
            
            model_buffer.seek(0)
            model = pickle.load(model_buffer)
            
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    # Add model promotion methods to your ModelTrainingPipeline class
    def promote_best_models_to_staging(self, training_results):
        """Promote best performing models to staging based on metrics"""
        promoted_models = []
        
        for symbol, symbol_results in training_results['model_results'].items():
            best_model = None
            best_metric = float('inf')
            
            # Find best model based on RMSE
            for model_type, model_info in symbol_results['models'].items():
                rmse = model_info['metrics'].get('rmse', float('inf'))
                if rmse < best_metric:
                    best_metric = rmse
                    best_model = (model_type, model_info)
            
            if best_model:
                model_type, model_info = best_model
                registered_name = model_info['registered_name']
                
                try:
                    # Promote to staging
                    version = self.mlflow_registry.promote_model_to_staging(registered_name)
                    
                    # Add description
                    description = f"Best {model_type} model for {symbol} - RMSE: {best_metric:.6f}"
                    self.mlflow_registry.add_model_description(registered_name, version, description)
                    
                    promoted_models.append({
                        'symbol': symbol,
                        'model_type': model_type,
                        'registered_name': registered_name,
                        'version': version,
                        'rmse': best_metric
                    })
                    
                    self.logger.info(f"Promoted {registered_name} to staging with RMSE: {best_metric:.6f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to promote {registered_name}: {e}")
        
        return promoted_models