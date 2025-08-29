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

from automated_data_validation import DataValidationSuite


# MinIO Configuration
MINIO_CONFIG = {
    'endpoint': 'minio:9000',
    'access_key': 'admin',
    'secret_key': 'admin123',
    'secure': False
}

BUCKET_NAME = 'crypto-models'

# ADD this configuration at the top of the file:
PREDICTION_CONFIGS = {
    'return_1step': {
        'target_col': 'target_return_1steps',
        'task_type': 'regression',
        'horizon_steps': 1,
        'description': '15-minute price return prediction'
    },
    'return_4step': {
        'target_col': 'target_return_4steps', 
        'task_type': 'regression',
        'horizon_steps': 4,
        'description': '1-hour price return prediction'
    },
    'return_16step': {
        'target_col': 'target_return_16steps',
        'task_type': 'regression', 
        'horizon_steps': 16,
        'description': '4-hour price return prediction'
    },
    'direction_4step': {
        'target_col': 'target_direction_4steps',
        'task_type': 'classification_binary',
        'horizon_steps': 4,
        'description': '1-hour direction prediction (up/down)'
    },
    'direction_multi_4step': {
        'target_col': 'target_direction_multi_4steps',
        'task_type': 'classification_multi',
        'horizon_steps': 4,
        'description': '1-hour multi-class direction prediction'
    },
    'volatility_4step': {
        'target_col': 'target_volatility_4steps',
        'task_type': 'regression',
        'horizon_steps': 4,
        'description': '1-hour volatility prediction'
    },
    'vol_regime_4step': {
        'target_col': 'target_vol_regime_4steps',
        'task_type': 'classification_multi',
        'horizon_steps': 4,
        'description': '1-hour volatility regime prediction'
    }
}


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
        """Log model to MLflow with full registry features for multi-task models"""
        
        # Parse composite model_type (e.g., "xgboost_vol_regime_4step")
        # Format: {algorithm}_{task_name}
        parts = model_type.split('_', 1)  # Split on first underscore only
        algorithm = parts[0]  # e.g., "xgboost" 
        task_name = parts[1] if len(parts) > 1 else "unknown"  # e.g., "vol_regime_4step"
        
        # Get task configuration for parameters
        task_config = PREDICTION_CONFIGS.get(task_name, {})
        task_type = task_config.get('task_type', 'regression')
        
        experiment_name = f"crypto_multi_models_{symbol}"
        experiment_id = self.create_experiment_if_not_exists(experiment_name)
        
        with mlflow.start_run(experiment_id=experiment_id, 
                            run_name=f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # 1. Log parameters based on algorithm and task type
            params = self._get_model_params(algorithm, task_type)
            mlflow.log_params(params)
            
            # 2. Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # 3. Log feature importance (if available)
            self._log_feature_importance(model, feature_cols, algorithm)
            
            # 4. Create model signature
            signature = infer_signature(features, target)
            
            # 5. Log model with correct registered name format
            registered_model_name = f"crypto_{algorithm}_{task_name}_{symbol}"
            model_info = self._log_model_by_algorithm(model, algorithm, signature, registered_model_name)
            
            # 6. Log additional artifacts and metadata
            self._log_model_artifacts(run.info.run_id, features, target, feature_cols, 
                                    dvc_version_id, symbol, model_type)
            
            # 7. Tag the run with task-specific information
            mlflow.set_tags({
                "algorithm": algorithm,
                "task_name": task_name,
                "task_type": task_type,
                "symbol": symbol,
                "data_version": dvc_version_id,
                "stage": "training",
                "framework": algorithm,
                "data_lineage": f"dvc_version:{dvc_version_id}",
                "composite_model_type": model_type
            })
            
            self.logger.info(f"Logged {model_type} model for {symbol} to MLflow as {registered_model_name}")
            return run.info.run_id, model_info.model_uri if model_info else None

    def _get_model_params(self, algorithm, task_type):
        """Get algorithm and task-specific parameters"""
        base_params = {
            'learning_rate': 0.05,
            'random_state': 42
        }
        
        if algorithm == 'lightgbm':
            params = {
                **base_params,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Task-specific objectives
            if task_type == 'regression':
                params.update({'objective': 'regression', 'metric': 'rmse'})
            elif task_type == 'classification_binary':
                params.update({'objective': 'binary', 'metric': 'binary_logloss'})
            elif task_type == 'classification_multi':
                params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 5})
                
        elif algorithm == 'xgboost':
            params = {
                **base_params,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            # Task-specific objectives
            if task_type == 'regression':
                params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse'})
            elif task_type == 'classification_binary':
                params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss'})
            elif task_type == 'classification_multi':
                params.update({'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 5})
        
        return params

    def _log_feature_importance(self, model, feature_cols, algorithm):
        """Log feature importance based on algorithm"""
        try:
            if algorithm == 'lightgbm' and hasattr(model, 'feature_importance'):
                feature_importance = model.feature_importance()
                importance_dict = dict(zip(feature_cols, feature_importance))
            elif algorithm == 'xgboost' and hasattr(model, 'get_score'):
                # XGBoost uses get_score() method
                importance_dict = model.get_score(importance_type='weight')
                # Map back to original feature names if needed
                importance_dict = {feat: importance_dict.get(f'f{i}', importance_dict.get(feat, 0)) 
                                for i, feat in enumerate(feature_cols)}
            else:
                return
            
            # Log top 10 most important features
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, importance in sorted_importance:
                mlflow.log_metric(f"feature_importance_{feat}", float(importance))
                
        except Exception as e:
            self.logger.warning(f"Failed to log feature importance: {e}")

    def _log_model_by_algorithm(self, model, algorithm, signature, registered_model_name):
        """Log model using algorithm-specific MLflow logging"""
        model_info = None
        
        if algorithm == 'lightgbm':
            model_info = mlflow.lightgbm.log_model(
                lgb_model=model,
                name="model",
                signature=signature,
                registered_model_name=registered_model_name
            )
        elif algorithm == 'xgboost':
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                name="model", 
                signature=signature,
                registered_model_name=registered_model_name
            )
        else:
            # Fallback to generic sklearn logging
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                registered_model_name=registered_model_name
            )
        
        return model_info
    
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
        # ADD validation suite
        self.validator = DataValidationSuite(db_config)

    
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

    def load_features_and_targets_from_dvc(self, feature_version_id, target_version_id=None):
        """Load both features and targets using DVC metadata"""
        try:
            # Load features
            features_df, feature_metadata = self.load_features_from_dvc_fallback(feature_version_id)
            
            # Load targets (use same version_id if not specified)
            if target_version_id is None:
                target_version_id = feature_version_id.replace('features_', 'targets_')
            
            # Try to load targets from MinIO
            try:
                # Extract symbol and date from version_id
                parts = feature_version_id.split('_')
                symbol = parts[1]
                timestamp = parts[2]
                date_partition = timestamp[:8]  # YYYYMMDD
                
                targets_path = f"targets/{date_partition}/{symbol}.parquet"
                
                response = self.minio_client.get_object('crypto-features', targets_path)
                targets_df = pd.read_parquet(BytesIO(response.read()))
                response.close()
                response.release_conn()
                
                self.logger.info(f"Loaded targets from {targets_path}")
                
            except Exception as e:
                self.logger.warning(f"Could not load separate targets, will create from features: {e}")
                # Fallback: create targets from features using your new method
                targets_df = self.create_multi_horizon_targets_from_features(features_df)
            
            return features_df, targets_df, feature_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load features and targets: {e}")
            raise

    def create_multi_horizon_targets_from_features(self, features_df):
        """Create targets from features DataFrame if targets not available separately"""
        targets = features_df[['open_time']].copy() if 'open_time' in features_df.columns else pd.DataFrame(index=features_df.index)
        
        # Assume close_price exists for target creation
        if 'close_price' not in features_df.columns:
            raise ValueError("close_price column needed for target creation")
        
        close = features_df['close_price']
        
        for h in [1, 4, 16, 96]:  # 15min, 1h, 4h, 24h
            # Price return targets
            targets[f'target_return_{h}steps'] = close.pct_change(periods=h).shift(-h)
            
            # Direction targets  
            targets[f'target_direction_{h}steps'] = (targets[f'target_return_{h}steps'] > 0).astype(int)
            
            # Multi-class direction
            returns = targets[f'target_return_{h}steps']
            targets[f'target_direction_multi_{h}steps'] = pd.cut(
                returns,
                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                labels=[0, 1, 2, 3, 4]
            )
            
            # Volatility targets
            targets[f'target_volatility_{h}steps'] = close.pct_change().rolling(h).std().shift(-h)
            
            # Volatility regime targets
            vol = targets[f'target_volatility_{h}steps']
            vol_quantiles = vol.quantile([0.33, 0.67])
            targets[f'target_vol_regime_{h}steps'] = pd.cut(
                vol,
                bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
                labels=[0, 1, 2]
            )
        
        return targets

    def prepare_training_data(self, features_df, targets_df, task_config):
        """
        Prepare features and target for training with proper multi-step alignment
        
        Args:
            features_df: DataFrame with engineered features
            targets_df: DataFrame with multi-horizon targets  
            task_config: Configuration dict from PREDICTION_CONFIGS
        """
        target_col = task_config['target_col']
        task_type = task_config['task_type']
        horizon_steps = task_config['horizon_steps']
        
        self.logger.info(f"Preparing training data for {target_col}")
        self.logger.info(f"Features shape: {features_df.shape}, Targets shape: {targets_df.shape}")
        
        # Check if target column exists
        if target_col not in targets_df.columns:
            self.logger.error(f"Target column '{target_col}' not found in targets DataFrame")
            self.logger.info(f"Available target columns: {list(targets_df.columns)}")
            raise ValueError(f"Target column {target_col} not found")
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['open_time', 'datetime'] + [col for col in features_df.columns if col.startswith('target_')]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols and features_df[col].dtype in ['float64', 'int64']]
        
        self.logger.info(f"Selected {len(feature_cols)} feature columns")
        
        # Align features and targets on index (datetime)
        common_index = features_df.index.intersection(targets_df.index)
        features_aligned = features_df.loc[common_index, feature_cols]
        target_aligned = targets_df.loc[common_index, target_col]
        
        self.logger.info(f"After alignment: Features {features_aligned.shape}, Target {target_aligned.shape}")
        
        # Remove rows with missing target values
        valid_target_mask = target_aligned.notna()
        features_clean = features_aligned[valid_target_mask]
        target_clean = target_aligned[valid_target_mask]
        
        self.logger.info(f"After removing missing targets: Features {features_clean.shape}, Target {target_clean.shape}")
        
        # Remove rows with too many missing features (> 20% missing)
        feature_missing_pct = features_clean.isnull().sum(axis=1) / len(feature_cols)
        valid_feature_mask = feature_missing_pct < 0.2
        
        features_final = features_clean[valid_feature_mask]
        target_final = target_clean[valid_feature_mask]
        
        self.logger.info(f"After feature cleaning: Features {features_final.shape}, Target {target_final.shape}")
        
        # Fill remaining missing values
        if features_final.isnull().sum().sum() > 0:
            features_final = features_final.fillna(method='ffill').fillna(features_final.median())
        
        # For classification tasks, ensure target is properly encoded
        if task_type in ['classification_binary', 'classification_multi']:
            if target_final.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                target_final = pd.Series(le.fit_transform(target_final), index=target_final.index)
            target_final = target_final.astype(int)
        
        if len(features_final) == 0:
            raise ValueError("No valid training data available after preprocessing")
        
        self.logger.info(f"Final dataset: {len(features_final)} samples, {len(feature_cols)} features")
        self.logger.info(f"Target statistics: min={target_final.min():.4f}, max={target_final.max():.4f}, mean={target_final.mean():.4f}")
        
        return features_final, target_final, feature_cols, task_config
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, task_config, params=None):
        """Train LightGBM model with task-specific configuration"""
        if params is None:
            if task_config['task_type'] == 'classification_binary':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }
            elif task_config['task_type'] == 'classification_multi':
                params = {
                    'objective': 'multiclass',
                    'num_class': 5,  # Adjust based on your classes
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }
            else:
                # Regression (default)
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

    def train_xgboost(self, X_train, y_train, X_val, y_val, task_config, params=None):
        """Train XGBoost model with task-specific configuration"""
        if params is None:
            if task_config['task_type'] == 'classification_binary':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            elif task_config['task_type'] == 'classification_multi':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': 5,  # Adjust based on your classes
                    'eval_metric': 'mlogloss',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            else:
                # Regression (default)
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
    
    def evaluate_model(self, model, X_test, y_test, model_type='lightgbm', task_config=None):
        """Evaluate model performance based on task type"""
        if model_type == 'lightgbm':
            if task_config and task_config['task_type'] in ['classification_binary', 'classification_multi']:
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int) if task_config['task_type'] == 'classification_binary' else np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
                
        elif model_type == 'xgboost':
            if task_config and task_config['task_type'] in ['classification_binary', 'classification_multi']:
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = model.predict(dtest)
                y_pred = (y_pred_proba > 0.5).astype(int) if task_config['task_type'] == 'classification_binary' else np.argmax(y_pred_proba, axis=1)
            else:
                dtest = xgb.DMatrix(X_test)
                y_pred = model.predict(dtest)
        
        # Calculate appropriate metrics based on task type
        if task_config and task_config['task_type'] == 'classification_binary':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'auc': roc_auc_score(y_test, y_pred_proba if 'y_pred_proba' in locals() else y_pred)
            }
        elif task_config and task_config['task_type'] == 'classification_multi':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'), 
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            # Regression metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            if len(y_test) > 0:
                metrics['directional_accuracy'] = np.mean(np.sign(y_pred) == np.sign(y_test))
        
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
    def train_symbol_models(self, symbol, feature_version_id, target_version_id=None, tasks_to_train='all', task_types=None):
        """Train multiple models with validation checks"""
        self.logger.info(f"Starting multi-task model training for {symbol} using version {feature_version_id}")
        
        # Load features and targets
        features_df, targets_df, feature_metadata = self.load_features_and_targets_from_dvc(feature_version_id, target_version_id)
        
        # Determine which tasks to train
        if tasks_to_train == 'all' or tasks_to_train is None:
            tasks_to_train = list(PREDICTION_CONFIGS.keys())
        elif isinstance(tasks_to_train, str) and tasks_to_train != 'all':
            tasks_to_train = [tasks_to_train]
        
        # Filter by task type if specified
        if task_types is not None:
            if isinstance(task_types, str):
                task_types = [task_types]
            
            filtered_tasks = []
            for task_name in tasks_to_train:
                if task_name in PREDICTION_CONFIGS:
                    if PREDICTION_CONFIGS[task_name]['task_type'] in task_types:
                        filtered_tasks.append(task_name)
            tasks_to_train = filtered_tasks
        
        self.logger.info(f"Training {len(tasks_to_train)} tasks: {tasks_to_train}")
        
        # VALIDATION STEP 1: Validate features and targets separately
        self.logger.info("Running data validation...")
        
        features_validation = self.validator.run_validation_suite(
            features_df, f"features_{symbol}", symbol, dataset_type='features'
        )
        
        targets_validation = self.validator.run_validation_suite(
            targets_df, f"targets_{symbol}", symbol, dataset_type='targets'
        )
        
        # VALIDATION STEP 2: Check training readiness for selected tasks
        selected_task_configs = [PREDICTION_CONFIGS[task] for task in tasks_to_train if task in PREDICTION_CONFIGS]
        
        training_readiness = self.validator.validate_model_training_readiness(
            features_df, targets_df, selected_task_configs, symbol
        )
        
        # Check if there are critical validation failures
        critical_failures = (
            features_validation['critical_failures'] + 
            targets_validation['critical_failures'] + 
            (1 if not training_readiness['ready_for_training'] else 0)
        )
        
        if critical_failures > 0:
            error_msg = f"Critical validation failures detected: {critical_failures} issues"
            self.logger.error(error_msg)
            
            # Log validation summaries
            self.logger.error(f"Features validation: {features_validation['summary']}")
            self.logger.error(f"Targets validation: {targets_validation['summary']}")
            self.logger.error(f"Training readiness: {training_readiness['summary']}")
            
            # You can choose to either raise an exception or continue with warnings
            raise ValueError(f"Data validation failed: {error_msg}")
        
        # Log validation success
        self.logger.info(f"Data validation passed:")
        self.logger.info(f"- Features: {features_validation['summary']}")
        self.logger.info(f"- Targets: {targets_validation['summary']}")
        self.logger.info(f"- Training readiness: {training_readiness['summary']}")
        
        # Continue with existing training logic...
        results = {
            'symbol': symbol,
            'feature_version_id': feature_version_id,
            'target_version_id': target_version_id,
            'training_timestamp': datetime.now().isoformat(),
            'data_shape': features_df.shape,
            'feature_metadata': feature_metadata,
            'validation_results': {
                'features': features_validation,
                'targets': targets_validation,
                'training_readiness': training_readiness
            },
            'tasks': {}
        }
        
        # Train models for each task
        for task_name in tasks_to_train:
            if task_name not in PREDICTION_CONFIGS:
                self.logger.warning(f"Unknown task: {task_name}")
                continue
                
            task_config = PREDICTION_CONFIGS[task_name]
            
            try:
                self.logger.info(f"Training {task_name} for {symbol}")
                
                # Prepare data for this specific task (with additional validation)
                features, target, feature_cols, task_info = self.prepare_training_data_validated(
                    features_df, targets_df, task_config, symbol
                )
                
                # Split data (80/20 train/test, time-aware)
                split_idx = int(0.8 * len(features))
                X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
                y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
                
                task_results = {'task_config': task_config, 'models': {}}
                
                # Train different model types for this task
                for model_type in ['lightgbm', 'xgboost']:
                    try:
                        if model_type == 'lightgbm':
                            model = self.train_lightgbm(X_train, y_train, X_test, y_test, task_config)
                        elif model_type == 'xgboost':
                            model = self.train_xgboost(X_train, y_train, X_test, y_test, task_config)
                        
                        # Evaluate model
                        metrics, predictions = self.evaluate_model(model, X_test, y_test, model_type, task_config)
                        
                        # Save model with MLflow
                        model_path, metadata_path, mlflow_run, mlflow_uri = self.save_model_with_mlflow_registry(
                            model, symbol, f"{model_type}_{task_name}", metrics, feature_cols,
                            feature_version_id, features, target
                        )
                        
                        task_results['models'][model_type] = {
                            'metrics': metrics,
                            'model_path': model_path,
                            'metadata_path': metadata_path,
                            'mlflow_run_id': mlflow_run,
                            'mlflow_model_uri': mlflow_uri,
                            'registered_name': f"crypto_{model_type}_{task_name}_{symbol}"
                        }
                        
                    except Exception as e:
                        self.logger.error(f"{model_type} training failed for {task_name}/{symbol}: {e}")
                
                results['tasks'][task_name] = task_results
                
            except Exception as e:
                self.logger.error(f"Task {task_name} failed for {symbol}: {e}")
        
        return results

    # ADD new method with additional validation during data preparation:
    def prepare_training_data_validated(self, features_df, targets_df, task_config, symbol):
        """Prepare training data with additional validation steps"""
        
        # Run the standard preparation
        features, target, feature_cols, task_info = self.prepare_training_data(
            features_df, targets_df, task_config
        )
        
        # Additional validation on prepared data
        target_col = task_config['target_col']
        task_type = task_config['task_type']
        
        # Validate final data quality
        if len(features) < 500:
            self.logger.warning(f"Small training dataset for {symbol}/{target_col}: {len(features)} samples")
        
        # Check for data leakage (features correlated too highly with targets)
        if task_type == 'regression':
            # For regression, check correlation between features and targets
            correlations = features.corrwith(target).abs()
            high_corr_features = correlations[correlations > 0.95]  # Suspiciously high correlation
            
            if len(high_corr_features) > 0:
                self.logger.warning(f"Potential data leakage detected in {symbol}/{target_col}:")
                for feat, corr in high_corr_features.items():
                    self.logger.warning(f"  {feat}: {corr:.3f} correlation with target")
        
        # Validate target distribution for classification
        if task_type in ['classification_binary', 'classification_multi']:
            class_counts = target.value_counts()
            min_class_size = class_counts.min()
            total_samples = len(target)
            
            if min_class_size < 50:
                self.logger.warning(f"Small class size in {symbol}/{target_col}: minimum class has {min_class_size} samples")
            
            # Check for class imbalance
            class_imbalance_ratio = class_counts.max() / class_counts.min()
            if class_imbalance_ratio > 10:
                self.logger.warning(f"High class imbalance in {symbol}/{target_col}: ratio {class_imbalance_ratio:.1f}:1")
        
        self.logger.info(f"Training data validation passed for {symbol}/{target_col}")
        
        return features, target, feature_cols, task_info

    # ADD method to validate pipeline results:
    def validate_training_results(self, results):
        """Validate training results for quality checks"""
        validation_summary = {
            'symbol': results['symbol'],
            'validation_timestamp': datetime.now().isoformat(),
            'tasks_trained': len(results['tasks']),
            'models_trained': 0,
            'failed_tasks': [],
            'poor_performance_models': [],
            'validation_warnings': []
        }
        
        for task_name, task_results in results['tasks'].items():
            task_config = PREDICTION_CONFIGS.get(task_name, {})
            task_type = task_config.get('task_type', 'regression')
            
            for model_type, model_info in task_results.get('models', {}).items():
                validation_summary['models_trained'] += 1
                metrics = model_info.get('metrics', {})
                
                # Validate model performance based on task type
                if task_type == 'regression':
                    r2 = metrics.get('r2', 0)
                    if r2 < 0.01:  # Very low R² suggests poor model
                        validation_summary['poor_performance_models'].append({
                            'model': f"{model_type}_{task_name}",
                            'issue': f"Low R²: {r2:.4f}",
                            'metrics': metrics
                        })
                
                elif task_type == 'classification_binary':
                    accuracy = metrics.get('accuracy', 0)
                    if accuracy < 0.55:  # Barely better than random
                        validation_summary['poor_performance_models'].append({
                            'model': f"{model_type}_{task_name}",
                            'issue': f"Low accuracy: {accuracy:.3f}",
                            'metrics': metrics
                        })
                
                # Check for overfitting indicators (you'd need validation metrics for this)
                # This is a placeholder for more sophisticated validation
        
        # Log validation summary
        if validation_summary['poor_performance_models']:
            self.logger.warning(f"Training validation found {len(validation_summary['poor_performance_models'])} poor-performing models")
            for model_issue in validation_summary['poor_performance_models']:
                self.logger.warning(f"  {model_issue['model']}: {model_issue['issue']}")
        
        results['training_validation'] = validation_summary
        return results   
    
    def run_training_pipeline(self, dvc_versioning_summary, enable_validation=True):
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
        for feature_version_id in feature_version_ids:
            try:
                # Extract symbol from version ID
                symbol = feature_version_id.split('_')[1]  # features_SYMBOL_timestamp_hash

                symbol_results = self.train_symbol_models(symbol, feature_version_id)
            
                # Validate training results if enabled
                if enable_validation:
                    symbol_results = self.validate_training_results(symbol_results)
                
                training_results['symbols_trained'].append(symbol)
                training_results['model_results'][symbol] = symbol_results
                
            except Exception as e:
                self.logger.error(f"Training failed for version {feature_version_id}: {e}")
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
        """Promote best performing models to staging based on task-specific metrics"""
        promoted_models = []
        
        # Define promotion criteria for each task type
        promotion_criteria = {
            'regression': {
                'primary_metric': 'rmse',
                'direction': 'minimize',  # lower is better
                'fallback_metric': 'r2',
                'fallback_direction': 'maximize'  # higher is better
            },
            'classification_binary': {
                'primary_metric': 'f1',
                'direction': 'maximize',  # higher is better
                'fallback_metric': 'accuracy',
                'fallback_direction': 'maximize'
            },
            'classification_multi': {
                'primary_metric': 'f1',
                'direction': 'maximize',
                'fallback_metric': 'accuracy', 
                'fallback_direction': 'maximize'
            }
        }
        
        for symbol, symbol_results in training_results['model_results'].items():
            # Group by task and find best model per task
            for task_name, task_results in symbol_results['tasks'].items():
                
                # Determine task type from PREDICTION_CONFIGS
                task_config = PREDICTION_CONFIGS.get(task_name, {})
                task_type = task_config.get('task_type', 'regression')
                
                # Get promotion criteria for this task type
                criteria = promotion_criteria.get(task_type, promotion_criteria['regression'])
                
                best_model = None
                best_metric_value = float('inf') if criteria['direction'] == 'minimize' else float('-inf')
                
                # Compare models for this specific task
                for model_type, model_info in task_results['models'].items():
                    metrics = model_info.get('metrics', {})
                    
                    # Get primary metric
                    primary_metric = metrics.get(criteria['primary_metric'])
                    
                    # Use fallback if primary not available
                    if primary_metric is None:
                        primary_metric = metrics.get(criteria['fallback_metric'])
                        if primary_metric is None:
                            self.logger.warning(f"No suitable metrics found for {symbol}/{task_name}/{model_type}")
                            continue
                    
                    # Check if this model is better
                    is_better = False
                    if criteria['direction'] == 'minimize':
                        is_better = primary_metric < best_metric_value
                    else:
                        is_better = primary_metric > best_metric_value
                    
                    if is_better:
                        best_metric_value = primary_metric
                        best_model = (model_type, model_info, primary_metric)
                
                # Promote the best model for this task
                if best_model:
                    model_type, model_info, metric_value = best_model
                    registered_name = model_info.get('registered_name')
                    
                    if not registered_name:
                        self.logger.error(f"No registered_name found for {symbol}/{task_name}/{model_type}")
                        continue
                    
                    try:
                        # Promote to staging
                        version = self.mlflow_registry.promote_model_to_staging(registered_name)
                        
                        # Create task-specific description
                        metric_name = criteria['primary_metric']
                        description = f"Best {model_type} model for {symbol}/{task_name} - {metric_name}: {metric_value:.6f}"
                        
                        self.mlflow_registry.add_model_description(registered_name, version, description)
                        
                        promoted_models.append({
                            'symbol': symbol,
                            'task': task_name,
                            'task_type': task_type,
                            'model_type': model_type,
                            'registered_name': registered_name,
                            'version': version,
                            'metric_name': metric_name,
                            'metric_value': metric_value
                        })
                        
                        self.logger.info(f"Promoted {registered_name} to staging - {metric_name}: {metric_value:.6f}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to promote {registered_name}: {e}")
        
        # Log summary
        tasks_promoted = {}
        for model in promoted_models:
            task = model['task']
            tasks_promoted[task] = tasks_promoted.get(task, 0) + 1
        
        self.logger.info(f"Promotion summary: {dict(tasks_promoted)}")
        return promoted_models