import pandas as pd
import numpy as np
import psycopg2
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from io import BytesIO
from minio import Minio


# Validation Rules and Results
class ValidationSeverity(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING" 
    INFO = "INFO"

@dataclass
class ValidationResult:
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime

class DataValidationSuite:
    """Automated data validation with configurable rules"""
    
    def __init__(self, db_config):
        self.logger = logging.getLogger("data_validation")
        self.db_config = db_config
        self.validation_results = []
        self._create_validation_tables()
    
    def _create_validation_tables(self):
        """Create tables to store validation results"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS validation_results (
            id SERIAL PRIMARY KEY,
            dataset_name VARCHAR(100) NOT NULL,
            rule_name VARCHAR(100) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            passed BOOLEAN NOT NULL,
            message TEXT,
            details JSONB,
            validation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_version VARCHAR(64),
            symbol VARCHAR(20)
        );
        
        CREATE INDEX IF NOT EXISTS idx_validation_dataset ON validation_results(dataset_name);
        CREATE INDEX IF NOT EXISTS idx_validation_time ON validation_results(validation_time);
        CREATE INDEX IF NOT EXISTS idx_validation_severity ON validation_results(severity);
        """
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_query)
                conn.commit()
            self.logger.info("Validation tables created/verified")
        except Exception as e:
            self.logger.error(f"Failed to create validation tables: {str(e)}")
            raise
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict, dataset_name: str) -> List[ValidationResult]:
        """Validate dataframe schema against expected structure"""
        results = []
        
        # Check required columns
        required_cols = set(expected_schema.get('required_columns', []))
        actual_cols = set(df.columns)
        
        missing_cols = required_cols - actual_cols
        if missing_cols:
            results.append(ValidationResult(
                rule_name="required_columns",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": list(missing_cols)},
                timestamp=datetime.now()
            ))
        else:
            results.append(ValidationResult(
                rule_name="required_columns",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="All required columns present",
                details={"checked_columns": list(required_cols)},
                timestamp=datetime.now()
            ))
        
        # Check data types
        expected_dtypes = expected_schema.get('dtypes', {})
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if expected_dtype not in actual_dtype:
                    results.append(ValidationResult(
                        rule_name="column_dtypes",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Column {col} has wrong dtype: expected {expected_dtype}, got {actual_dtype}",
                        details={"column": col, "expected": expected_dtype, "actual": actual_dtype},
                        timestamp=datetime.now()
                    ))
        
        # Check row count
        min_rows = expected_schema.get('min_rows', 0)
        max_rows = expected_schema.get('max_rows', float('inf'))
        
        if len(df) < min_rows:
            results.append(ValidationResult(
                rule_name="row_count_min",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Insufficient rows: {len(df)} < {min_rows}",
                details={"actual_rows": len(df), "min_required": min_rows},
                timestamp=datetime.now()
            ))
        elif len(df) > max_rows:
            results.append(ValidationResult(
                rule_name="row_count_max",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Too many rows: {len(df)} > {max_rows}",
                details={"actual_rows": len(df), "max_allowed": max_rows},
                timestamp=datetime.now()
            ))
        
        return results
    
    def validate_data_quality(self, df: pd.DataFrame, quality_rules: Dict, dataset_name: str) -> List[ValidationResult]:
        """Validate data quality metrics"""
        results = []
        
        # Null value checks
        max_null_pct = quality_rules.get('max_null_percentage', 10.0)
        null_percentages = (df.isnull().sum() / len(df) * 100)
        
        for col, null_pct in null_percentages.items():
            if null_pct > max_null_pct:
                results.append(ValidationResult(
                    rule_name="null_percentage",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Column {col} has high null percentage: {null_pct:.2f}%",
                    details={"column": col, "null_percentage": null_pct, "threshold": max_null_pct},
                    timestamp=datetime.now()
                ))
        
        # Duplicate rows check
        duplicate_count = df.duplicated().sum()
        max_duplicates = quality_rules.get('max_duplicate_percentage', 5.0)
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        if duplicate_pct > max_duplicates:
            results.append(ValidationResult(
                rule_name="duplicate_rows",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"High duplicate percentage: {duplicate_pct:.2f}%",
                details={"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_pct},
                timestamp=datetime.now()
            ))
        
        # Outlier detection for numeric columns
        outlier_z_threshold = quality_rules.get('outlier_z_threshold', 4.0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and not df[col].empty:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_count = (z_scores > outlier_z_threshold).sum()
                outlier_pct = (outlier_count / len(df)) * 100
                
                max_outlier_pct = quality_rules.get('max_outlier_percentage', 2.0)
                if outlier_pct > max_outlier_pct:
                    results.append(ValidationResult(
                        rule_name="outlier_detection",
                        severity=ValidationSeverity.INFO,
                        passed=False,
                        message=f"Column {col} has high outlier percentage: {outlier_pct:.2f}%",
                        details={"column": col, "outlier_count": outlier_count, "outlier_percentage": outlier_pct},
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def validate_crypto_specific(self, df: pd.DataFrame, symbol: str) -> List[ValidationResult]:
        """Crypto-specific validation rules"""
        results = []
        
        # Price validation
        if 'close_price' in df.columns:
            # Prices should be positive
            negative_prices = (df['close_price'] <= 0).sum()
            if negative_prices > 0:
                results.append(ValidationResult(
                    rule_name="positive_prices",
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Found {negative_prices} non-positive prices",
                    details={"negative_count": negative_prices},
                    timestamp=datetime.now()
                ))
            
            # Price change validation (no more than 50% in 1 hour)
            if 'return_1h' in df.columns:
                extreme_changes = (np.abs(df['return_1h']) > 0.5).sum()
                if extreme_changes > 0:
                    results.append(ValidationResult(
                        rule_name="extreme_price_changes",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Found {extreme_changes} extreme price changes (>50%)",
                        details={"extreme_count": extreme_changes},
                        timestamp=datetime.now()
                    ))
        
        # Volume validation
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            zero_volume_pct = (zero_volume / len(df)) * 100
            
            if zero_volume_pct > 5.0:  # More than 5% zero volume is suspicious
                results.append(ValidationResult(
                    rule_name="zero_volume",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"High zero volume percentage: {zero_volume_pct:.2f}%",
                    details={"zero_volume_count": zero_volume, "zero_volume_percentage": zero_volume_pct},
                    timestamp=datetime.now()
                ))
        
        # Technical indicator validation
        if 'rsi_14' in df.columns:
            invalid_rsi = ((df['rsi_14'] < 0) | (df['rsi_14'] > 100)).sum()
            if invalid_rsi > 0:
                results.append(ValidationResult(
                    rule_name="rsi_bounds",
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"RSI values outside 0-100 range: {invalid_rsi} records",
                    details={"invalid_rsi_count": invalid_rsi},
                    timestamp=datetime.now()
                ))
        
        return results
    
    def validate_temporal_consistency(self, df: pd.DataFrame, symbol: str) -> List[ValidationResult]:
        """Validate temporal aspects of the data"""
        results = []
        
        if hasattr(df.index, 'to_series'):
            time_diffs = df.index.to_series().diff().dropna()
            
            # Check for regular intervals (expecting hourly data)
            expected_interval = timedelta(hours=1)
            tolerance = timedelta(minutes=5)
            
            irregular_intervals = ((time_diffs < expected_interval - tolerance) | 
                                 (time_diffs > expected_interval + tolerance)).sum()
            
            if irregular_intervals > 0:
                results.append(ValidationResult(
                    rule_name="temporal_regularity",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Found {irregular_intervals} irregular time intervals",
                    details={"irregular_count": irregular_intervals, "expected_interval": "1 hour"},
                    timestamp=datetime.now()
                ))
            
            # Check for data freshness
            if not df.empty:
                latest_time = df.index.max()
                time_since_latest = datetime.now() - latest_time.to_pydatetime()
                
                if time_since_latest > timedelta(hours=6):  # Data older than 6 hours
                    results.append(ValidationResult(
                        rule_name="data_freshness",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Data is not fresh: latest data is {time_since_latest} old",
                        details={"latest_timestamp": latest_time.isoformat(), "hours_old": time_since_latest.total_seconds() / 3600},
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def validate_statistical_drift(self, current_df: pd.DataFrame, reference_stats: Dict, symbol: str) -> List[ValidationResult]:
        """Detect statistical drift from reference statistics"""
        results = []
        
        if not reference_stats:
            return results
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_stats.get('numeric_stats', {}):
                ref_stats = reference_stats['numeric_stats'][col]
                current_mean = current_df[col].mean()
                current_std = current_df[col].std()
                
                # Mean drift check (more than 2 standard deviations)
                if not pd.isna(current_mean) and not pd.isna(ref_stats.get('mean')):
                    mean_change = abs(current_mean - ref_stats['mean'])
                    ref_std = ref_stats.get('std', 1.0)
                    
                    if mean_change > 2 * ref_std:
                        results.append(ValidationResult(
                            rule_name="statistical_drift_mean",
                            severity=ValidationSeverity.WARNING,
                            passed=False,
                            message=f"Mean drift detected in {col}: {mean_change:.4f} change",
                            details={
                                "column": col, 
                                "current_mean": current_mean,
                                "reference_mean": ref_stats['mean'],
                                "change": mean_change
                            },
                            timestamp=datetime.now()
                        ))
        
        return results
    
    def run_validation_suite(self, df: pd.DataFrame, dataset_name: str, symbol: str = None, 
                        reference_stats: Dict = None, dataset_type: str = 'features') -> Dict[str, Any]:
        """Run complete validation suite with dataset type awareness"""
        
        all_results = []
        
        # Define validation rules based on dataset type
        if dataset_type == 'raw_data' or 'raw_data' in dataset_name:
            schema_rules = {
                'required_columns': ['close_price', 'volume', 'open_time'],
                'dtypes': {'close_price': 'float', 'volume': 'float'},
                'min_rows': 100
            }
            quality_rules = {
                'max_null_percentage': 5.0,
                'max_duplicate_percentage': 2.0,
                'outlier_z_threshold': 4.0,
                'max_outlier_percentage': 1.0
            }
        elif dataset_type == 'targets' or 'targets' in dataset_name:
            # NEW: Validation rules for target datasets
            schema_rules = {
                'required_columns': ['target_return_4steps'],  # At least one target should exist
                'min_rows': 100
            }
            quality_rules = {
                'max_null_percentage': 30.0,  # Targets can have more nulls due to forward-looking nature
                'max_duplicate_percentage': 5.0,
                'outlier_z_threshold': 3.0,
                'max_outlier_percentage': 5.0
            }
        else:  # features
            schema_rules = {
                'required_columns': ['close_price'],  # Basic price data should be present
                'min_rows': 100
            }
            quality_rules = {
                'max_null_percentage': 15.0,  # Features can have more nulls
                'max_duplicate_percentage': 5.0,
                'outlier_z_threshold': 3.0,
                'max_outlier_percentage': 3.0
            }
        
        # Run standard validations
        all_results.extend(self.validate_schema(df, schema_rules, dataset_name))
        all_results.extend(self.validate_data_quality(df, quality_rules, dataset_name))
        
        if symbol:
            all_results.extend(self.validate_crypto_specific(df, symbol))
            all_results.extend(self.validate_temporal_consistency(df, symbol))
        
        # NEW: Run target-specific validation if this is a targets dataset
        if dataset_type == 'targets' or 'targets' in dataset_name:
            all_results.extend(self.validate_target_data(df, symbol))
        
        if reference_stats:
            all_results.extend(self.validate_statistical_drift(df, reference_stats, symbol))
        
        # Store results in database
        self._store_validation_results(all_results, dataset_name, symbol)
        
        # Generate summary
        summary = self._generate_validation_summary(all_results)
        
        return {
            'dataset_name': dataset_name,
            'dataset_type': dataset_type,
            'symbol': symbol,
            'validation_time': datetime.now().isoformat(),
            'total_rules': len(all_results),
            'passed_rules': len([r for r in all_results if r.passed]),
            'failed_rules': len([r for r in all_results if not r.passed]),
            'critical_failures': len([r for r in all_results if not r.passed and r.severity == ValidationSeverity.CRITICAL]),
            'warnings': len([r for r in all_results if not r.passed and r.severity == ValidationSeverity.WARNING]),
            'summary': summary,
            'results': [
                {
                    'rule_name': r.rule_name,
                    'severity': r.severity.value,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details
                } for r in all_results
            ]
        }
    
    def _store_validation_results(self, results: List[ValidationResult], dataset_name: str, symbol: str):
        """Store validation results in database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    for result in results:
                        serializable_details = self._make_json_serializable(result.details)
                        cur.execute("""
                            INSERT INTO validation_results 
                            (dataset_name, rule_name, severity, passed, message, details, symbol)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            dataset_name, result.rule_name, result.severity.value,
                            result.passed, result.message, json.dumps(serializable_details), symbol
                        ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store validation results: {str(e)}")

    def _make_json_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
        
    def _generate_validation_summary(self, results: List[ValidationResult]) -> str:
        """Generate human-readable validation summary"""
        total = len(results)
        passed = len([r for r in results if r.passed])
        critical_failed = len([r for r in results if not r.passed and r.severity == ValidationSeverity.CRITICAL])
        
        if critical_failed > 0:
            return f"CRITICAL: {critical_failed} critical validation failures out of {total} checks"
        elif passed == total:
            return f"PASSED: All {total} validation checks passed"
        else:
            failed = total - passed
            return f"WARNING: {failed} validation warnings out of {total} checks"
    
    def get_validation_history(self, dataset_name: str = None, days: int = 7) -> List[Dict]:
        """Get validation history for analysis"""
        query = """
        SELECT dataset_name, symbol, validation_time, severity, 
               COUNT(*) as rule_count,
               SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_count
        FROM validation_results 
        WHERE validation_time >= NOW() - INTERVAL '%s days'
        """
        params = [days]
        
        if dataset_name:
            query += " AND dataset_name = %s"
            params.append(dataset_name)
        
        query += " GROUP BY dataset_name, symbol, validation_time, severity ORDER BY validation_time DESC"
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                df = pd.read_sql(query, conn, params=params)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Failed to get validation history: {str(e)}")
            return []

    def validate_target_data(self, df: pd.DataFrame, symbol: str) -> List[ValidationResult]:
        """Validate multi-horizon target data"""
        results = []
        
        # Check for target columns with proper naming
        expected_target_prefixes = ['target_return_', 'target_direction_', 'target_volatility_', 'target_vol_regime_']
        target_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in expected_target_prefixes)]
        
        if not target_columns:
            results.append(ValidationResult(
                rule_name="target_columns_exist",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message="No target columns found in dataset",
                details={"expected_prefixes": expected_target_prefixes},
                timestamp=datetime.now()
            ))
            return results
        
        # Validate target column naming convention
        valid_horizons = ['1steps', '4steps', '16steps', '96steps']
        for col in target_columns:
            horizon_found = any(horizon in col for horizon in valid_horizons)
            if not horizon_found:
                results.append(ValidationResult(
                    rule_name="target_naming_convention",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Target column {col} doesn't follow naming convention",
                    details={"column": col, "expected_horizons": valid_horizons},
                    timestamp=datetime.now()
                ))
        
        # Validate return targets (should be reasonable ranges)
        return_columns = [col for col in target_columns if 'target_return_' in col]
        for col in return_columns:
            if col in df.columns and not df[col].empty:
                # Returns should typically be between -50% and +50% for crypto
                extreme_returns = ((df[col] < -0.5) | (df[col] > 0.5)).sum()
                if extreme_returns > 0:
                    extreme_pct = (extreme_returns / len(df)) * 100
                    severity = ValidationSeverity.CRITICAL if extreme_pct > 1.0 else ValidationSeverity.WARNING
                    
                    results.append(ValidationResult(
                        rule_name="return_targets_range",
                        severity=severity,
                        passed=False,
                        message=f"Column {col} has {extreme_returns} extreme returns (>50% or <-50%)",
                        details={"column": col, "extreme_count": extreme_returns, "extreme_percentage": extreme_pct},
                        timestamp=datetime.now()
                    ))
        
        # Validate direction targets (should be 0 or 1 for binary)
        direction_columns = [col for col in target_columns if 'target_direction_' in col and 'multi' not in col]
        for col in direction_columns:
            if col in df.columns and not df[col].empty:
                unique_values = df[col].dropna().unique()
                if not all(val in [0, 1] for val in unique_values):
                    results.append(ValidationResult(
                        rule_name="direction_targets_binary",
                        severity=ValidationSeverity.CRITICAL,
                        passed=False,
                        message=f"Binary direction column {col} contains non-binary values",
                        details={"column": col, "unique_values": unique_values.tolist()},
                        timestamp=datetime.now()
                    ))
        
        # Validate multi-class direction targets (should be 0-4)
        direction_multi_columns = [col for col in target_columns if 'target_direction_multi_' in col]
        for col in direction_multi_columns:
            if col in df.columns and not df[col].empty:
                unique_values = df[col].dropna().unique()
                expected_values = [0, 1, 2, 3, 4]  # strong_down, down, neutral, up, strong_up
                if not all(val in expected_values for val in unique_values):
                    results.append(ValidationResult(
                        rule_name="direction_multi_targets_range",
                        severity=ValidationSeverity.CRITICAL,
                        passed=False,
                        message=f"Multi-class direction column {col} contains invalid values",
                        details={"column": col, "unique_values": unique_values.tolist(), "expected": expected_values},
                        timestamp=datetime.now()
                    ))
        
        # Validate volatility targets (should be positive)
        volatility_columns = [col for col in target_columns if 'target_volatility_' in col]
        for col in volatility_columns:
            if col in df.columns and not df[col].empty:
                negative_vol = (df[col] < 0).sum()
                if negative_vol > 0:
                    results.append(ValidationResult(
                        rule_name="volatility_targets_positive",
                        severity=ValidationSeverity.CRITICAL,
                        passed=False,
                        message=f"Volatility column {col} contains negative values",
                        details={"column": col, "negative_count": negative_vol},
                        timestamp=datetime.now()
                    ))
        
        # Check for sufficient non-null targets across horizons
        for horizon in valid_horizons:
            horizon_columns = [col for col in target_columns if horizon in col]
            if horizon_columns:
                for col in horizon_columns:
                    if col in df.columns:
                        null_pct = (df[col].isnull().sum() / len(df)) * 100
                        max_null_pct = 30.0  # Allow more nulls in targets due to forward-looking nature
                        
                        if null_pct > max_null_pct:
                            results.append(ValidationResult(
                                rule_name="target_null_percentage",
                                severity=ValidationSeverity.WARNING,
                                passed=False,
                                message=f"Target column {col} has high null percentage: {null_pct:.2f}%",
                                details={"column": col, "null_percentage": null_pct, "threshold": max_null_pct},
                                timestamp=datetime.now()
                            ))
        
        return results

    # ADD method to be called from model training pipeline:
    def validate_model_training_readiness(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                        task_configs: List[Dict], symbol: str) -> Dict[str, Any]:
        """Validate readiness for model training across multiple tasks"""
        all_results = []
        
        # Run alignment validation for each task
        for task_config in task_configs:
            task_results = self.validate_training_data_alignment(features_df, targets_df, task_config, symbol)
            all_results.extend(task_results)
        
        # Store results
        dataset_name = f"training_readiness_{symbol}"
        self._store_validation_results(all_results, dataset_name, symbol)
        
        # Generate summary
        summary = self._generate_validation_summary(all_results)
        
        return {
            'dataset_name': dataset_name,
            'symbol': symbol,
            'validation_time': datetime.now().isoformat(),
            'tasks_validated': len(task_configs),
            'total_rules': len(all_results),
            'passed_rules': len([r for r in all_results if r.passed]),
            'failed_rules': len([r for r in all_results if not r.passed]),
            'critical_failures': len([r for r in all_results if not r.passed and r.severity == ValidationSeverity.CRITICAL]),
            'warnings': len([r for r in all_results if not r.passed and r.severity == ValidationSeverity.WARNING]),
            'summary': summary,
            'ready_for_training': len([r for r in all_results if not r.passed and r.severity == ValidationSeverity.CRITICAL]) == 0
        }

    def validate_training_data_alignment(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                                    task_config: Dict, symbol: str) -> List[ValidationResult]:
        """Validate alignment between features and targets for training"""
        results = []
        
        target_col = task_config['target_col']
        
        # Check if target column exists
        if target_col not in targets_df.columns:
            results.append(ValidationResult(
                rule_name="target_column_exists",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Target column {target_col} not found in targets dataset",
                details={"target_col": target_col, "available_columns": list(targets_df.columns)},
                timestamp=datetime.now()
            ))
            return results
        
        # Check temporal alignment
        features_start = features_df.index.min()
        features_end = features_df.index.max()
        targets_start = targets_df.index.min()
        targets_end = targets_df.index.max()
        
        # Features should start before or at the same time as targets
        if features_start > targets_start:
            results.append(ValidationResult(
                rule_name="temporal_alignment_start",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="Features start after targets - potential alignment issue",
                details={"features_start": features_start.isoformat(), "targets_start": targets_start.isoformat()},
                timestamp=datetime.now()
            ))
        
        # Check overlap period
        common_index = features_df.index.intersection(targets_df.index)
        overlap_pct = (len(common_index) / max(len(features_df), len(targets_df))) * 100
        
        if overlap_pct < 80.0:  # Less than 80% overlap is concerning
            results.append(ValidationResult(
                rule_name="features_targets_overlap",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Low overlap between features and targets: {overlap_pct:.1f}%",
                details={"overlap_percentage": overlap_pct, "common_periods": len(common_index)},
                timestamp=datetime.now()
            ))
        
        # Check for sufficient non-null aligned data
        if len(common_index) > 0:
            aligned_target = targets_df.loc[common_index, target_col]
            valid_target_pct = (aligned_target.notna().sum() / len(aligned_target)) * 100
            
            if valid_target_pct < 70.0:  # Less than 70% valid targets
                results.append(ValidationResult(
                    rule_name="aligned_target_completeness",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Low valid target percentage in aligned data: {valid_target_pct:.1f}%",
                    details={"valid_target_percentage": valid_target_pct, "target_column": target_col},
                    timestamp=datetime.now()
                ))
        
        return results

# Integration functions
def validate_raw_data(db_config: Dict, symbols: List[str]) -> Dict[str, Any]:
    """Validate raw crypto data for all symbols"""
    validator = DataValidationSuite(db_config)
    results = {}
    
    for symbol in symbols:
        # Get raw data from database
        query = """
        SELECT open_time, close_price, high_price, low_price, open_price, volume, quote_volume
        FROM crypto_data 
        WHERE symbol = %s 
        ORDER BY open_time
        """
        
        with psycopg2.connect(**db_config) as conn:
            df = pd.read_sql(query, conn, params=[symbol])
        
        if df.empty:
            continue
            
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('datetime')
        
        # Run validation
        validation_result = validator.run_validation_suite(
            df, f"raw_data_{symbol}", symbol
        )
        results[symbol] = validation_result
    
    return results

def validate_feature_and_target_data(db_config: Dict, pipeline_results: Dict) -> Dict[str, Any]:
    """Validate both engineered features and targets"""
    validator = DataValidationSuite(db_config)
    results = {}
    
    minio_client = Minio('minio:9000', access_key='admin', secret_key='admin123', secure=False)
    
    # Validate features
    for symbol, storage_path in pipeline_results['storage_paths'].items():
        try:
            # Load features from MinIO
            response = minio_client.get_object('crypto-features', storage_path)
            df = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()
            
            # Run feature validation
            validation_result = validator.run_validation_suite(
                df, f"features_{symbol}", symbol, dataset_type='features'
            )
            results[f"{symbol}_features"] = validation_result
            
        except Exception as e:
            logging.getLogger().error(f"Failed to validate features for {symbol}: {e}")
            continue
    
    # Validate targets (if they exist)
    if 'target_paths' in pipeline_results:
        for symbol, target_path in pipeline_results['target_paths'].items():
            try:
                # Load targets from MinIO
                response = minio_client.get_object('crypto-features', target_path)
                df = pd.read_parquet(BytesIO(response.read()))
                response.close()
                response.release_conn()
                
                # Run target validation
                validation_result = validator.run_validation_suite(
                    df, f"targets_{symbol}", symbol, dataset_type='targets'
                )
                results[f"{symbol}_targets"] = validation_result
                
            except Exception as e:
                logging.getLogger().error(f"Failed to validate targets for {symbol}: {e}")
                continue
    
    return results