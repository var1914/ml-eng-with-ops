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
                           reference_stats: Dict = None) -> Dict[str, Any]:
        """Run complete validation suite"""
        
        all_results = []
        
        # Define validation rules based on dataset type
        if 'raw_data' in dataset_name:
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
        else:  # features
            schema_rules = {
                'required_columns': ['close_price', 'rsi_14', 'volume'],
                'min_rows': 100
            }
            quality_rules = {
                'max_null_percentage': 15.0,  # Features can have more nulls
                'max_duplicate_percentage': 5.0,
                'outlier_z_threshold': 3.0,
                'max_outlier_percentage': 3.0
            }
        
        # Run validations
        all_results.extend(self.validate_schema(df, schema_rules, dataset_name))
        all_results.extend(self.validate_data_quality(df, quality_rules, dataset_name))
        
        if symbol:
            all_results.extend(self.validate_crypto_specific(df, symbol))
            all_results.extend(self.validate_temporal_consistency(df, symbol))
        
        if reference_stats:
            all_results.extend(self.validate_statistical_drift(df, reference_stats, symbol))
        
        # Store results in database
        self._store_validation_results(all_results, dataset_name, symbol)
        
        # Generate summary
        summary = self._generate_validation_summary(all_results)
        
        return {
            'dataset_name': dataset_name,
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

def validate_feature_data(db_config: Dict, feature_results: Dict) -> Dict[str, Any]:
    """Validate engineered features"""
    validator = DataValidationSuite(db_config)
    results = {}
    
    minio_client = Minio('minio:9000', access_key='admin', secret_key='admin123', secure=False)
    
    for symbol, storage_path in feature_results['storage_paths'].items():
        try:
            # Load features from MinIO
            response = minio_client.get_object('crypto-features', storage_path)
            df = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()
            
            # Get reference statistics for drift detection
            reference_stats = None  # Could load from previous validation
            
            # Run validation
            validation_result = validator.run_validation_suite(
                df, f"features_{symbol}", symbol, reference_stats
            )
            results[symbol] = validation_result
            
        except Exception as e:
            logging.getLogger().error(f"Failed to validate features for {symbol}: {e}")
            continue
    
    return results