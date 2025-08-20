from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

import json
import logging

from data_quality import DataQualityAssessment
from feature_eng import FeatureEngineeringPipeline
from viz import DataQualityMonitoring, FeatureEngineeringMonitoring, ValidationMonitoring, ModelTrainingMonitoring, ModelLifecycleMonitoring, ComprehensiveMonitoring
from data_versioning import DVCDataVersioning, dvc_version_raw_data, dvc_version_features_data
from automated_data_validation import validate_raw_data, validate_feature_data
from model_training import ModelTrainingPipeline

DB_CONFIG = {
    "dbname": "postgres",
    "user": "varunrajput", 
    "password": "yourpassword",
    "host": "host.docker.internal",
    "port": "5432"
}

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT']
BASE_URL = "https://api.binance.com/api/v3/klines"

logger = logging.getLogger(__name__)

def run_raw_data_validation(**context):
    """Validate raw data quality"""
    try:
        
        logger.info("Starting raw data validation...")
        validation_results = validate_raw_data(DB_CONFIG, SYMBOLS)
        
        # Check for critical failures
        critical_failures = []
        total_warnings = 0
        
        for symbol, result in validation_results.items():
            if result['critical_failures'] > 0:
                critical_failures.append(f"{symbol}: {result['critical_failures']} critical issues")
            total_warnings += result['warnings']
        
        # Log summary
        logger.info(f"Raw data validation completed for {len(validation_results)} symbols")
        logger.info(f"Total warnings: {total_warnings}")
        
        if critical_failures:
            logger.error(f"Critical validation failures: {critical_failures}")
            # You can choose to fail the task or just warn
            # raise ValueError(f"Critical data validation failures: {critical_failures}")
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='raw_data_validation', value=validation_results)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Raw data validation failed: {str(e)}")
        raise

def run_feature_data_validation(**context):
    """Validate feature data quality"""
    try:
        
        # Get feature results from upstream task
        feature_results = context['task_instance'].xcom_pull(
            task_ids='feature_engineering', 
            key='feature_results'
        )
        
        if not feature_results:
            raise ValueError("No feature results found for validation")
        
        logger.info("Starting feature data validation...")
        validation_results = validate_feature_data(DB_CONFIG, feature_results)
        
        # Check for critical failures
        critical_failures = []
        total_warnings = 0
        
        for symbol, result in validation_results.items():
            if result['critical_failures'] > 0:
                critical_failures.append(f"{symbol}: {result['critical_failures']} critical issues")
            total_warnings += result['warnings']
        
        # Log summary
        logger.info(f"Feature validation completed for {len(validation_results)} symbols")
        logger.info(f"Total warnings: {total_warnings}")
        
        if critical_failures:
            logger.error(f"Critical feature validation failures: {critical_failures}")
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='feature_data_validation', value=validation_results)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Feature data validation failed: {str(e)}")
        raise


def run_data_quality_assessment(**context):
    """Run data quality assessment and return results via XCom"""
    try:
        assessor = DataQualityAssessment(DB_CONFIG)
        quality_results = assessor.run_quality_assessment(SYMBOLS)
        
        logger.info(f"Data quality assessment completed for {len(SYMBOLS)} symbols")
        
        # Push results to XCom for downstream tasks
        context['task_instance'].xcom_push(key='quality_results', value=quality_results)
        return quality_results
        
    except Exception as e:
        logger.error(f"Data quality assessment failed: {str(e)}")
        raise

def run_feature_engineering(**context):
    """Run feature engineering pipeline and return results via XCom"""
    try:
        pipeline = FeatureEngineeringPipeline(DB_CONFIG)
        feature_results = pipeline.run_feature_pipeline(SYMBOLS)
        
        logger.info(f"Feature engineering completed for {len(SYMBOLS)} symbols")
        
        # Push results to XCom for downstream tasks
        context['task_instance'].xcom_push(key='feature_results', value=feature_results)
        return feature_results
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

def run_dvc_data_versioning(**context):
    """Version both raw data and engineered features using DVC"""
    try:
        logger.info("Starting DVC data versioning...")
        
        # Version raw data for all symbols using DVC
        raw_version_ids = dvc_version_raw_data(DB_CONFIG, SYMBOLS)
        logger.info(f"DVC raw data versioning completed: {len(raw_version_ids)} versions created")
        
        # Get feature results from upstream task
        feature_results = context['task_instance'].xcom_pull(
            task_ids='feature_engineering', 
            key='feature_results'
        )
        
        feature_version_ids = []
        if feature_results:
            feature_version_ids = dvc_version_features_data(DB_CONFIG, feature_results)
            logger.info(f"DVC feature data versioning completed: {len(feature_version_ids)} versions created")
        else:
            logger.warning("No feature results found for versioning")
        
        # Push DVC versioning summary to XCom
        versioning_summary = {
            'dvc_enabled': True,
            'raw_version_ids': raw_version_ids,
            'feature_version_ids': feature_version_ids,
            'raw_data_versioned': len(raw_version_ids),
            'feature_data_versioned': len(feature_version_ids),
            'timestamp': datetime.now().isoformat()
        }
        
        context['task_instance'].xcom_push(key='dvc_versioning_summary', value=versioning_summary)
        logger.info("DVC data versioning completed successfully")
        
    except Exception as e:
        logger.error(f"DVC data versioning failed: {str(e)}")
        raise

def run_model_training(**context):
    """Enhanced model training with MLflow registry integration"""    
    try:
        # Get DVC versioning results
        dvc_versioning_summary = context['task_instance'].xcom_pull(
            task_ids='dvc_data_versioning', 
            key='dvc_versioning_summary'
        )
        
        if not dvc_versioning_summary:
            raise ValueError("No DVC versioning summary found")
        
        # Initialize training pipeline
        trainer = ModelTrainingPipeline(DB_CONFIG)
        
        # Run training with MLflow integration
        training_results = trainer.run_training_pipeline(dvc_versioning_summary)
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='training_results', value=training_results)
        
        logger.info(f"Model training completed with MLflow integration for {len(training_results['symbols_trained'])} symbols")
        return training_results
        
    except Exception as e:
        logger.error(f"Model training with MLflow failed: {str(e)}")
        raise


def log_monitoring_metrics(**context):
    """Log monitoring metrics using results from upstream tasks"""
    try:
        # Pull results from upstream tasks via XCom
        quality_results = context['task_instance'].xcom_pull(
            task_ids='data_quality_assessment', 
            key='quality_results'
        )
        feature_results = context['task_instance'].xcom_pull(
            task_ids='feature_engineering', 
            key='feature_results'
        )
        versioning_summary = context['task_instance'].xcom_pull(
            task_ids='dvc_data_versioning', 
            key='dvc_versioning_summary'
        )
        training_results = context['task_instance'].xcom_pull(
            task_ids='model_training', 
            key='training_results'
        )
        # Pull validation results
        raw_validation = context['task_instance'].xcom_pull(
            task_ids='raw_data_validation', 
            key='raw_data_validation'
        )
        feature_validation = context['task_instance'].xcom_pull(
            task_ids='feature_data_validation', 
            key='feature_data_validation'
        )
        
        # Combine validation results
        validation_results = {
            'raw_validation': raw_validation,
            'feature_validation': feature_validation
        }
        
        if not quality_results or not feature_results:
            raise ValueError("Missing required results from upstream tasks")
        
        # Initialize monitoring components
        dq_monitor = DataQualityMonitoring()
        fe_monitor = FeatureEngineeringMonitoring()
        validation_monitor = ValidationMonitoring(db_config=DB_CONFIG)
        training_monitor = ModelTrainingMonitoring()
        lifecycle_monitor = ModelLifecycleMonitoring()
        comprehensive_monitor = ComprehensiveMonitoring(db_config=DB_CONFIG)

        # Log metrics
        dq_monitor.log_data_quality_metrics(quality_results, SYMBOLS)
        fe_monitor.log_feature_engineering_metrics(feature_results, SYMBOLS)
        validation_monitor.log_comprehensive_metrics(
            quality_results, feature_results, validation_results, SYMBOLS
        )
        training_monitor.log_model_training_metrics(training_results, SYMBOLS)
        # lifecycle_monitor.log_lifecycle_metrics(lifecycle_results, SYMBOLS)
        # comprehensive_monitor.log_comprehensive_metrics(
        #     quality_results, feature_results, training_results, 
        #     lifecycle_results, validation_results, SYMBOLS
        # )

        # Log summary
        total_critical = 0
        total_warnings = 0
        
        if raw_validation:
            total_critical += sum(r['critical_failures'] for r in raw_validation.values())
            total_warnings += sum(r['warnings'] for r in raw_validation.values())
        
        if feature_validation:
            total_critical += sum(r['critical_failures'] for r in feature_validation.values())
            total_warnings += sum(r['warnings'] for r in feature_validation.values())
        
        logger.info(f"Comprehensive monitoring completed:")
        logger.info(f"  - Quality assessment: {len(quality_results) if quality_results else 0} symbols")
        logger.info(f"  - Feature engineering: {len(feature_results['storage_paths']) if feature_results else 0} symbols")
        logger.info(f"  - DVC versioning: {versioning_summary['raw_data_versioned'] if versioning_summary else 0} versions")
        logger.info(f"  - Validation: {total_critical} critical failures, {total_warnings} warnings")
        
        if total_critical > 0:
            logger.warning(f"⚠️ {total_critical} critical validation failures detected!")
        
        logger.info("Successfully logged comprehensive monitoring metrics")
        
    except Exception as e:
        logger.error(f"Comprehensive monitoring failed: {str(e)}")
        raise


default_args = {
    'owner': 'varunrajput',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 7),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'crypto_ml_pipeline',
    description='Complete Crypto ML pipeline',
    default_args=default_args,
    schedule='@daily',
    catchup=False,
) 

# Define tasks with proper naming and dependencies
data_quality_task = PythonOperator(
    task_id='data_quality_assessment',
    python_callable=run_data_quality_assessment,
    dag=dag,
    retries=1,
)

raw_data_validation_task = PythonOperator(
    task_id='raw_data_validation',
    python_callable=run_raw_data_validation,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag,
)

feature_data_validation_task = PythonOperator(
    task_id='feature_data_validation',
    python_callable=run_feature_data_validation,
    dag=dag,
)

# NEW: DVC Data versioning task
data_versioning_task = PythonOperator(
    task_id='dvc_data_versioning',
    python_callable=run_dvc_data_versioning,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

monitoring_task = PythonOperator(
    task_id='logging_monitoring',
    python_callable=log_monitoring_metrics,
    dag=dag,
)

# Define task dependencies - updated to include versioning
data_quality_task >> raw_data_validation_task >> feature_engineering_task >> feature_data_validation_task >> data_versioning_task >> model_training_task >> monitoring_task