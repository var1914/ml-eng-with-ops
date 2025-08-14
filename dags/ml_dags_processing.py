from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

import json
import logging

from data_quality import DataQualityAssessment
from feature_eng import FeatureEngineeringPipeline
from viz import DataQualityMonitoring, FeatureEngineeringMonitoring
from data_versioning import DVCDataVersioning, dvc_version_raw_data, dvc_version_features_data

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
        
        if not quality_results or not feature_results:
            raise ValueError("Missing required results from upstream tasks")
        
        # Initialize monitoring components
        dq_monitor = DataQualityMonitoring()
        fe_monitor = FeatureEngineeringMonitoring()

        # Log metrics
        dq_monitor.log_data_quality_metrics(quality_results, SYMBOLS)
        fe_monitor.log_feature_engineering_metrics(feature_results, SYMBOLS)
        
        # Log versioning info if available
        if versioning_summary:
            logger.info(f"Versioning summary: {versioning_summary}")
        
        logger.info("Successfully logged monitoring metrics")
        
    except Exception as e:
        logger.error(f"Logging monitoring metrics failed: {str(e)}")
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
    schedule='@hourly',
    catchup=False,
) 

# Define tasks with proper naming and dependencies
data_quality_task = PythonOperator(
    task_id='data_quality_assessment',
    python_callable=run_data_quality_assessment,
    dag=dag,
    retries=1,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag,
)

# NEW: DVC Data versioning task
data_versioning_task = PythonOperator(
    task_id='dvc_data_versioning',
    python_callable=run_dvc_data_versioning,
    dag=dag,
)

monitoring_task = PythonOperator(
    task_id='logging_monitoring',
    python_callable=log_monitoring_metrics,
    dag=dag,
)

# Define task dependencies - updated to include versioning
data_quality_task >> feature_engineering_task >> data_versioning_task >> monitoring_task