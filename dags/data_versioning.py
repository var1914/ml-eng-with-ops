import os
import json
import subprocess
import psycopg2
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import yaml
import shutil
from typing import Dict, List, Optional, Tuple
import hashlib
from io import BytesIO


from minio import Minio
from minio.error import S3Error

# DVC Configuration
DVC_CONFIG = {
    'data_dir': '/opt/airflow/data',  # Adjust based on your Airflow setup
    'remote_name': 'crypto_storage',
    'remote_url': 's3://crypto-data-versions',
    'remote_config': {
        'endpointurl': 'http://minio:9000',
        'access_key_id': 'admin',
        'secret_access_key': 'admin123'
    }
}

DB_CONFIG = {
    "dbname": "postgres",
    "user": "varunrajput", 
    "password": "yourpassword",
    "host": "host.docker.internal",
    "port": "5432"
}

# MinIO Configuration
MINIO_CONFIG = {
    'endpoint': 'minio:9000',  # Adjust based on your Helm setup
    'access_key': 'admin',  # Change these in production!
    'secret_key': 'admin123',
    'secure': False  # Set to True if using HTTPS
}

BUCKET_NAME = 'crypto-features'

class DVCDataVersioning:
    """Professional data versioning using DVC with S3/MinIO backend"""
    
    def __init__(self, db_config, data_dir=None, git_repo_path=None):
        self.logger = logging.getLogger("dvc_data_versioning")
        self.db_config = db_config
        self.data_dir = Path(data_dir or DVC_CONFIG['data_dir'])
        self.git_repo_path = git_repo_path or "/opt/airflow/dags"  # Your DAGs directory
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "features").mkdir(exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)
        
        self._setup_dvc()
        self._create_metadata_table()
        self.minio_client = self._get_minio_client()
    
    def _get_minio_client(self):
        try:
            client = Minio(
                MINIO_CONFIG['endpoint'],
                access_key=MINIO_CONFIG['access_key'],
                secret_key=MINIO_CONFIG['secret_key'],
                secure=MINIO_CONFIG['secure']
            )
            self.logger.info(f"MinIO Client Initialised")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialise MinIO Client: {str(e)}")
            raise

    def _setup_dvc(self):
        """Initialize DVC repository and remote storage"""
        os.chdir(self.git_repo_path)
        
        try:
            # Initialize DVC if not already done
            if not os.path.exists('.dvc'):
                self._run_command(['dvc', 'init', '--no-scm'])
                self.logger.info("DVC repository initialized")
            
            # Configure remote storage
            self._setup_remote_storage()
            
        except Exception as e:
            self.logger.error(f"DVC setup failed: {str(e)}")
            raise

    def _setup_remote_storage(self):
        """Configure DVC remote storage (S3 or MinIO)"""
        try:
            # Check if remote already exists
            result = subprocess.run(['dvc', 'remote', 'list'], 
                                  capture_output=True, text=True)
            
            if DVC_CONFIG['remote_name'] not in result.stdout:
                # Add remote storage
                self._run_command([
                    'dvc', 'remote', 'add', '-d', 
                    DVC_CONFIG['remote_name'], 
                    DVC_CONFIG['remote_url']
                ])
                
                # Configure MinIO if using MinIO instead of S3
                if 'minio' in DVC_CONFIG['remote_url'] or 'localhost' in DVC_CONFIG['remote_url']:
                    self._run_command([
                        'dvc', 'remote', 'modify', DVC_CONFIG['remote_name'], 
                        'endpointurl', 'http://minio:9000'
                    ])
                    self._run_command([
                        'dvc', 'remote', 'modify', DVC_CONFIG['remote_name'], 
                        'access_key_id', 'admin'
                    ])
                    self._run_command([
                        'dvc', 'remote', 'modify', DVC_CONFIG['remote_name'], 
                        'secret_access_key', 'admin123'
                    ])
                
                self.logger.info(f"DVC remote '{DVC_CONFIG['remote_name']}' configured")
            
        except Exception as e:
            self.logger.error(f"Remote storage setup failed: {str(e)}")
            raise
    
    def _run_command(self, cmd: List[str], check=True):
        """Execute shell command with error handling"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            if result.stdout:
                self.logger.debug(f"Command output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}")
            self.logger.error(f"Error: {e.stderr}")
            raise
    
    def _create_metadata_table(self):
        """Create table to track DVC data versions with enhanced metadata"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS dvc_data_versions (
            version_id VARCHAR(64) PRIMARY KEY,
            dataset_name VARCHAR(100) NOT NULL,
            file_path VARCHAR(255) NOT NULL,
            dvc_file_path VARCHAR(255) NOT NULL,
            git_commit_hash VARCHAR(40),
            creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            record_count INTEGER,
            file_size_mb FLOAT,
            data_schema JSONB,
            metadata JSONB,
            parent_version VARCHAR(64),
            tags TEXT[],
            FOREIGN KEY (parent_version) REFERENCES dvc_data_versions(version_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_dvc_dataset_name ON dvc_data_versions(dataset_name);
        CREATE INDEX IF NOT EXISTS idx_dvc_creation_time ON dvc_data_versions(creation_time);
        CREATE INDEX IF NOT EXISTS idx_dvc_tags ON dvc_data_versions USING GIN(tags);
        """
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_query)
                conn.commit()
            self.logger.info("DVC data versions table created/verified")
        except Exception as e:
            self.logger.error(f"Failed to create DVC versions table: {str(e)}")
            raise
    
    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash"""
        try:
            result = self._run_command(['git', 'rev-parse', 'HEAD'])
            return result.stdout.strip()
        except:
            return "no_git_commit"
    
    def _calculate_schema_hash(self, df: pd.DataFrame) -> Dict:
        """Calculate schema information for data validation"""
        schema = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            schema['numeric_stats'] = {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'null_count': int(df[col].isnull().sum())
                } for col in numeric_cols
            }
        
        return schema
    
    def create_data_version(self, 
                          dataset_name: str, 
                          df: pd.DataFrame, 
                          metadata: Optional[Dict] = None,
                          tags: Optional[List[str]] = None,
                          parent_version: Optional[str] = None) -> str:
        """Create a new DVC-tracked data version"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_hash = hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()[:8]
        version_id = f"{dataset_name}_{timestamp}_{data_hash}"
        
        # Determine file paths
        if 'raw_data' in dataset_name:
            file_path = self.data_dir / "raw" / f"{dataset_name}.parquet"
        elif 'features' in dataset_name:
            file_path = self.data_dir / "features" / f"{dataset_name}.parquet"
        else:
            file_path = self.data_dir / f"{dataset_name}.parquet"
        
        dvc_file_path = f"{file_path}.dvc"
        
        try:
            # Save dataframe to file
            df.to_parquet(file_path, index=True)
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            
            # Add to DVC tracking
            os.chdir(self.git_repo_path)
            self._run_command(['dvc', 'add', str(file_path)])
            
            # Calculate schema
            schema = self._calculate_schema_hash(df)
            
            # Get git commit hash
            git_commit = self._get_git_commit_hash()
            
            # Save metadata to database
            insert_query = """
            INSERT INTO dvc_data_versions 
            (version_id, dataset_name, file_path, dvc_file_path, git_commit_hash,
             record_count, file_size_mb, data_schema, metadata, parent_version, tags)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (
                        version_id, dataset_name, str(file_path), dvc_file_path,
                        git_commit, len(df), file_size_mb, json.dumps(schema),
                        json.dumps(metadata) if metadata else None, parent_version,
                        tags or []
                    ))
                conn.commit()
            
            # Commit DVC files to git
            self._run_command(['git', 'add', dvc_file_path, '.dvc/config'])
            self._run_command(['git', 'commit', '-m', f'Add data version: {version_id}'])
            
            # Push to DVC remote
            self._run_command(['dvc', 'push'])
            
            self.logger.info(f"Created DVC data version: {version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create DVC data version: {str(e)}")
            # Cleanup on failure
            if file_path.exists():
                file_path.unlink()
            if Path(dvc_file_path).exists():
                Path(dvc_file_path).unlink()
            raise
    
    def load_data_version(self, version_id: str) -> Dict:
        """Load a specific DVC data version"""
        query = """
        SELECT file_path, dataset_name, data_schema, metadata, git_commit_hash
        FROM dvc_data_versions 
        WHERE version_id = %s
        """
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (version_id,))
                    result = cur.fetchone()
                    
                    if not result:
                        raise ValueError(f"Version {version_id} not found")
                    
                    file_path, dataset_name, schema, metadata, git_commit = result
            
            # Ensure we have the right git commit
            if git_commit != "no_git_commit":
                try:
                    self._run_command(['git', 'checkout', git_commit])
                except:
                    self.logger.warning(f"Could not checkout git commit {git_commit}")
            
            # Pull data from DVC remote if not present locally
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self._run_command(['dvc', 'pull', str(file_path)])
            
            # Load the data
            df = pd.read_parquet(file_path_obj)
            
            self.logger.info(f"Loaded DVC data version: {version_id}")
            return {
                'data': df,
                'metadata': json.loads(metadata) if metadata else None,
                'schema': json.loads(schema) if schema else None,
                'dataset_name': dataset_name,
                'git_commit': git_commit
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load DVC data version {version_id}: {str(e)}")
            raise
    
    def list_versions(self, 
                     dataset_name: Optional[str] = None, 
                     tags: Optional[List[str]] = None,
                     limit: int = 10) -> List[Dict]:
        """List available DVC data versions with filtering"""
        base_query = """
        SELECT version_id, dataset_name, creation_time, record_count, 
               file_size_mb, tags, metadata, git_commit_hash
        FROM dvc_data_versions 
        """
        
        conditions = []
        params = []
        
        if dataset_name:
            conditions.append("dataset_name = %s")
            params.append(dataset_name)
        
        if tags:
            conditions.append("tags && %s")
            params.append(tags)
        
        if conditions:
            base_query += "WHERE " + " AND ".join(conditions)
        
        base_query += " ORDER BY creation_time DESC LIMIT %s"
        params.append(limit)
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                df = pd.read_sql(base_query, conn, params=params)
            
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Failed to list DVC versions: {str(e)}")
            raise
    
    def get_latest_version(self, dataset_name: str) -> Dict:
        """Get the latest version of a dataset"""
        query = """
        SELECT version_id FROM dvc_data_versions 
        WHERE dataset_name = %s 
        ORDER BY creation_time DESC LIMIT 1
        """
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (dataset_name,))
                    result = cur.fetchone()
                    
                    if result:
                        return self.load_data_version(result[0])
                    else:
                        raise ValueError(f"No versions found for dataset: {dataset_name}")
                        
        except Exception as e:
            self.logger.error(f"Failed to get latest DVC version: {str(e)}")
            raise
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict:
        """Compare two data versions for schema and statistical differences"""
        try:
            version_1 = self.load_data_version(version_id_1)
            version_2 = self.load_data_version(version_id_2)
            
            df1, df2 = version_1['data'], version_2['data']
            schema1, schema2 = version_1['schema'], version_2['schema']
            
            comparison = {
                'version_1': version_id_1,
                'version_2': version_id_2,
                'schema_changes': {
                    'columns_added': list(set(schema2['columns']) - set(schema1['columns'])),
                    'columns_removed': list(set(schema1['columns']) - set(schema2['columns'])),
                    'shape_change': {
                        'old': schema1['shape'],
                        'new': schema2['shape']
                    }
                },
                'statistical_differences': {}
            }
            
            # Compare common numeric columns
            common_numeric = set(schema1.get('numeric_stats', {}).keys()) & \
                           set(schema2.get('numeric_stats', {}).keys())
            
            for col in common_numeric:
                old_stats = schema1['numeric_stats'][col]
                new_stats = schema2['numeric_stats'][col]
                
                comparison['statistical_differences'][col] = {
                    'mean_change': new_stats['mean'] - old_stats['mean'],
                    'std_change': new_stats['std'] - old_stats['std'],
                    'null_count_change': new_stats['null_count'] - old_stats['null_count']
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {str(e)}")
            raise
    
    def tag_version(self, version_id: str, tags: List[str]):
        """Add tags to a data version"""
        update_query = """
        UPDATE dvc_data_versions 
        SET tags = tags || %s
        WHERE version_id = %s
        """
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(update_query, (tags, version_id))
                conn.commit()
            
            self.logger.info(f"Added tags {tags} to version {version_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to tag version: {str(e)}")
            raise


# Integration functions for your existing pipeline

def dvc_version_raw_data(db_config: Dict, symbols: List[str]) -> List[str]:
    """Version raw crypto data using DVC"""
    versioning = DVCDataVersioning(db_config)
    version_ids = []
    
    for symbol in symbols:
        # Get raw data
        query = """
        SELECT open_time, close_price, high_price, low_price, open_price, volume, quote_volume
        FROM crypto_data 
        WHERE symbol = %s 
        ORDER BY open_time
        """
        
        with psycopg2.connect(**db_config) as conn:
            df = pd.read_sql(query, conn, params=[symbol])
        
        if df.empty:
            versioning.logger.warning(f"No data found for {symbol}")
            continue
        
        # Add datetime column
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('datetime')
        
        # Create version with enhanced metadata
        metadata = {
            'symbol': symbol,
            'data_type': 'raw_crypto_data',
            'source': 'binance_api',
            'features': list(df.columns),
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            },
            'collection_timestamp': datetime.now().isoformat()
        }
        
        tags = ['raw_data', symbol.lower(), 'binance']
        
        version_id = versioning.create_data_version(
            dataset_name=f"raw_data_{symbol}",
            df=df,
            metadata=metadata,
            tags=tags
        )
        
        version_ids.append(version_id)
        print(f"Created DVC version for {symbol}: {version_id}")
    
    return version_ids

def dvc_version_features_data(db_config: Dict, feature_results: Dict) -> List[str]:
    """Version engineered features using DVC - Bridge from MinIO to DVC"""
    versioning = DVCDataVersioning(db_config)
    version_ids = []
    
    for symbol, storage_path in feature_results['storage_paths'].items():
        try:
            versioning.logger.info(f"Loading features for {symbol} from MinIO: {storage_path}")
            
            # Load features from your existing MinIO storage
            response = versioning.minio_client.get_object(BUCKET_NAME, storage_path)
            df = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()
            
            versioning.logger.info(f"Loaded {len(df)} records with {len(df.columns)} features for {symbol}")
            
            # Create enhanced metadata for DVC
            metadata = {
                'symbol': symbol,
                'data_type': 'engineered_features',
                'feature_count': len(df.columns),
                'record_count': len(df),
                'features': list(df.columns),
                'pipeline_time': feature_results['pipeline_time'],
                'source_storage_path': storage_path,
                'feature_categories': _categorize_features(df.columns),
                'data_quality': {
                    'null_percentage': (df.isnull().sum() / len(df) * 100).mean(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                'date_range': {
                    'start': df.index.min().isoformat() if hasattr(df.index, 'min') else None,
                    'end': df.index.max().isoformat() if hasattr(df.index, 'max') else None
                }
            }
            
            # Create DVC version with rich tags
            tags = [
                'engineered_features', 
                symbol.lower(), 
                'crypto',
                feature_results['pipeline_time'][:8],  # YYYYMMDD
                f"features_{len(df.columns)}"
            ]
            
            # Version in DVC (this creates the enterprise-grade versioning)
            version_id = versioning.create_data_version(
                dataset_name=f"features_{symbol}",
                df=df,
                metadata=metadata,
                tags=tags
            )
            
            version_ids.append(version_id)
            versioning.logger.info(f"✅ Created DVC version for {symbol}: {version_id}")
            
        except Exception as e:
            versioning.logger.warning(f"Could not version features for {symbol}: {e}")
            continue
    
    return version_ids

def _categorize_features(columns):
    """Helper function to categorize features for metadata"""
    categories = {
        'technical': 0,
        'price': 0, 
        'volume': 0,
        'time': 0,
        'cross_symbol': 0,
        'other': 0
    }
    
    for col in columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['rsi', 'macd', 'sma', 'ema', 'bb_', 'atr']):
            categories['technical'] += 1
        elif any(x in col_lower for x in ['return', 'volatility', 'price', 'high', 'low', 'close', 'open']):
            categories['price'] += 1
        elif any(x in col_lower for x in ['volume', 'obv', 'vwap']):
            categories['volume'] += 1
        elif any(x in col_lower for x in ['hour', 'day', 'month', 'weekend', 'session']):
            categories['time'] += 1
        elif any(x in col_lower for x in ['corr_', 'relative_', 'ratio_to_', 'btc']):
            categories['cross_symbol'] += 1
        else:
            categories['other'] += 1
            
    return categories


# # DVC Pipeline Configuration Generator
# def generate_dvc_pipeline():
#     """Generate dvc.yaml for your ML pipeline"""
#     pipeline_config = {
#         'stages': {
#             'data_quality': {
#                 'cmd': 'python data_quality.py',
#                 'deps': ['data/raw/'],
#                 'outs': ['data/quality_reports/'],
#                 'metrics': ['metrics/data_quality.json']
#             },
#             'feature_engineering': {
#                 'cmd': 'python feature_eng.py',
#                 'deps': ['data/raw/', 'data/quality_reports/'],
#                 'outs': ['data/features/'],
#                 'metrics': ['metrics/feature_metrics.json']
#             },
#             'model_training': {
#                 'cmd': 'python train_model.py',
#                 'deps': ['data/features/'],
#                 'outs': ['models/'],
#                 'metrics': ['metrics/model_metrics.json'],
#                 'params': ['params.yaml:train']
#             }
#         }
#     }
    
#     with open('dvc.yaml', 'w') as f:
#         yaml.dump(pipeline_config, f, default_flow_style=False)
    
#     print("DVC pipeline configuration generated: dvc.yaml")


# Usage example:
"""
# Initialize DVC versioning
versioning = DVCDataVersioning(DB_CONFIG)

# Version raw data
raw_version_ids = dvc_version_raw_data(DB_CONFIG, SYMBOLS)

# List recent versions
recent_versions = versioning.list_versions(limit=5)
for version in recent_versions:
    print(f"{version['version_id']} - {version['dataset_name']} ({version['creation_time']})")

# Load specific version
data_version = versioning.load_data_version(raw_version_ids[0])
df = data_version['data']
print(f"Loaded {len(df)} records with {len(df.columns)} columns")

# Compare two versions
if len(raw_version_ids) >= 2:
    comparison = versioning.compare_versions(raw_version_ids[0], raw_version_ids[1])
    print(f"Schema changes: {comparison['schema_changes']}")
"""