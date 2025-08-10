import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import logging

DB_CONFIG = {
    "dbname": "postgres",
    "user": "varunrajput", 
    "password": "yourpassword",
    "host": "host.docker.internal",
    "port": "5432"
}

# Keep your existing imports and constants
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT']
BASE_URL = "https://api.binance.com/api/v3/klines"

class DataQualityAssessment():
    def __init__(self, db_config):
        self.logger = logging.getLogger("data_quality")
        self.db_config = db_config

    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def validate_timestamp_continuity(self, symbol, interval_minutes=60):
        """Check for gaps in timestamp continuity"""
        """Uses the LAG window function to get the previous row's open_time value when rows are ordered by open_time, aliasing it as "prev_time"""
        """Calculates the time difference between current and previous open_time, divides by 60000 (converts milliseconds to minutes), aliases as "gap_minutes"""
        query = """
        SELECT open_time, 
            LAG(open_time) OVER (ORDER BY CAST(open_time AS bigint)) as prev_time,
            (CAST(open_time AS bigint) - LAG(CAST(open_time AS bigint)) OVER (ORDER BY CAST(open_time AS bigint))) / 60000 as gap_minutes
        FROM crypto_data 
        WHERE symbol = %s 
        ORDER BY CAST(open_time AS bigint)
        """

        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=[symbol])
        
        # Find gaps larger than expected interval
        expected_gap = interval_minutes
        gaps = df[df['gap_minutes'] > expected_gap * 1.5]  # 50% tolerance
        
        return {
            'symbol': symbol,
            'total_records': len(df),
            'gaps_found': len(gaps),
            'gap_details': gaps[['open_time', 'prev_time', 'gap_minutes']].to_dict('records')
        }
    
    def check_price_volume_outliers(self, symbol, z_threshold=3):
        """Detect outliers using Z-score method"""
        """A positive Z-score means the value is above the mean, while a negative Z-score indicates it's below the mean."""
        """Outlier is a data point that significantly differs from other observations in a dataset"""
        query = """
        SELECT open_time, close_price, volume, high_price, low_price
        FROM crypto_data 
        WHERE symbol = %s 
        ORDER BY open_time
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=[symbol])
        
        outliers = {}
        
        # Check price outliers
        for col in ['close_price', 'high_price', 'low_price']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = z_scores > z_threshold
            outliers[f'{col}_outliers'] = df[outlier_mask][['open_time', col]].to_dict('records')
        
        # Check volume outliers
        log_volume = np.log1p(df['volume'])  # log transform for volume
        z_scores = np.abs((log_volume - log_volume.mean()) / log_volume.std())
        volume_outliers = df[z_scores > z_threshold][['open_time', 'volume']].to_dict('records')
        outliers['volume_outliers'] = volume_outliers
        
        return {
            'symbol': symbol,
            'outliers': outliers,
            'total_outliers': sum(len(v) for v in outliers.values())
        }
    
    def assess_data_completeness(self):
        """Check completeness across all symbols"""
        query = """
        SELECT 
            symbol,
            COUNT(*) as record_count,
            MIN(CAST(open_time AS bigint)) as earliest_time,
            MAX(CAST(open_time AS bigint)) as latest_time,
            COUNT(CASE WHEN close_price IS NULL THEN 1 END) as null_prices,
            COUNT(CASE WHEN volume IS NULL THEN 1 END) as null_volumes
        FROM crypto_data 
        GROUP BY symbol
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        
        # Calculate time span for each symbol (now both columns are numeric)
        df['time_span_hours'] = (df['latest_time'] - df['earliest_time']) / (1000 * 3600)
        df['expected_records'] = df['time_span_hours']  # 1 record per hour
        df['completeness_ratio'] = df['record_count'] / df['expected_records']
        
        return {
            'symbols_assessed': len(df),
            'completeness_summary': df.to_dict('records'),
            'avg_completeness': df['completeness_ratio'].mean(),
            'symbols_with_nulls': len(df[(df['null_prices'] > 0) | (df['null_volumes'] > 0)])
        }
    
    def run_quality_assessment(self, symbols):
        """Run all quality checks"""
        results = {
            'assessment_time': datetime.now().isoformat(),
            'timestamp_continuity': {},
            'outlier_detection': {},
            'completeness': None
        }
        
        # Check each symbol
        for symbol in symbols:
            self.logger.info(f"Assessing {symbol}...")
            
            # Timestamp continuity
            results['timestamp_continuity'][symbol] = self.validate_timestamp_continuity(symbol)
            
            # Outlier detection
            results['outlier_detection'][symbol] = self.check_price_volume_outliers(symbol)
        
        # Overall completeness
        results['completeness'] = self.assess_data_completeness()
        
        return results
    



# Usage example:
"""
# Run assessment
assessor = DataQualityAssessment(DB_CONFIG)
quality_report = assessor.run_quality_assessment(SYMBOLS)

# Print summary
print(f"Completeness: {quality_report['completeness']['avg_completeness']:.2%}")
for symbol in SYMBOLS:
    gaps = quality_report['timestamp_continuity'][symbol]['gaps_found']
    outliers = quality_report['outlier_detection'][symbol]['total_outliers']
    print(f"{symbol}: {gaps} gaps, {outliers} outliers")
"""
