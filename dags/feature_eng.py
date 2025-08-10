import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from io import BytesIO


from minio import Minio
from minio.error import S3Error

# MinIO Configuration
MINIO_CONFIG = {
    'endpoint': 'minio:9000',  # Adjust based on your Helm setup
    'access_key': 'admin',  # Change these in production!
    'secret_key': 'admin123',
    'secure': False  # Set to True if using HTTPS
}

BUCKET_NAME = 'crypto-features'

class FeatureEngineeringPipeline:
    def __init__(self, db_config):
        self.logger = logging.getLogger("feature_engineering")
        self.db_config = db_config
        self.minio_client = self._get_minio_client()
        self._ensure_bucket_exists()
    
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

    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def get_symbol_data(self, symbol, limit=None):
        """Fetch symbol data ordered by time"""
        query = """
        SELECT open_time, close_price, high_price, low_price, open_price, volume, quote_volume
        FROM crypto_data 
        WHERE symbol = %s 
        ORDER BY open_time
        """
        if limit:
            query += f" LIMIT {limit}"
            
        with self.get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=[symbol])
        
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        return df.set_index('datetime')
    
    def calculate_technical_indicators(self, df):
        """Calculate RSI, MACD, Bollinger Bands, Moving Averages"""
        features = df.copy()
        close = df['close_price']
        
        # Moving Averages
        for period in [7, 14, 21, 50]:
            features[f'sma_{period}'] = close.rolling(window=period).mean()
            features[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        features['rsi_14'] = calculate_rsi(close)
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        features['bb_upper'] = sma20 + (std20 * 2)
        features['bb_lower'] = sma20 - (std20 * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (close - features['bb_lower']) / features['bb_width']
        
        return features
    
    def calculate_price_features(self, df):
        """Calculate returns, volatility, price ratios"""
        features = df.copy()
        close = df['close_price']
        high = df['high_price']
        low = df['low_price']
        open_price = df['open_price']
        
        # Returns
        features['return_1h'] = close.pct_change()
        features['return_4h'] = close.pct_change(periods=4)
        features['return_24h'] = close.pct_change(periods=24)
        features['return_7d'] = close.pct_change(periods=168)  # 7 days * 24 hours
        
        # Log returns
        features['log_return_1h'] = np.log(close / close.shift(1))
        
        # Volatility (rolling standard deviation of returns)
        features['volatility_24h'] = features['return_1h'].rolling(window=24).std()
        features['volatility_7d'] = features['return_1h'].rolling(window=168).std()
        
        # Price ratios
        features['high_low_ratio'] = high / low
        features['close_open_ratio'] = close / open_price
        features['hl2'] = (high + low) / 2
        features['hlc3'] = (high + low + close) / 3
        features['ohlc4'] = (open_price + high + low + close) / 4
        
        # Price position within range
        features['price_position'] = (close - low) / (high - low)
        
        # True Range and Average True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        features['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        features['atr_14'] = features['true_range'].rolling(window=14).mean()
        
        return features
    
    def calculate_volume_features(self, df):
        """Calculate volume ratios, volume moving averages"""
        features = df.copy()
        volume = df['volume']
        quote_volume = df['quote_volume']
        
        # Volume moving averages
        for period in [7, 14, 24, 168]:  # 7h, 14h, 24h, 7d
            features[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
        
        # Volume ratios
        features['volume_ratio_7'] = volume / features['volume_sma_7']
        features['volume_ratio_24'] = volume / features['volume_sma_24']
        
        # Volume-Price relationship
        features['vwap'] = (quote_volume / volume).fillna(0)  # Volume Weighted Average Price
        features['volume_price_trend'] = features['return_1h'] * features['volume_ratio_24']
        
        # On-Balance Volume (OBV)
        obv = []
        obv_val = 0
        returns = features['return_1h'].fillna(0)
        
        for i, ret in enumerate(returns):
            if ret > 0:
                obv_val += volume.iloc[i]
            elif ret < 0:
                obv_val -= volume.iloc[i]
            obv.append(obv_val)
        
        features['obv'] = obv
        features['obv_sma_14'] = pd.Series(obv).rolling(window=14).mean()
        
        return features
    
    def calculate_time_features(self, df):
        """Calculate hour of day, day of week, seasonality features"""
        features = df.copy()
        
        # Extract time components
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['day_of_month'] = features.index.day
        features['month'] = features.index.month
        
        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Weekend indicator
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Trading session indicators (assuming UTC time)
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['european_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['american_session'] = ((features['hour'] >= 16) & (features['hour'] < 24)).astype(int)
        
        return features
    
    def calculate_cross_symbol_features(self, symbols, reference_symbol='BTCUSDT'):
        """Calculate correlation and relative strength across symbols"""
        # Get data for all symbols
        symbol_data = {}
        for symbol in symbols:
            symbol_data[symbol] = self.get_symbol_data(symbol)
        
        # Find common time range
        common_index = symbol_data[symbols[0]].index
        for symbol in symbols[1:]:
            common_index = common_index.intersection(symbol_data[symbol].index)
        
        # Align all data to common timeframe
        aligned_data = {}
        for symbol in symbols:
            aligned_data[symbol] = symbol_data[symbol].loc[common_index]
        
        # Calculate cross-symbol features
        cross_features = {}
        
        for symbol in symbols:
            features = aligned_data[symbol].copy()
            
            if symbol != reference_symbol:
                ref_data = aligned_data[reference_symbol]
                
                # Price correlation (rolling)
                features['corr_with_btc_24h'] = features['close_price'].rolling(window=24).corr(
                    ref_data['close_price'])
                features['corr_with_btc_7d'] = features['close_price'].rolling(window=168).corr(
                    ref_data['close_price'])
                
                # Relative strength
                features['relative_strength_btc'] = (features['close_price'] / features['close_price'].iloc[0]) / \
                                                   (ref_data['close_price'] / ref_data['close_price'].iloc[0])
                
                # Price ratio to BTC
                features['price_ratio_to_btc'] = features['close_price'] / ref_data['close_price']
                features['ratio_ma_7'] = features['price_ratio_to_btc'].rolling(window=7).mean()
                
                # Volume correlation
                features['volume_corr_btc_24h'] = features['volume'].rolling(window=24).corr(
                    ref_data['volume'])
            
            cross_features[symbol] = features
        
        return cross_features
    
    def engineer_features_for_symbol(self, symbol):
        """Complete feature engineering for a single symbol"""
        self.logger.info(f"Engineering features for {symbol}")
        
        # Get base data
        df = self.get_symbol_data(symbol)
        
        # Apply all feature engineering steps
        df = self.calculate_technical_indicators(df)
        df = self.calculate_price_features(df)
        df = self.calculate_volume_features(df)
        df = self.calculate_time_features(df)
        
        return df
    
    def run_feature_pipeline(self, symbols):
        """Run complete feature engineering pipeline"""
        results = {
            'pipeline_time': datetime.now().isoformat(),
            'symbols_processed': [],
            'storage_paths': {}
        }
        
        # Step 1: Individual symbol features
        individual_features = {}
        for symbol in symbols:
            individual_features[symbol] = self.engineer_features_for_symbol(symbol)
            results['symbols_processed'].append(symbol)
        
        # Step 2: Cross-symbol features
        self.logger.info("Calculating cross-symbol features")
        cross_features = self.calculate_cross_symbol_features(symbols)
        
        # Step 3: Combine individual and cross-symbol features
        for symbol in symbols:
            # Get common columns between individual and cross features
            individual_cols = set(individual_features[symbol].columns)
            cross_cols = set(cross_features[symbol].columns)
            
            # Merge on index (datetime)
            combined = individual_features[symbol].copy()
            
            # Add cross-symbol features that aren't already present
            new_cross_features = cross_cols - individual_cols
            for col in new_cross_features:
                combined[col] = cross_features[symbol][col]
            
            # Save to MinIO
            date_partition = datetime.now().strftime('%Y%m%d')
            object_name = f"features/{date_partition}/{symbol}.parquet"
        
            # Convert DataFrame to parquet bytes
            parquet_buffer = BytesIO()
            combined.to_parquet(parquet_buffer, index=True)
            parquet_buffer.seek(0)
            
            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=object_name,
                data=parquet_buffer,
                length=len(parquet_buffer.getvalue()),
                content_type='application/octet-stream'
            )
            
            results['storage_paths'][symbol] = object_name
            self.logger.info(f"Saved features for {symbol} to MinIO: {object_name}")
        
        return results


# Usage example:
"""
DB_CONFIG = {
    "dbname": "postgres",
    "user": "varunrajput", 
    "password": "yourpassword",
    "host": "host.docker.internal",
    "port": "5432"
}

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

# Run feature engineering
pipeline = FeatureEngineeringPipeline(DB_CONFIG)
feature_results = pipeline.run_feature_pipeline(SYMBOLS)

# Access features for each symbol
for symbol in SYMBOLS:
    features_df = feature_results['feature_data'][symbol]
    print(f"{symbol}: {len(features_df.columns)} features, {len(features_df)} records")
    
    # Example: Get latest features
    latest_features = features_df.iloc[-1]
    print(f"Latest RSI: {latest_features['rsi_14']:.2f}")
    print(f"Latest 24h return: {latest_features['return_24h']:.4f}")
"""
