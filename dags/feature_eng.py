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
        """Calculate returns, volatility, price ratios - UPDATED for multi-step"""
        features = df.copy()
        close = df['close_price']
        high = df['high_price']
        low = df['low_price']
        open_price = df['open_price']
        
        # CURRENT ISSUE: You're calculating returns but not targets properly
        # Keep these as FEATURES (not targets)
        features['return_15m'] = close.pct_change()  
        features['return_1h'] = close.pct_change(periods=4)
        features['return_4h'] = close.pct_change(periods=16)
        features['return_24h'] = close.pct_change(periods=96)
        features['return_7d'] = close.pct_change(periods=672)
        
        # Log returns
        features['log_return_15m'] = np.log(close / close.shift(1))
        
        # Volatility (rolling standard deviation of returns)
        features['volatility_1h'] = features['return_15m'].rolling(window=4).std()
        features['volatility_24h'] = features['return_15m'].rolling(window=96).std()
        features['volatility_7d'] = features['return_15m'].rolling(window=672).std()
        
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
        features['atr_14'] = features['true_range'].rolling(window=56).mean()
        
        return features

    def create_multi_horizon_targets(self, df, horizons=[1, 4, 16, 96]):
        """
        Create multiple prediction targets for different horizons
        horizons in 15-min periods: 1=15min, 4=1hour, 16=4hour, 96=24hour
        """
        targets = df[['open_time', 'close_price']].copy()  # Keep basic info
        close = df['close_price']
        
        for h in horizons:
            # 1. Price return prediction (regression)
            targets[f'target_return_{h}steps'] = close.pct_change(periods=h).shift(-h)
            
            # 2. Direction prediction (binary classification) 
            targets[f'target_direction_{h}steps'] = (targets[f'target_return_{h}steps'] > 0).astype(int)
            
            # 3. Multi-class direction prediction
            returns = targets[f'target_return_{h}steps']
            # Define thresholds based on your data (adjust these)
            targets[f'target_direction_multi_{h}steps'] = pd.cut(
                returns,
                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                labels=[0, 1, 2, 3, 4]  # strong_down, down, neutral, up, strong_up
            )
            
            # 4. Volatility prediction (regression)
            # Realized volatility over next h periods
            future_vol = close.pct_change().rolling(h).std().shift(-h)
            targets[f'target_volatility_{h}steps'] = future_vol
            
            # 5. Volatility regime prediction (classification)
            # Only create regime if we have valid volatility data
            valid_vol = future_vol.dropna()
            if len(valid_vol) > 10:  # Need minimum data points for quantiles
                vol_quantiles = valid_vol.quantile([0.33, 0.67])
                
                # Check if quantiles are valid (not NaN and different values)
                if not vol_quantiles.isna().any() and vol_quantiles.iloc[0] != vol_quantiles.iloc[1]:
                    targets[f'target_vol_regime_{h}steps'] = pd.cut(
                        future_vol,
                        bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
                        labels=[0, 1, 2]  # low, medium, high volatility
                    )
                else:
                    # Fallback: use fixed thresholds if quantiles fail
                    targets[f'target_vol_regime_{h}steps'] = pd.cut(
                        future_vol,
                        bins=[-np.inf, 0.01, 0.03, np.inf],  # Fixed volatility thresholds
                        labels=[0, 1, 2]
                    )
            else:
                # Not enough data - create empty series with correct dtype
                targets[f'target_vol_regime_{h}steps'] = pd.Series(
                    np.nan, index=targets.index, dtype='category'
                )
            
            # 6. Price level prediction (regression) - alternative to returns
            targets[f'target_price_{h}steps'] = close.shift(-h)
            
            # 7. High/Low prediction for next h periods (regression)
            targets[f'target_high_{h}steps'] = df['high_price'].rolling(h).max().shift(-h)
            targets[f'target_low_{h}steps'] = df['low_price'].rolling(h).min().shift(-h)
        
        return targets
    
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
        features['obv_sma_14'] = pd.Series(obv).rolling(window=56).mean()

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
                features['corr_with_btc_24h'] = features['close_price'].rolling(window=96).corr(
                    ref_data['close_price'])
                features['corr_with_btc_7d'] = features['close_price'].rolling(window=672).corr(
                    ref_data['close_price'])
                
                # Relative strength
                features['relative_strength_btc'] = (features['close_price'] / features['close_price'].iloc[0]) / \
                                                   (ref_data['close_price'] / ref_data['close_price'].iloc[0])
                
                # Price ratio to BTC
                features['price_ratio_to_btc'] = features['close_price'] / ref_data['close_price']
                features['ratio_ma_7'] = features['price_ratio_to_btc'].rolling(window=28).mean()
                
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
    
    def run_feature_pipeline(self, symbols, create_targets=True):
        """Run complete feature engineering pipeline with optional target creation"""
        results = {
            'pipeline_time': datetime.now().isoformat(),
            'symbols_processed': [],
            'storage_paths': {},
            'target_paths': {}  # NEW: separate storage for targets
        }
        
        # Step 1: Individual symbol features
        individual_features = {}
        for symbol in symbols:
            individual_features[symbol] = self.engineer_features_for_symbol(symbol)
            results['symbols_processed'].append(symbol)
        
        # Step 2: Cross-symbol features
        self.logger.info("Calculating cross-symbol features")
        cross_features = self.calculate_cross_symbol_features(symbols)
        
        # Step 3: Create targets if requested
        targets_data = {}
        if create_targets:
            self.logger.info("Creating multi-horizon targets")
            for symbol in symbols:
                base_data = individual_features[symbol]
                targets_data[symbol] = self.create_multi_horizon_targets(base_data)
        
        # Step 4: Combine and save
        for symbol in symbols:
            # Combine features
            individual_cols = set(individual_features[symbol].columns)
            cross_cols = set(cross_features[symbol].columns)
            
            combined_features = individual_features[symbol].copy()
            new_cross_features = cross_cols - individual_cols
            for col in new_cross_features:
                combined_features[col] = cross_features[symbol][col]
            
            # Save features
            date_partition = datetime.now().strftime('%Y%m%d')
            features_path = f"features/{date_partition}/{symbol}.parquet"
            
            parquet_buffer = BytesIO()
            combined_features.to_parquet(parquet_buffer, index=True)
            parquet_buffer.seek(0)
            
            self.minio_client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=features_path,
                data=parquet_buffer,
                length=len(parquet_buffer.getvalue()),
                content_type='application/octet-stream'
            )
            
            results['storage_paths'][symbol] = features_path
            
            # Save targets separately if created
            if create_targets:
                targets_path = f"targets/{date_partition}/{symbol}.parquet"
                
                targets_buffer = BytesIO()
                targets_data[symbol].to_parquet(targets_buffer, index=True)
                targets_buffer.seek(0)
                
                self.minio_client.put_object(
                    bucket_name=BUCKET_NAME,
                    object_name=targets_path,
                    data=targets_buffer,
                    length=len(targets_buffer.getvalue()),
                    content_type='application/octet-stream'
                )
                
                results['target_paths'][symbol] = targets_path
                
            self.logger.info(f"Saved features for {symbol} to MinIO: {features_path}")
            if create_targets:
                self.logger.info(f"Saved targets for {symbol} to MinIO: {targets_path}")
        
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
