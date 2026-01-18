"""
Feature Engineering Module
Combines price, technical indicators, and options data into ML-ready features
Creates target variables for prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class FeatureEngineering:
    """Transform raw data into ML-ready features and targets"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_columns = []
        
    @staticmethod
    def create_target_variables(df: pd.DataFrame, 
                                horizons: List[int] = [1, 5]) -> pd.DataFrame:
        """
        Create prediction target variables
        
        Args:
            df: DataFrame with price data
            horizons: List of forward-looking periods (days)
            
        Returns:
            DataFrame with target variables added
        """
        result = df.copy()
        
        for horizon in horizons:
            # Next-period close price
            result[f'target_close_{horizon}d'] = result['Close'].shift(-horizon)
            
            # Next-period return
            result[f'target_return_{horizon}d'] = (
                result[f'target_close_{horizon}d'] / result['Close'] - 1
            )
            
            # Direction (binary: 1 = up, 0 = down)
            result[f'target_direction_{horizon}d'] = (
                result[f'target_return_{horizon}d'] > 0
            ).astype(int)
            
            # Range prediction (high - low over next N days)
            result[f'target_high_{horizon}d'] = result['High'].rolling(window=horizon).max().shift(-horizon)
            result[f'target_low_{horizon}d'] = result['Low'].rolling(window=horizon).min().shift(-horizon)
            result[f'target_range_{horizon}d'] = (
                result[f'target_high_{horizon}d'] - result[f'target_low_{horizon}d']
            )
            
            # Volatility realized over next N days
            result[f'target_volatility_{horizon}d'] = (
                result['Close'].pct_change().shift(-horizon).rolling(window=horizon).std() * np.sqrt(252)
            )
        
        return result
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: DataFrame with features
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features added
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        
        return result
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling statistics features
        
        Args:
            df: DataFrame with features
            columns: Columns to create rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                # Rolling mean
                result[f'{col}_rolling_mean_{window}'] = result[col].rolling(window=window).mean()
                
                # Rolling std
                result[f'{col}_rolling_std_{window}'] = result[col].rolling(window=window).std()
                
                # Rolling min/max
                result[f'{col}_rolling_min_{window}'] = result[col].rolling(window=window).min()
                result[f'{col}_rolling_max_{window}'] = result[col].rolling(window=window).max()
                
                # Z-score (current value vs rolling stats)
                mean = result[f'{col}_rolling_mean_{window}']
                std = result[f'{col}_rolling_std_{window}']
                result[f'{col}_zscore_{window}'] = (result[col] - mean) / std
        
        return result
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key indicators
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features added
        """
        result = df.copy()
        
        # RSI * Volume Ratio (momentum with volume confirmation)
        if 'RSI' in df.columns and 'Volume_Ratio' in df.columns:
            result['RSI_x_Volume'] = result['RSI'] * result['Volume_Ratio']
        
        # MACD * ATR (momentum with volatility context)
        if 'MACD_Hist' in df.columns and 'ATR' in df.columns:
            result['MACD_x_ATR'] = result['MACD_Hist'] * result['ATR']
        
        # Bollinger position * volatility
        if 'BB_Position' in df.columns and 'Volatility_20d' in df.columns:
            result['BB_x_Vol'] = result['BB_Position'] * result['Volatility_20d']
        
        # Options skew * volatility (if available)
        if 'options_pc_skew' in df.columns and 'Volatility_20d' in df.columns:
            result['Skew_x_Vol'] = result['options_pc_skew'] * result['Volatility_20d']
        
        # Premium imbalance * volume ratio
        if 'options_premium_imbalance' in df.columns and 'Volume_Ratio' in df.columns:
            result['Premium_x_Volume'] = result['options_premium_imbalance'] * result['Volume_Ratio']
        
        return result
    
    @staticmethod
    def add_options_features(df: pd.DataFrame, 
                            options_metrics: Dict,
                            date: pd.Timestamp) -> Dict:
        """
        Add options-derived features for a specific date
        
        Args:
            df: DataFrame with price data
            options_metrics: Dictionary of options metrics
            date: Date for these metrics
            
        Returns:
            Dictionary of options features
        """
        features = {}
        
        # Prefix all options metrics
        for key, value in options_metrics.items():
            # Skip non-numeric values
            if isinstance(value, (list, str)):
                continue
            
            # Add with options_ prefix
            features[f'options_{key}'] = value
        
        return features
    
    @staticmethod
    def create_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime classification features
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with regime features added
        """
        result = df.copy()
        
        # Trend regime (based on MA alignment and slope)
        if all(col in df.columns for col in ['SMA_20', 'SMA_50', 'SMA_200', 'Close']):
            # Uptrend: price > all MAs and MAs aligned
            result['regime_uptrend'] = (
                (result['Close'] > result['SMA_20']) &
                (result['Close'] > result['SMA_50']) &
                (result['Close'] > result['SMA_200']) &
                (result['SMA_20'] > result['SMA_50']) &
                (result['SMA_50'] > result['SMA_200'])
            ).astype(int)
            
            # Downtrend: opposite
            result['regime_downtrend'] = (
                (result['Close'] < result['SMA_20']) &
                (result['Close'] < result['SMA_50']) &
                (result['Close'] < result['SMA_200']) &
                (result['SMA_20'] < result['SMA_50']) &
                (result['SMA_50'] < result['SMA_200'])
            ).astype(int)
            
            # Range-bound: neither
            result['regime_ranging'] = (
                (result['regime_uptrend'] == 0) & 
                (result['regime_downtrend'] == 0)
            ).astype(int)
        
        # Volatility regime
        if 'Volatility_20d' in df.columns:
            vol_median = df['Volatility_20d'].median()
            result['regime_high_vol'] = (df['Volatility_20d'] > vol_median * 1.5).astype(int)
            result['regime_low_vol'] = (df['Volatility_20d'] < vol_median * 0.5).astype(int)
        
        # Volume regime
        if 'Volume_Ratio' in df.columns:
            result['regime_high_volume'] = (df['Volume_Ratio'] > 1.5).astype(int)
            result['regime_low_volume'] = (df['Volume_Ratio'] < 0.5).astype(int)
        
        return result
    
    @staticmethod
    def select_features(df: pd.DataFrame, 
                       target_col: str,
                       method: str = 'correlation',
                       top_n: int = 50) -> List[str]:
        """
        Select most important features
        
        Args:
            df: DataFrame with all features and target
            target_col: Name of target column
            method: Selection method ('correlation', 'variance')
            top_n: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        # Remove non-numeric and target columns
        feature_cols = [col for col in df.columns 
                       if col not in ['Ticker', target_col] 
                       and df[col].dtype in ['int64', 'float64']
                       and not col.startswith('target_')]
        
        features_df = df[feature_cols + [target_col]].dropna()
        
        if method == 'correlation':
            # Calculate correlation with target
            correlations = features_df.corr()[target_col].abs().sort_values(ascending=False)
            selected = correlations.head(top_n + 1).index.tolist()
            selected = [f for f in selected if f != target_col][:top_n]
            
        elif method == 'variance':
            # Select features with highest variance
            variances = features_df[feature_cols].var().sort_values(ascending=False)
            selected = variances.head(top_n).index.tolist()
        
        else:
            # Default: return all numeric features
            selected = feature_cols[:top_n]
        
        return selected
    
    def create_ml_dataset(self,
                         price_data: pd.DataFrame,
                         options_metrics_history: Optional[Dict] = None,
                         target_horizon: int = 1) -> pd.DataFrame:
        """
        Create complete ML-ready dataset
        
        Args:
            price_data: DataFrame with price and technical indicators
            options_metrics_history: Optional dict of {date: options_metrics}
            target_horizon: Days ahead to predict
            
        Returns:
            Complete feature dataset with targets
        """
        print(f"\n{'='*60}")
        print("CREATING ML DATASET")
        print(f"{'='*60}")
        
        # Start with price data (already has technical indicators)
        df = price_data.copy()
        initial_rows = len(df)
        print(f"\n✓ Starting with {initial_rows} rows")
        
        # Add options features if available
        if options_metrics_history:
            print(f"✓ Adding options features from {len(options_metrics_history)} dates")
            
            for date, metrics in options_metrics_history.items():
                if date in df.index:
                    opt_features = self.add_options_features(df, metrics, date)
                    for key, value in opt_features.items():
                        if key not in df.columns:
                            df[key] = np.nan
                        df.loc[date, key] = value
        
        # Create target variables
        print(f"✓ Creating target variables (horizon: {target_horizon}d)")
        df = self.create_target_variables(df, horizons=[target_horizon])
        
        # Create lagged features for key indicators
        print("✓ Creating lagged features")
        lag_cols = ['Close', 'Volume', 'RSI', 'MACD_Hist', 'ATR', 'Volatility_20d']
        lag_cols = [col for col in lag_cols if col in df.columns]
        df = self.create_lag_features(df, lag_cols, lags=[1, 2, 3, 5])
        
        # Create rolling features
        print("✓ Creating rolling statistics features")
        rolling_cols = ['RSI', 'Volume_Ratio', 'MACD_Hist']
        rolling_cols = [col for col in rolling_cols if col in df.columns]
        df = self.create_rolling_features(df, rolling_cols, windows=[5, 10, 20])
        
        # Create interaction features
        print("✓ Creating interaction features")
        df = self.create_interaction_features(df)
        
        # Create regime features
        print("✓ Creating regime classification features")
        df = self.create_regime_features(df)
        
        # Drop rows with NaN in target
        target_col = f'target_direction_{target_horizon}d'
        df_clean = df.dropna(subset=[target_col])
        dropped = len(df) - len(df_clean)
        
        print(f"\n✓ Dropped {dropped} rows with missing targets")
        print(f"✓ Final dataset: {len(df_clean)} rows")
        
        # Count features
        feature_cols = [col for col in df_clean.columns 
                       if not col.startswith('target_') 
                       and col not in ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✓ Total features: {len(feature_cols)}")
        
        self.feature_columns = feature_cols
        self.target_columns = [col for col in df_clean.columns if col.startswith('target_')]
        
        return df_clean
    
    @staticmethod
    def train_test_split_time_series(df: pd.DataFrame,
                                     test_size: float = 0.2,
                                     validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into train/validation/test
        Maintains temporal order (no random shuffling)
        
        Args:
            df: DataFrame with features and targets
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        
        # Calculate split points
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - validation_size))
        
        train_df = df.iloc[:val_start].copy()
        val_df = df.iloc[val_start:test_start].copy()
        test_df = df.iloc[test_start:].copy()
        
        print(f"\nTime Series Split:")
        print(f"  Train: {len(train_df)} rows ({train_df.index.min():%Y-%m-%d} to {train_df.index.max():%Y-%m-%d})")
        print(f"  Val:   {len(val_df)} rows ({val_df.index.min():%Y-%m-%d} to {val_df.index.max():%Y-%m-%d})")
        print(f"  Test:  {len(test_df)} rows ({test_df.index.min():%Y-%m-%d} to {test_df.index.max():%Y-%m-%d})")
        
        return train_df, val_df, test_df


def main():
    """Test feature engineering"""
    from data_pipeline import DataPipeline
    from technical_indicators import TechnicalIndicators
    from options_analytics import OptionsAnalytics
    
    # Generate synthetic data
    print("Generating synthetic data for testing...")
    
    import sys
    sys.path.append('.')
    from demo_synthetic import generate_synthetic_price_data, generate_synthetic_options
    
    # Generate SPY data
    spy_data = generate_synthetic_price_data('SPY', days=1260)
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    spy_with_indicators = TechnicalIndicators.calculate_all_indicators(spy_data)
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Create ML dataset
    ml_dataset = fe.create_ml_dataset(
        spy_with_indicators,
        target_horizon=1
    )
    
    # Show feature summary
    print(f"\n{'='*60}")
    print("FEATURE SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nDataset shape: {ml_dataset.shape}")
    print(f"Date range: {ml_dataset.index.min():%Y-%m-%d} to {ml_dataset.index.max():%Y-%m-%d}")
    
    # Show sample features
    print(f"\n{'='*60}")
    print("SAMPLE FEATURES (Last Row)")
    print(f"{'='*60}")
    
    feature_cols = fe.feature_columns[:20]  # Show first 20
    print(ml_dataset[feature_cols].tail(1).T)
    
    # Show targets
    print(f"\n{'='*60}")
    print("TARGET VARIABLES (Last 5 Rows)")
    print(f"{'='*60}")
    
    target_cols = [col for col in ml_dataset.columns if col.startswith('target_')][:5]
    print(ml_dataset[target_cols].tail())
    
    # Feature selection
    print(f"\n{'='*60}")
    print("TOP 20 FEATURES BY CORRELATION WITH NEXT-DAY DIRECTION")
    print(f"{'='*60}")
    
    top_features = fe.select_features(
        ml_dataset,
        target_col='target_direction_1d',
        method='correlation',
        top_n=20
    )
    
    correlations = ml_dataset[top_features + ['target_direction_1d']].corr()['target_direction_1d'].sort_values(ascending=False)
    print(correlations[1:21])  # Skip the target itself
    
    # Train/val/test split
    print(f"\n{'='*60}")
    print("TRAIN/VALIDATION/TEST SPLIT")
    print(f"{'='*60}")
    
    train, val, test = fe.train_test_split_time_series(ml_dataset)
    
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING COMPLETE ✓")
    print(f"{'='*60}")
    
    return ml_dataset, fe


if __name__ == "__main__":
    dataset, feature_eng = main()
