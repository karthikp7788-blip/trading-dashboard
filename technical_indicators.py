"""
Technical Indicators Module
Calculates technical indicators and features from price data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data"""
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list = [20, 50, 200], column: str = 'Close') -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in periods:
            df[f'SMA_{period}'] = df[column].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list = [12, 26, 50], column: str = 'Close') -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for period in periods:
            df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, 
                 fast: int = 12, 
                 slow: int = 26, 
                 signal: int = 9,
                 column: str = 'Close') -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)"""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, 
                           period: int = 20, 
                           std_dev: float = 2.0,
                           column: str = 'Close') -> pd.DataFrame:
        """Add Bollinger Bands"""
        df['BB_Middle'] = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (std * std_dev)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df[column] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add volume-based features"""
        df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
        """Add price momentum features"""
        for period in periods:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
        return df
    
    @staticmethod
    def add_volatility_features(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """Add volatility features"""
        returns = df['Close'].pct_change()
        
        for period in periods:
            df[f'Volatility_{period}d'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend identification features"""
        # Price above/below moving averages
        df['Above_SMA_20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['Above_SMA_50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['Above_SMA_200'] = (df['Close'] > df['SMA_200']).astype(int)
        
        # Moving average alignment
        df['MA_Alignment'] = (
            (df['SMA_20'] > df['SMA_50']).astype(int) + 
            (df['SMA_50'] > df['SMA_200']).astype(int)
        )
        
        return df
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        result = df.copy()
        
        print(f"Calculating technical indicators...")
        
        # Moving averages
        result = TechnicalIndicators.add_sma(result)
        result = TechnicalIndicators.add_ema(result)
        
        # Oscillators
        result = TechnicalIndicators.add_rsi(result)
        result = TechnicalIndicators.add_macd(result)
        
        # Bands and ranges
        result = TechnicalIndicators.add_bollinger_bands(result)
        result = TechnicalIndicators.add_atr(result)
        
        # Volume
        result = TechnicalIndicators.add_volume_features(result)
        
        # Momentum
        result = TechnicalIndicators.add_momentum_features(result)
        
        # Volatility
        result = TechnicalIndicators.add_volatility_features(result)
        
        # Trend
        result = TechnicalIndicators.add_trend_features(result)
        
        # Drop NaN rows from indicator calculations
        initial_rows = len(result)
        result = result.dropna()
        dropped_rows = initial_rows - len(result)
        
        print(f"✓ Calculated {len([c for c in result.columns if c not in df.columns])} indicators")
        print(f"✓ Dropped {dropped_rows} rows with NaN values")
        print(f"✓ Final dataset: {len(result)} rows")
        
        return result


def main():
    """Test technical indicators"""
    from data_pipeline import DataPipeline
    
    # Fetch SPY data
    pipeline = DataPipeline(['SPY'])
    results = pipeline.fetch_all_data()
    
    spy_data = results['price_data']['SPY']
    
    if not spy_data.empty:
        # Calculate indicators
        spy_with_indicators = TechnicalIndicators.calculate_all_indicators(spy_data)
        
        print("\n" + "=" * 60)
        print("TECHNICAL INDICATORS SAMPLE (Last 5 Days)")
        print("=" * 60)
        
        # Show key indicators
        key_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Width', 'ATR', 'Volume_Ratio']
        print(spy_with_indicators[key_cols].tail())
        
        print("\n" + "=" * 60)
        print("INDICATOR STATISTICS")
        print("=" * 60)
        print(spy_with_indicators[key_cols].describe())


if __name__ == "__main__":
    main()
