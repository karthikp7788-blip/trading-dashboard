"""
Data Pipeline Module
Handles data ingestion from yfinance for stock prices and options chains
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPipeline:
    """Main data pipeline for fetching and validating market data"""
    
    def __init__(self, tickers: List[str] = None):
        """
        Initialize data pipeline
        
        Args:
            tickers: List of ticker symbols (default: SPY, TSLA, GOOGL, NVDA, AMD, META)
        """
        self.tickers = tickers or ['SPY', 'TSLA', 'GOOGL', 'NVDA', 'AMD', 'META']
        self.price_data = {}
        self.options_data = {}
        
    def fetch_price_data(self, 
                        ticker: str, 
                        period: str = '5y',
                        interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical price data for a ticker
        
        Args:
            ticker: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                print(f"⚠️  No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean data
            df = df.dropna()
            
            # Add ticker column
            df['Ticker'] = ticker
            
            print(f"✓ Fetched {len(df)} rows for {ticker} ({interval} interval)")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_options_chain(self, ticker: str) -> Dict:
        """
        Fetch current options chain for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            
            if not expirations:
                print(f"⚠️  No options available for {ticker}")
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
            
            # Get nearest expiration (most liquid)
            nearest_exp = expirations[0]
            
            # Fetch options chain
            opt_chain = stock.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Add metadata
            calls['expiration'] = nearest_exp
            puts['expiration'] = nearest_exp
            calls['ticker'] = ticker
            puts['ticker'] = ticker
            
            print(f"✓ Fetched options for {ticker} (exp: {nearest_exp})")
            print(f"  Calls: {len(calls)} contracts, Puts: {len(puts)} contracts")
            
            return {
                'calls': calls,
                'puts': puts,
                'expirations': expirations,
                'nearest_expiration': nearest_exp
            }
            
        except Exception as e:
            print(f"✗ Error fetching options for {ticker}: {str(e)}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
    
    def validate_price_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate price data quality
        
        Args:
            df: Price DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if df.empty:
            return False, ["DataFrame is empty"]
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for negative prices
        if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            issues.append("Negative prices detected")
        
        # Check High >= Low
        if (df['High'] < df['Low']).any():
            issues.append("High < Low detected")
        
        # Check for zero volume days
        zero_vol_days = (df['Volume'] == 0).sum()
        if zero_vol_days > 0:
            issues.append(f"{zero_vol_days} days with zero volume")
        
        # Check for missing data (gaps)
        if df.index.name == 'Date':
            date_diff = df.index.to_series().diff()
            max_gap = date_diff.max().days if len(date_diff) > 1 else 0
            if max_gap > 7:  # More than a week gap (excluding weekends)
                issues.append(f"Large data gap detected: {max_gap} days")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_options_data(self, options_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate options chain data quality
        
        Args:
            options_dict: Dictionary with calls and puts
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        calls = options_dict.get('calls', pd.DataFrame())
        puts = options_dict.get('puts', pd.DataFrame())
        
        if calls.empty and puts.empty:
            return False, ["No options data available"]
        
        # Check calls
        if not calls.empty:
            # Check for negative bid/ask
            if (calls[['bid', 'ask']] < 0).any().any():
                issues.append("Negative bid/ask in calls")
            
            # Check bid <= ask
            if (calls['bid'] > calls['ask']).any():
                issues.append("Bid > Ask in calls")
            
            # Check for zero implied volatility
            if 'impliedVolatility' in calls.columns:
                zero_iv = (calls['impliedVolatility'] == 0).sum()
                if zero_iv > len(calls) * 0.5:  # More than 50%
                    issues.append(f"{zero_iv} calls with zero IV")
        
        # Check puts
        if not puts.empty:
            # Check for negative bid/ask
            if (puts[['bid', 'ask']] < 0).any().any():
                issues.append("Negative bid/ask in puts")
            
            # Check bid <= ask
            if (puts['bid'] > puts['ask']).any():
                issues.append("Bid > Ask in puts")
            
            # Check for zero implied volatility
            if 'impliedVolatility' in puts.columns:
                zero_iv = (puts['impliedVolatility'] == 0).sum()
                if zero_iv > len(puts) * 0.5:
                    issues.append(f"{zero_iv} puts with zero IV")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def fetch_all_data(self) -> Dict:
        """
        Fetch all data for configured tickers
        
        Returns:
            Dictionary with price and options data for all tickers
        """
        results = {
            'price_data': {},
            'options_data': {},
            'validation': {}
        }
        
        print("=" * 60)
        print("FETCHING MARKET DATA")
        print("=" * 60)
        
        for ticker in self.tickers:
            print(f"\n📊 Processing {ticker}...")
            
            # Fetch price data
            price_df = self.fetch_price_data(ticker, period='5y', interval='1d')
            results['price_data'][ticker] = price_df
            
            # Validate price data
            if not price_df.empty:
                is_valid, issues = self.validate_price_data(price_df)
                results['validation'][f'{ticker}_price'] = {
                    'valid': is_valid,
                    'issues': issues
                }
                
                if not is_valid:
                    print(f"⚠️  Price data issues: {', '.join(issues)}")
            
            # Fetch options data
            options_dict = self.fetch_options_chain(ticker)
            results['options_data'][ticker] = options_dict
            
            # Validate options data
            is_valid, issues = self.validate_options_data(options_dict)
            results['validation'][f'{ticker}_options'] = {
                'valid': is_valid,
                'issues': issues
            }
            
            if not is_valid:
                print(f"⚠️  Options data issues: {', '.join(issues)}")
        
        print("\n" + "=" * 60)
        print("DATA FETCH COMPLETE")
        print("=" * 60)
        
        return results
    
    def get_data_summary(self, results: Dict) -> pd.DataFrame:
        """
        Generate summary statistics for fetched data
        
        Args:
            results: Results from fetch_all_data()
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for ticker in self.tickers:
            price_df = results['price_data'].get(ticker, pd.DataFrame())
            options_dict = results['options_data'].get(ticker, {})
            
            summary_data.append({
                'Ticker': ticker,
                'Price Rows': len(price_df),
                'Date Range': f"{price_df.index.min():%Y-%m-%d} to {price_df.index.max():%Y-%m-%d}" if not price_df.empty else "N/A",
                'Call Contracts': len(options_dict.get('calls', [])),
                'Put Contracts': len(options_dict.get('puts', [])),
                'Options Expirations': len(options_dict.get('expirations', [])),
                'Price Valid': results['validation'].get(f'{ticker}_price', {}).get('valid', False),
                'Options Valid': results['validation'].get(f'{ticker}_options', {}).get('valid', False)
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Test the data pipeline"""
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Fetch all data
    results = pipeline.fetch_all_data()
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    summary = pipeline.get_data_summary(results)
    print(summary.to_string(index=False))
    
    # Show sample price data for SPY
    if 'SPY' in results['price_data'] and not results['price_data']['SPY'].empty:
        print("\n" + "=" * 60)
        print("SPY PRICE DATA SAMPLE (Last 5 Days)")
        print("=" * 60)
        print(results['price_data']['SPY'].tail())
    
    # Show sample options data for SPY
    if 'SPY' in results['options_data']:
        spy_options = results['options_data']['SPY']
        if not spy_options['calls'].empty:
            print("\n" + "=" * 60)
            print("SPY CALL OPTIONS SAMPLE (First 5 Contracts)")
            print("=" * 60)
            print(spy_options['calls'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head())
        
        if not spy_options['puts'].empty:
            print("\n" + "=" * 60)
            print("SPY PUT OPTIONS SAMPLE (First 5 Contracts)")
            print("=" * 60)
            print(spy_options['puts'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head())
    
    return results


if __name__ == "__main__":
    main()
