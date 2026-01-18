"""
Massive.io API Integration
Real market data provider for stock prices and options chains
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class MassiveDataProvider:
    """Interface to Massive.io market data API"""
    
    def __init__(self, api_key: str):
        """
        Initialize Massive data provider
        
        Args:
            api_key: Massive.io API key
        """
        self.api_key = api_key
        self.base_url = "https://api.massive.io/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise ValueError("Invalid API key")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded")
            else:
                raise ValueError(f"API error: {e}")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {e}")
    
    def get_stock_bars(self,
                      ticker: str,
                      start_date: str,
                      end_date: str,
                      timeframe: str = '1Day') -> pd.DataFrame:
        """
        Get historical stock price data
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"stocks/{ticker}/bars"
        
        params = {
            'start': start_date,
            'end': end_date,
            'timeframe': timeframe
        }
        
        print(f"📡 Fetching {ticker} data from {start_date} to {end_date}...")
        
        try:
            data = self._make_request(endpoint, params)
            
            # Parse response (format may vary by API)
            if 'bars' in data:
                bars = data['bars']
            elif 'results' in data:
                bars = data['results']
            else:
                bars = data
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            
            if df.empty:
                print(f"⚠️  No data returned for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            column_map = {
                't': 'Date',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_map)
            
            # Convert timestamp to datetime
            if 'Date' in df.columns:
                if df['Date'].dtype in ['int64', 'float64']:
                    # Unix timestamp (milliseconds)
                    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                else:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                df = df.set_index('Date')
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Add ticker
            df['Ticker'] = ticker
            
            # Sort by date
            df = df.sort_index()
            
            print(f"✓ Retrieved {len(df)} bars for {ticker}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
        
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_options_chain(self,
                         ticker: str,
                         expiration_date: Optional[str] = None) -> Dict:
        """
        Get options chain data
        
        Args:
            ticker: Stock symbol
            expiration_date: Specific expiration (YYYY-MM-DD), or None for nearest
            
        Returns:
            Dictionary with calls and puts DataFrames
        """
        endpoint = f"options/{ticker}/chain"
        
        params = {}
        if expiration_date:
            params['expiration'] = expiration_date
        
        print(f"📡 Fetching options chain for {ticker}...")
        
        try:
            data = self._make_request(endpoint, params)
            
            # Parse calls and puts
            calls_data = []
            puts_data = []
            
            if 'calls' in data and 'puts' in data:
                calls_data = data['calls']
                puts_data = data['puts']
            elif 'results' in data:
                for contract in data['results']:
                    if contract.get('type') == 'call':
                        calls_data.append(contract)
                    elif contract.get('type') == 'put':
                        puts_data.append(contract)
            
            # Convert to DataFrames
            calls = pd.DataFrame(calls_data) if calls_data else pd.DataFrame()
            puts = pd.DataFrame(puts_data) if puts_data else pd.DataFrame()
            
            # Standardize column names
            column_map = {
                'strike_price': 'strike',
                'last_price': 'lastPrice',
                'bid_price': 'bid',
                'ask_price': 'ask',
                'open_interest': 'openInterest',
                'implied_volatility': 'impliedVolatility'
            }
            
            if not calls.empty:
                calls = calls.rename(columns=column_map)
            if not puts.empty:
                puts = puts.rename(columns=column_map)
            
            # Get expiration date
            exp_date = expiration_date
            if not exp_date and not calls.empty and 'expiration' in calls.columns:
                exp_date = calls['expiration'].iloc[0]
            
            print(f"✓ Retrieved {len(calls)} calls, {len(puts)} puts")
            
            return {
                'calls': calls,
                'puts': puts,
                'nearest_expiration': exp_date,
                'expirations': [exp_date] if exp_date else []
            }
        
        except Exception as e:
            print(f"✗ Error fetching options: {str(e)}")
            return {
                'calls': pd.DataFrame(),
                'puts': pd.DataFrame(),
                'nearest_expiration': None,
                'expirations': []
            }
    
    def get_latest_price(self, ticker: str) -> float:
        """
        Get latest price for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Latest close price
        """
        endpoint = f"stocks/{ticker}/quote"
        
        try:
            data = self._make_request(endpoint)
            
            # Try different response formats
            if 'price' in data:
                return float(data['price'])
            elif 'last' in data:
                return float(data['last'])
            elif 'close' in data:
                return float(data['close'])
            elif 'c' in data:
                return float(data['c'])
            else:
                # Fallback: get latest bar
                today = datetime.now().strftime('%Y-%m-%d')
                yesterday = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                bars = self.get_stock_bars(ticker, yesterday, today)
                if not bars.empty:
                    return float(bars['Close'].iloc[-1])
                return 0.0
        
        except Exception as e:
            print(f"✗ Error fetching latest price: {str(e)}")
            return 0.0
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful
        """
        print("\n🔌 Testing Massive.io API connection...")
        
        try:
            # Try to get SPY data for last week
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            data = self.get_stock_bars('SPY', start_date, end_date)
            
            if not data.empty:
                print("✓ Connection successful!")
                print(f"✓ Retrieved {len(data)} days of SPY data")
                print(f"✓ Latest close: ${data['Close'].iloc[-1]:.2f}")
                return True
            else:
                print("✗ Connection failed: No data returned")
                return False
        
        except Exception as e:
            print(f"✗ Connection failed: {str(e)}")
            return False


def main():
    """Test Massive.io data provider"""
    
    # API key
    API_KEY = "xxu7bdpXlzs9EwWXFk_x0BDkM25FSBFg"
    
    print("=" * 70)
    print("MASSIVE.IO API INTEGRATION TEST")
    print("=" * 70)
    
    # Initialize provider
    provider = MassiveDataProvider(API_KEY)
    
    # Test connection
    if not provider.test_connection():
        print("\n❌ Cannot connect to Massive.io API")
        print("\nPossible issues:")
        print("  • Invalid API key")
        print("  • API endpoint has changed")
        print("  • Network restrictions")
        print("  • Free tier limitations")
        return None
    
    # Try to fetch SPY data
    print("\n" + "=" * 70)
    print("FETCHING SPY HISTORICAL DATA")
    print("=" * 70)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    spy_data = provider.get_stock_bars('SPY', start_date, end_date)
    
    if not spy_data.empty:
        print(f"\n✓ Successfully retrieved SPY data")
        print(f"\nData summary:")
        print(spy_data.tail())
        print(f"\nDate range: {spy_data.index.min():%Y-%m-%d} to {spy_data.index.max():%Y-%m-%d}")
        print(f"Total days: {len(spy_data)}")
    
    # Try to fetch options
    print("\n" + "=" * 70)
    print("FETCHING SPY OPTIONS CHAIN")
    print("=" * 70)
    
    options = provider.get_options_chain('SPY')
    
    if not options['calls'].empty:
        print(f"\n✓ Successfully retrieved options data")
        print(f"\nCalls: {len(options['calls'])} contracts")
        print(f"Puts: {len(options['puts'])} contracts")
        print(f"Expiration: {options['nearest_expiration']}")
    
    return provider


if __name__ == "__main__":
    provider = main()
