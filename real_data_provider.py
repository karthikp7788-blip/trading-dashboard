"""
Real Data Provider with Multiple Sources
Tries Massive.io first, falls back to yfinance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf


class RealDataProvider:
    """Unified interface for real market data"""
    
    def __init__(self, massive_api_key: Optional[str] = None):
        """
        Initialize data provider
        
        Args:
            massive_api_key: Massive.io API key (optional)
        """
        self.massive_api_key = massive_api_key
        self.provider = 'yfinance'  # Default to yfinance
        
        # Try Massive.io if key provided
        if massive_api_key:
            try:
                from massive_data import MassiveDataProvider
                self.massive_provider = MassiveDataProvider(massive_api_key)
                if self.massive_provider.test_connection():
                    self.provider = 'massive'
                    print("✓ Using Massive.io for market data")
                else:
                    print("⚠ Massive.io connection failed, using yfinance")
            except:
                print("⚠ Could not load Massive.io, using yfinance")
    
    def get_stock_data(self,
                      ticker: str,
                      start_date: str,
                      end_date: str,
                      interval: str = '1d') -> pd.DataFrame:
        """
        Get historical stock data
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.provider == 'massive':
            return self._get_from_massive(ticker, start_date, end_date, interval)
        else:
            return self._get_from_yfinance(ticker, start_date, end_date, interval)
    
    def _get_from_yfinance(self,
                          ticker: str,
                          start_date: str,
                          end_date: str,
                          interval: str) -> pd.DataFrame:
        """Get data from yfinance"""
        
        print(f"📡 Fetching {ticker} from yfinance ({start_date} to {end_date})...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"⚠️  No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean and standardize
            df = df.dropna()
            df['Ticker'] = ticker
            
            # Ensure column names are correct
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            print(f"✓ Retrieved {len(df)} bars for {ticker}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
        
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _get_from_massive(self,
                         ticker: str,
                         start_date: str,
                         end_date: str,
                         interval: str) -> pd.DataFrame:
        """Get data from Massive.io"""
        
        # Map interval format
        interval_map = {
            '1d': '1Day',
            '1h': '1Hour',
            '5m': '5Min',
            '15m': '15Min'
        }
        
        massive_interval = interval_map.get(interval, '1Day')
        
        return self.massive_provider.get_stock_bars(
            ticker, start_date, end_date, massive_interval
        )
    
    def get_options_chain(self, ticker: str) -> Dict:
        """
        Get options chain
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with calls and puts
        """
        if self.provider == 'massive':
            return self.massive_provider.get_options_chain(ticker)
        else:
            return self._get_options_from_yfinance(ticker)
    
    def _get_options_from_yfinance(self, ticker: str) -> Dict:
        """Get options from yfinance"""
        
        print(f"📡 Fetching options chain for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                print(f"⚠️  No options available for {ticker}")
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Add metadata
            calls['expiration'] = nearest_exp
            puts['expiration'] = nearest_exp
            calls['ticker'] = ticker
            puts['ticker'] = ticker
            
            print(f"✓ Retrieved {len(calls)} calls, {len(puts)} puts (exp: {nearest_exp})")
            
            return {
                'calls': calls,
                'puts': puts,
                'nearest_expiration': nearest_exp,
                'expirations': expirations
            }
        
        except Exception as e:
            print(f"✗ Error fetching options: {str(e)}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
    
    def get_latest_price(self, ticker: str) -> float:
        """Get latest price"""
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return 0.0
        except:
            return 0.0


def main():
    """Test real data provider"""
    
    print("=" * 70)
    print("REAL DATA PROVIDER TEST")
    print("=" * 70)
    
    # Initialize (will use yfinance since Massive.io is blocked)
    provider = RealDataProvider(massive_api_key="xxu7bdpXlzs9EwWXFk_x0BDkM25FSBFg")
    
    print(f"\n✓ Using provider: {provider.provider}")
    
    # Test SPY data
    print("\n" + "=" * 70)
    print("FETCHING SPY DATA (LAST YEAR)")
    print("=" * 70)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    spy_data = provider.get_stock_data('SPY', start_date, end_date)
    
    if not spy_data.empty:
        print(f"\n✓ Data retrieved successfully")
        print(f"\nLast 5 days:")
        print(spy_data.tail())
        print(f"\nStats:")
        print(f"  Date range: {spy_data.index.min():%Y-%m-%d} to {spy_data.index.max():%Y-%m-%d}")
        print(f"  Total days: {len(spy_data)}")
        print(f"  Latest close: ${spy_data['Close'].iloc[-1]:.2f}")
    
    # Test options
    print("\n" + "=" * 70)
    print("FETCHING SPY OPTIONS")
    print("=" * 70)
    
    options = provider.get_options_chain('SPY')
    
    if not options['calls'].empty:
        print(f"\n✓ Options retrieved successfully")
        print(f"\nStats:")
        print(f"  Calls: {len(options['calls'])}")
        print(f"  Puts: {len(options['puts'])}")
        print(f"  Expiration: {options['nearest_expiration']}")
        
        print(f"\nSample call options:")
        print(options['calls'][['strike', 'lastPrice', 'volume', 'impliedVolatility']].head())
    
    print("\n" + "=" * 70)
    print("REAL DATA PROVIDER READY ✓")
    print("=" * 70)
    
    return provider


if __name__ == "__main__":
    provider = main()
