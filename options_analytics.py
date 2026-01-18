"""
Options Analytics Module
Calculates options-derived features: skew, premium imbalance, OI concentration, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import norm


class OptionsAnalytics:
    """Calculate options-derived signals and features"""
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes call option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes put option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
        """
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def calculate_skew_metrics(calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float) -> Dict:
        """
        Calculate options skew metrics
        
        Args:
            calls: DataFrame with call options
            puts: DataFrame with put options
            spot_price: Current stock price
            
        Returns:
            Dictionary with skew metrics
        """
        if calls.empty or puts.empty:
            return {}
        
        try:
            # Filter for valid IV
            calls = calls[calls['impliedVolatility'] > 0].copy()
            puts = puts[puts['impliedVolatility'] > 0].copy()
            
            if calls.empty or puts.empty:
                return {}
            
            # Calculate moneyness (K/S)
            calls['moneyness'] = calls['strike'] / spot_price
            puts['moneyness'] = puts['strike'] / spot_price
            
            # ATM definition: strikes within 2% of spot
            atm_calls = calls[(calls['moneyness'] >= 0.98) & (calls['moneyness'] <= 1.02)]
            atm_puts = puts[(puts['moneyness'] >= 0.98) & (puts['moneyness'] <= 1.02)]
            
            # OTM puts (protective puts, bearish positioning)
            otm_puts = puts[puts['moneyness'] < 0.95]
            
            # OTM calls (upside bets)
            otm_calls = calls[calls['moneyness'] > 1.05]
            
            metrics = {}
            
            # ATM IV levels
            if not atm_calls.empty:
                metrics['atm_call_iv'] = atm_calls['impliedVolatility'].mean()
            if not atm_puts.empty:
                metrics['atm_put_iv'] = atm_puts['impliedVolatility'].mean()
            
            # Put-Call IV skew
            if 'atm_call_iv' in metrics and 'atm_put_iv' in metrics:
                metrics['pc_iv_skew'] = metrics['atm_put_iv'] - metrics['atm_call_iv']
            
            # OTM skew (risk reversal proxy)
            if not otm_puts.empty and not otm_calls.empty:
                otm_put_iv = otm_puts['impliedVolatility'].mean()
                otm_call_iv = otm_calls['impliedVolatility'].mean()
                metrics['otm_skew'] = otm_put_iv - otm_call_iv
            
            # Skew slope (how steep is the smile)
            if len(puts) > 5:
                # Fit linear regression on moneyness vs IV
                from scipy.stats import linregress
                valid_puts = puts[['moneyness', 'impliedVolatility']].dropna()
                if len(valid_puts) > 5:
                    slope, _, _, _, _ = linregress(valid_puts['moneyness'], valid_puts['impliedVolatility'])
                    metrics['put_skew_slope'] = slope
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating skew: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_premium_metrics(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
        """
        Calculate premium imbalance and activity metrics
        
        Args:
            calls: DataFrame with call options
            puts: DataFrame with put options
            
        Returns:
            Dictionary with premium metrics
        """
        if calls.empty or puts.empty:
            return {}
        
        try:
            # Calculate total premium (price * volume)
            calls['premium'] = calls['lastPrice'] * calls['volume']
            puts['premium'] = puts['lastPrice'] * puts['volume']
            
            total_call_premium = calls['premium'].sum()
            total_put_premium = puts['premium'].sum()
            
            metrics = {
                'total_call_premium': total_call_premium,
                'total_put_premium': total_put_premium,
            }
            
            # Premium ratio
            if total_put_premium > 0:
                metrics['call_put_premium_ratio'] = total_call_premium / total_put_premium
            
            # Premium imbalance
            total_premium = total_call_premium + total_put_premium
            if total_premium > 0:
                metrics['premium_imbalance'] = (total_call_premium - total_put_premium) / total_premium
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating premium metrics: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_oi_concentration(calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float) -> Dict:
        """
        Calculate open interest concentration and pin zones
        
        Args:
            calls: DataFrame with call options
            puts: DataFrame with put options
            spot_price: Current stock price
            
        Returns:
            Dictionary with OI metrics and pin zones
        """
        if calls.empty or puts.empty:
            return {}
        
        try:
            # Combine calls and puts
            all_options = pd.concat([
                calls[['strike', 'openInterest']].assign(type='call'),
                puts[['strike', 'openInterest']].assign(type='put')
            ])
            
            # Group by strike
            oi_by_strike = all_options.groupby('strike')['openInterest'].sum().sort_values(ascending=False)
            
            # Find top 5 strikes by OI
            top_strikes = oi_by_strike.head(5)
            
            # Calculate max pain (strike with most OI)
            max_pain_strike = oi_by_strike.idxmax()
            
            # Find strikes near spot (within 5%)
            nearby_strikes = oi_by_strike[
                (oi_by_strike.index >= spot_price * 0.95) & 
                (oi_by_strike.index <= spot_price * 1.05)
            ]
            
            metrics = {
                'max_pain_strike': max_pain_strike,
                'max_pain_distance': (max_pain_strike - spot_price) / spot_price,
                'total_oi': oi_by_strike.sum(),
                'top_5_strikes': top_strikes.index.tolist(),
                'top_5_oi': top_strikes.values.tolist(),
            }
            
            # Gamma concentration zones
            if not nearby_strikes.empty:
                metrics['nearby_oi_concentration'] = nearby_strikes.sum() / oi_by_strike.sum()
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating OI concentration: {str(e)}")
            return {}
    
    @staticmethod
    def estimate_fair_value(option_type: str, 
                           S: float, 
                           K: float, 
                           T: float, 
                           IV: float, 
                           r: float = 0.05) -> float:
        """
        Estimate theoretical fair value using Black-Scholes
        
        Args:
            option_type: 'call' or 'put'
            S: Spot price
            K: Strike
            T: Time to expiration (years)
            IV: Implied volatility
            r: Risk-free rate
            
        Returns:
            Theoretical fair value
        """
        if option_type == 'call':
            return OptionsAnalytics.black_scholes_call(S, K, T, r, IV)
        else:
            return OptionsAnalytics.black_scholes_put(S, K, T, r, IV)
    
    @staticmethod
    def analyze_rich_cheap(calls: pd.DataFrame, 
                          puts: pd.DataFrame, 
                          spot_price: float,
                          days_to_expiry: int) -> Dict:
        """
        Classify options as rich or cheap vs fair value
        
        Args:
            calls: DataFrame with call options
            puts: DataFrame with put options
            spot_price: Current stock price
            days_to_expiry: Days until expiration
            
        Returns:
            Dictionary with rich/cheap analysis
        """
        if calls.empty or puts.empty:
            return {}
        
        try:
            T = days_to_expiry / 365.0
            
            # Calculate theoretical values for calls
            calls = calls.copy()
            calls['fair_value'] = calls.apply(
                lambda row: OptionsAnalytics.estimate_fair_value(
                    'call', spot_price, row['strike'], T, row['impliedVolatility']
                ) if row['impliedVolatility'] > 0 else 0,
                axis=1
            )
            
            calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
            calls['vs_fair'] = (calls['mid_price'] - calls['fair_value']) / calls['fair_value']
            
            # Calculate theoretical values for puts
            puts = puts.copy()
            puts['fair_value'] = puts.apply(
                lambda row: OptionsAnalytics.estimate_fair_value(
                    'put', spot_price, row['strike'], T, row['impliedVolatility']
                ) if row['impliedVolatility'] > 0 else 0,
                axis=1
            )
            
            puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
            puts['vs_fair'] = (puts['mid_price'] - puts['fair_value']) / puts['fair_value']
            
            # Classify as rich (> 10% above fair) or cheap (> 10% below fair)
            rich_calls = calls[calls['vs_fair'] > 0.10]
            cheap_calls = calls[calls['vs_fair'] < -0.10]
            rich_puts = puts[puts['vs_fair'] > 0.10]
            cheap_puts = puts[puts['vs_fair'] < -0.10]
            
            metrics = {
                'rich_calls_count': len(rich_calls),
                'cheap_calls_count': len(cheap_calls),
                'rich_puts_count': len(rich_puts),
                'cheap_puts_count': len(cheap_puts),
                'avg_call_vs_fair': calls['vs_fair'].mean(),
                'avg_put_vs_fair': puts['vs_fair'].mean(),
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing rich/cheap: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_all_metrics(options_dict: Dict, spot_price: float) -> Dict:
        """
        Calculate all options metrics
        
        Args:
            options_dict: Dictionary with calls and puts DataFrames
            spot_price: Current stock price
            
        Returns:
            Dictionary with all options metrics
        """
        calls = options_dict.get('calls', pd.DataFrame())
        puts = options_dict.get('puts', pd.DataFrame())
        
        if calls.empty or puts.empty:
            return {}
        
        print("Calculating options metrics...")
        
        all_metrics = {}
        
        # Skew metrics
        skew = OptionsAnalytics.calculate_skew_metrics(calls, puts, spot_price)
        all_metrics.update({f'skew_{k}': v for k, v in skew.items()})
        
        # Premium metrics
        premium = OptionsAnalytics.calculate_premium_metrics(calls, puts)
        all_metrics.update({f'premium_{k}': v for k, v in premium.items()})
        
        # OI concentration
        oi = OptionsAnalytics.calculate_oi_concentration(calls, puts, spot_price)
        all_metrics.update({f'oi_{k}': v for k, v in oi.items()})
        
        # Rich/cheap analysis (estimate days to expiry from expiration date)
        if 'nearest_expiration' in options_dict:
            from datetime import datetime
            exp_date = datetime.strptime(options_dict['nearest_expiration'], '%Y-%m-%d')
            days_to_expiry = (exp_date - datetime.now()).days
            
            if days_to_expiry > 0:
                rich_cheap = OptionsAnalytics.analyze_rich_cheap(calls, puts, spot_price, days_to_expiry)
                all_metrics.update({f'richcheap_{k}': v for k, v in rich_cheap.items()})
        
        print(f"✓ Calculated {len(all_metrics)} options metrics")
        
        return all_metrics


def main():
    """Test options analytics"""
    from data_pipeline import DataPipeline
    
    # Fetch SPY data
    pipeline = DataPipeline(['SPY'])
    results = pipeline.fetch_all_data()
    
    spy_options = results['options_data']['SPY']
    spy_price = results['price_data']['SPY']
    
    if not spy_price.empty and spy_options['calls'].shape[0] > 0:
        current_price = spy_price['Close'].iloc[-1]
        
        print("\n" + "=" * 60)
        print(f"OPTIONS ANALYTICS FOR SPY (Spot: ${current_price:.2f})")
        print("=" * 60)
        
        # Calculate all metrics
        metrics = OptionsAnalytics.calculate_all_metrics(spy_options, current_price)
        
        # Print results
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, list):
                print(f"{key}: {value[:3]}...")  # Show first 3 items
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
