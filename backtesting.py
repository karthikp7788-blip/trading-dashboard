"""
Backtesting Framework
Walk-forward validation and strategy simulation for trading models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """Backtest trading strategies with realistic assumptions"""
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.0,
                 slippage: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade ($ or %)
            slippage: Slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.equity_curve = []
        
    def walk_forward_validation(self,
                                model,
                                data: pd.DataFrame,
                                feature_cols: List[str],
                                target_col: str,
                                train_window: int = 252,
                                test_window: int = 21,
                                retrain_frequency: int = 21) -> pd.DataFrame:
        """
        Walk-forward validation: train on rolling window, test on next period
        
        Args:
            model: Model object with train() and predict_proba() methods
            data: Full dataset with features and targets
            feature_cols: List of feature column names
            target_col: Target column name
            train_window: Days to use for training
            test_window: Days to test on
            retrain_frequency: How often to retrain (days)
            
        Returns:
            DataFrame with predictions and actual values
        """
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION")
        print("="*70)
        
        print(f"\n📊 Configuration:")
        print(f"   Training window: {train_window} days (~1 year)")
        print(f"   Test window: {test_window} days (~1 month)")
        print(f"   Retrain frequency: {retrain_frequency} days")
        
        results = []
        start_idx = train_window
        
        iterations = 0
        while start_idx + test_window < len(data):
            iterations += 1
            
            # Training data
            train_start = start_idx - train_window
            train_end = start_idx
            train_data = data.iloc[train_start:train_end]
            
            # Test data
            test_start = start_idx
            test_end = min(start_idx + test_window, len(data))
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            
            if iterations == 1 or iterations % (retrain_frequency // test_window) == 0:
                print(f"\n🔄 Iteration {iterations}: Training on {train_data.index[0]:%Y-%m-%d} to {train_data.index[-1]:%Y-%m-%d}")
                model.train(X_train, y_train, verbose=False)
            
            # Predict on test data
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            predictions = model.predict_with_confidence(X_test)
            
            # Store results
            for idx in predictions.index:
                results.append({
                    'date': idx,
                    'actual': y_test.loc[idx],
                    'prediction': predictions.loc[idx, 'prediction'],
                    'prob_up': predictions.loc[idx, 'prob_up'],
                    'confidence': predictions.loc[idx, 'confidence'],
                    'confidence_level': predictions.loc[idx, 'confidence_level']
                })
            
            # Move forward
            start_idx += test_window
        
        results_df = pd.DataFrame(results).set_index('date')
        
        print(f"\n✓ Completed {iterations} iterations")
        print(f"✓ Total predictions: {len(results_df)}")
        
        return results_df
    
    def simulate_strategy(self,
                         predictions: pd.DataFrame,
                         price_data: pd.DataFrame,
                         confidence_threshold: float = 0.6,
                         position_size: float = 1.0) -> pd.DataFrame:
        """
        Simulate trading strategy based on predictions
        
        Args:
            predictions: DataFrame with predictions and confidence
            price_data: DataFrame with OHLCV data
            confidence_threshold: Minimum confidence to trade
            position_size: Fraction of capital to use per trade
            
        Returns:
            DataFrame with trades and equity curve
        """
        print("\n" + "="*70)
        print("STRATEGY SIMULATION")
        print("="*70)
        
        print(f"\n📊 Strategy Parameters:")
        print(f"   Initial capital: ${self.initial_capital:,.0f}")
        print(f"   Confidence threshold: {confidence_threshold:.0%}")
        print(f"   Position size: {position_size:.0%} of capital")
        print(f"   Commission: ${self.commission:.2f} per trade")
        print(f"   Slippage: {self.slippage:.1%}")
        
        capital = self.initial_capital
        position = None
        trades = []
        equity = []
        
        # Align predictions with prices
        common_dates = predictions.index.intersection(price_data.index)
        predictions = predictions.loc[common_dates]
        
        for date in predictions.index[:-1]:  # Skip last day (can't exit)
            current_price = price_data.loc[date, 'Close']
            next_date = predictions.index[predictions.index > date][0]
            next_price = price_data.loc[next_date, 'Close']
            
            pred = predictions.loc[date]
            
            # Entry logic
            if position is None and pred['confidence'] >= confidence_threshold:
                # Enter position
                entry_price = current_price * (1 + self.slippage)
                shares = int((capital * position_size) / entry_price)
                cost = shares * entry_price + self.commission
                
                if cost <= capital:
                    position = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'direction': pred['prediction'],
                        'confidence': pred['confidence']
                    }
                    capital -= cost
            
            # Exit logic (exit after 1 day)
            elif position is not None:
                exit_price = next_price * (1 - self.slippage)
                proceeds = position['shares'] * exit_price - self.commission
                
                # Calculate P&L
                actual_return = (next_price / current_price) - 1
                
                if position['direction'] == 1:  # Long
                    pnl = proceeds - (position['shares'] * position['entry_price'])
                else:  # Short (inverse)
                    pnl = -(proceeds - (position['shares'] * position['entry_price']))
                
                capital += proceeds
                
                # Record trade
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': next_date,
                    'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return': pnl / (position['shares'] * position['entry_price']),
                    'confidence': position['confidence'],
                    'actual_direction': 1 if actual_return > 0 else 0,
                    'correct': position['direction'] == (1 if actual_return > 0 else 0)
                })
                
                position = None
            
            # Record equity
            equity.append({
                'date': date,
                'capital': capital,
                'position_value': position['shares'] * current_price if position else 0,
                'total_equity': capital + (position['shares'] * current_price if position else 0)
            })
        
        self.trades = pd.DataFrame(trades)
        self.equity_curve = pd.DataFrame(equity).set_index('date')
        
        print(f"\n✓ Simulation complete")
        print(f"✓ Total trades: {len(self.trades)}")
        
        return self.trades
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.trades) == 0:
            return {}
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        
        trades = self.trades
        equity = self.equity_curve
        
        # Basic stats
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L stats
        total_pnl = trades['pnl'].sum()
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades[trades['pnl'] > 0]['pnl'].sum() / 
                           trades[trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        # Accuracy
        accuracy = trades['correct'].mean()
        
        # Returns
        total_return = (equity['total_equity'].iloc[-1] / self.initial_capital) - 1
        
        # Drawdown
        equity['peak'] = equity['total_equity'].cummax()
        equity['drawdown'] = (equity['total_equity'] - equity['peak']) / equity['peak']
        max_drawdown = equity['drawdown'].min()
        
        # Sharpe ratio (annualized)
        returns = equity['total_equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'expectancy': expectancy
        }
        
        # Print metrics
        print(f"\n📊 Trading Performance:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winners: {winning_trades} ({win_rate:.1%})")
        print(f"   Losers: {losing_trades}")
        print(f"   Prediction Accuracy: {accuracy:.1%}")
        
        print(f"\n💰 Profitability:")
        print(f"   Total P&L: ${total_pnl:,.0f}")
        print(f"   Total Return: {total_return:.1%}")
        print(f"   Average Win: ${avg_win:,.0f}")
        print(f"   Average Loss: ${avg_loss:,.0f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Expectancy: ${expectancy:,.2f}")
        
        print(f"\n📉 Risk Metrics:")
        print(f"   Max Drawdown: {max_drawdown:.1%}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        
        return metrics
    
    def compare_strategies(self,
                          results1: Dict,
                          results2: Dict,
                          name1: str = "Strategy 1",
                          name2: str = "Strategy 2") -> None:
        """
        Compare two strategies side by side
        
        Args:
            results1: Metrics from first strategy
            results2: Metrics from second strategy
            name1: Name of first strategy
            name2: Name of second strategy
        """
        print("\n" + "="*70)
        print(f"STRATEGY COMPARISON: {name1} vs {name2}")
        print("="*70)
        
        metrics = [
            ('Win Rate', 'win_rate', '%'),
            ('Accuracy', 'accuracy', '%'),
            ('Total Return', 'total_return', '%'),
            ('Profit Factor', 'profit_factor', ''),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Total Trades', 'total_trades', ''),
        ]
        
        print(f"\n{'Metric':<20} {name1:>20} {name2:>20} {'Difference':>15}")
        print("-" * 80)
        
        for metric_name, metric_key, unit in metrics:
            val1 = results1.get(metric_key, 0)
            val2 = results2.get(metric_key, 0)
            diff = val2 - val1
            
            if unit == '%':
                str1 = f"{val1*100:>18.1f}%"
                str2 = f"{val2*100:>18.1f}%"
                diff_str = f"{diff*100:+13.1f}%"
            elif unit == '':
                str1 = f"{val1:>20.2f}"
                str2 = f"{val2:>20.2f}"
                diff_str = f"{diff:+15.2f}"
            
            # Highlight improvement
            if (metric_key in ['win_rate', 'accuracy', 'total_return', 'profit_factor', 'sharpe_ratio'] and diff > 0) or \
               (metric_key == 'max_drawdown' and diff > 0):  # Less negative is better
                symbol = "✓"
            elif diff < 0:
                symbol = "✗"
            else:
                symbol = "="
            
            print(f"{metric_name:<20} {str1} {str2} {diff_str} {symbol}")


def main():
    """Test backtesting framework"""
    from real_data_provider import RealDataProvider
    from technical_indicators import TechnicalIndicators
    from feature_engineering import FeatureEngineering
    from xgboost_model import XGBoostPredictor
    
    print("="*70)
    print("BACKTESTING FRAMEWORK DEMONSTRATION")
    print("="*70)
    
    # Get real SPY data
    print("\n📡 Fetching real SPY data...")
    provider = RealDataProvider()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
    
    spy_data = provider.get_stock_data('SPY', start_date, end_date)
    
    print(f"✓ Retrieved {len(spy_data)} days of data")
    
    # Calculate indicators
    print("\n📈 Calculating technical indicators...")
    spy_with_indicators = TechnicalIndicators.calculate_all_indicators(spy_data)
    
    # Create features
    print("\n🔧 Engineering features...")
    fe = FeatureEngineering()
    ml_dataset = fe.create_ml_dataset(spy_with_indicators, target_horizon=1)
    
    # Initialize backtester
    backtester = Backtester(initial_capital=100000, commission=1.0, slippage=0.001)
    
    # Initialize model
    model = XGBoostPredictor()
    
    # Walk-forward validation
    predictions = backtester.walk_forward_validation(
        model=model,
        data=ml_dataset,
        feature_cols=fe.feature_columns,
        target_col='target_direction_1d',
        train_window=252,
        test_window=21,
        retrain_frequency=21
    )
    
    # Simulate strategy
    trades = backtester.simulate_strategy(
        predictions=predictions,
        price_data=spy_data,
        confidence_threshold=0.6,
        position_size=1.0
    )
    
    # Calculate metrics
    metrics = backtester.calculate_metrics()
    
    print("\n" + "="*70)
    print("BACKTESTING COMPLETE ✓")
    print("="*70)
    
    return backtester, metrics


if __name__ == "__main__":
    backtester, metrics = main()
