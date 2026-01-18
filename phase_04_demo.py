"""
Phase 0.4: Backtesting Demo
Complete demonstration of strategy validation on real market data
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from real_data_provider import RealDataProvider
from technical_indicators import TechnicalIndicators
from feature_engineering import FeatureEngineering
from xgboost_model import XGBoostPredictor
from backtesting import Backtester


def print_section(title, char="="):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}")


def main():
    """Run complete Phase 0.4 demonstration"""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  TRADING PREDICTION SYSTEM - PHASE 0.4 DEMONSTRATION".center(78) + "║")
    print("║" + "          Backtesting on Real Market Data".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Step 1: Get Real Data
    print_section("STEP 1: FETCHING REAL MARKET DATA")
    
    print("\n📡 Connecting to market data provider...")
    provider = RealDataProvider(massive_api_key="xxu7bdpXlzs9EwWXFk_x0BDkM25FSBFg")
    
    print(f"✓ Using provider: {provider.provider}")
    
    # Get 2 years of SPY data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print(f"\n📊 Fetching SPY data: {start_date} to {end_date}")
    spy_data = provider.get_stock_data('SPY', start_date, end_date)
    
    if spy_data.empty:
        print("\n❌ ERROR: Could not fetch real data")
        print("   This demo requires network access to yfinance")
        print("   The code will work when deployed with internet access")
        return None, None
    
    print(f"✓ Retrieved {len(spy_data)} trading days")
    print(f"✓ Date range: {spy_data.index.min():%Y-%m-%d} to {spy_data.index.max():%Y-%m-%d}")
    print(f"✓ Latest close: ${spy_data['Close'].iloc[-1]:.2f}")
    
    # Step 2: Calculate Indicators
    print_section("STEP 2: CALCULATING TECHNICAL INDICATORS")
    
    spy_with_indicators = TechnicalIndicators.calculate_all_indicators(spy_data)
    
    # Step 3: Feature Engineering
    print_section("STEP 3: FEATURE ENGINEERING")
    
    fe = FeatureEngineering()
    ml_dataset = fe.create_ml_dataset(spy_with_indicators, target_horizon=1)
    
    print(f"✓ Created {len(fe.feature_columns)} features")
    print(f"✓ Dataset: {len(ml_dataset)} days ready for backtesting")
    
    # Step 4: Backtesting Setup
    print_section("STEP 4: BACKTESTING CONFIGURATION")
    
    print("\n🎯 Testing Strategy:")
    print("   • Predict next-day direction (UP/DOWN)")
    print("   • Only trade high-confidence signals (≥60%)")
    print("   • Hold for 1 day, then exit")
    print("   • Position size: 100% of capital")
    print("   • Commission: $1 per trade")
    print("   • Slippage: 0.1% of price")
    
    print("\n📊 Validation Method:")
    print("   • Walk-forward validation (realistic)")
    print("   • Train on 252 days (~1 year)")
    print("   • Test on next 21 days (~1 month)")
    print("   • Retrain every 21 days")
    print("   • No look-ahead bias")
    
    # Step 5: Run Backtest
    print_section("STEP 5: RUNNING BACKTEST ON REAL DATA")
    
    print("\n⏳ This will take a few minutes...")
    print("   Training model multiple times on rolling windows")
    print("   Testing on out-of-sample periods")
    
    # Initialize
    backtester = Backtester(
        initial_capital=100000,
        commission=1.0,
        slippage=0.001
    )
    
    model = XGBoostPredictor()
    
    # Walk-forward validation
    try:
        predictions = backtester.walk_forward_validation(
            model=model,
            data=ml_dataset,
            feature_cols=fe.feature_columns,
            target_col='target_direction_1d',
            train_window=252,
            test_window=21,
            retrain_frequency=21
        )
        
        # Simulate trading strategy
        trades = backtester.simulate_strategy(
            predictions=predictions,
            price_data=spy_data,
            confidence_threshold=0.6,
            position_size=1.0
        )
        
        # Calculate performance
        metrics = backtester.calculate_metrics()
        
    except Exception as e:
        print(f"\n❌ Error during backtesting: {str(e)}")
        print("   This typically happens when dataset is too small")
        return None, None
    
    # Step 6: Results Analysis
    print_section("STEP 6: BACKTEST RESULTS")
    
    if len(trades) == 0:
        print("\n⚠️  No trades generated")
        print("   Possible reasons:")
        print("   • Confidence threshold too high")
        print("   • Insufficient data")
        print("   • Model not confident enough")
        return backtester, metrics
    
    print(f"\n📈 Trade Summary:")
    print(f"   First trade: {trades['entry_date'].min():%Y-%m-%d}")
    print(f"   Last trade: {trades['exit_date'].max():%Y-%m-%d}")
    print(f"   Period: {(trades['exit_date'].max() - trades['entry_date'].min()).days} days")
    
    # Step 7: Performance Breakdown
    print_section("STEP 7: DETAILED PERFORMANCE ANALYSIS")
    
    if metrics['total_trades'] > 0:
        print("\n💡 Key Findings:")
        
        # Win rate analysis
        if metrics['win_rate'] > 0.55:
            print(f"   ✓ Win rate {metrics['win_rate']:.1%} > 55% (GOOD EDGE)")
        elif metrics['win_rate'] > 0.50:
            print(f"   ~ Win rate {metrics['win_rate']:.1%} slightly above 50%")
        else:
            print(f"   ✗ Win rate {metrics['win_rate']:.1%} below 50% (NO EDGE)")
        
        # Accuracy vs win rate
        print(f"\n   Prediction Accuracy: {metrics['accuracy']:.1%}")
        print(f"   Trade Win Rate: {metrics['win_rate']:.1%}")
        if metrics['accuracy'] != metrics['win_rate']:
            print("   Note: Different because of slippage/commissions")
        
        # Profitability
        if metrics['total_return'] > 0:
            print(f"\n   ✓ Total Return: +{metrics['total_return']:.1%} (PROFITABLE)")
            annual_return = metrics['total_return'] * (252 / len(predictions))
            print(f"   Annualized: ~{annual_return:.1%}")
        else:
            print(f"\n   ✗ Total Return: {metrics['total_return']:.1%} (LOSS)")
        
        # Profit factor
        if metrics['profit_factor'] > 1.5:
            print(f"\n   ✓ Profit Factor: {metrics['profit_factor']:.2f} (STRONG)")
        elif metrics['profit_factor'] > 1.0:
            print(f"\n   ~ Profit Factor: {metrics['profit_factor']:.2f} (MARGINAL)")
        else:
            print(f"\n   ✗ Profit Factor: {metrics['profit_factor']:.2f} (LOSING)")
        
        # Risk metrics
        if abs(metrics['max_drawdown']) < 0.10:
            print(f"\n   ✓ Max Drawdown: {metrics['max_drawdown']:.1%} (LOW RISK)")
        elif abs(metrics['max_drawdown']) < 0.20:
            print(f"\n   ~ Max Drawdown: {metrics['max_drawdown']:.1%} (MODERATE)")
        else:
            print(f"\n   ⚠ Max Drawdown: {metrics['max_drawdown']:.1%} (HIGH RISK)")
        
        # Sharpe ratio
        if metrics['sharpe_ratio'] > 1.0:
            print(f"   ✓ Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (GOOD)")
        elif metrics['sharpe_ratio'] > 0:
            print(f"   ~ Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (OK)")
        else:
            print(f"   ✗ Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (BAD)")
        
        # Expectancy
        if metrics['expectancy'] > 0:
            print(f"\n   ✓ Expectancy: ${metrics['expectancy']:.2f} per trade (POSITIVE)")
        else:
            print(f"\n   ✗ Expectancy: ${metrics['expectancy']:.2f} per trade (NEGATIVE)")
    
    # Step 8: Sample Trades
    print_section("STEP 8: SAMPLE TRADES")
    
    if len(trades) >= 3:
        print("\n📋 First 3 trades:")
        print("\n" + "-" * 80)
        for idx, trade in trades.head(3).iterrows():
            result = "WIN ✓" if trade['pnl'] > 0 else "LOSS ✗"
            correct = "✓" if trade['correct'] else "✗"
            print(f"\nTrade {idx + 1}:")
            print(f"  Entry: {trade['entry_date']:%Y-%m-%d} @ ${trade['entry_price']:.2f}")
            print(f"  Exit:  {trade['exit_date']:%Y-%m-%d} @ ${trade['exit_price']:.2f}")
            print(f"  Direction: {trade['direction']} (Prediction {correct})")
            print(f"  Confidence: {trade['confidence']:.1%}")
            print(f"  P&L: ${trade['pnl']:,.2f} ({trade['return']:.1%}) {result}")
    
    # Step 9: What We Learned
    print_section("STEP 9: WHAT WE LEARNED")
    
    print("\n🎯 From This Backtest:")
    
    if metrics.get('total_trades', 0) > 20:
        print(f"\n   ✓ Sufficient trades ({metrics['total_trades']}) for statistical significance")
    elif metrics.get('total_trades', 0) > 0:
        print(f"\n   ⚠ Limited trades ({metrics['total_trades']}) - results may not be reliable")
    
    if metrics.get('accuracy', 0) > 0.50:
        print(f"   ✓ Model can predict direction better than chance")
    else:
        print(f"   ✗ Model not beating random guess on real data")
    
    if metrics.get('total_return', 0) > 0:
        print(f"   ✓ Strategy would have been profitable")
    else:
        print(f"   ✗ Strategy lost money after costs")
    
    print("\n💡 Interpretation:")
    
    if metrics.get('win_rate', 0) >= 0.55 and metrics.get('total_return', 0) > 0:
        print("   🎉 EDGE DETECTED - This system shows promise")
        print("   • Win rate above random")
        print("   • Positive returns after costs")
        print("   • Worth considering for live trading (with caution)")
    elif metrics.get('win_rate', 0) >= 0.52:
        print("   ~ MARGINAL EDGE - Results are borderline")
        print("   • Slight edge over random")
        print("   • Would need optimization")
        print("   • Not ready for live trading yet")
    else:
        print("   ❌ NO EDGE - System doesn't beat market on real data")
        print("   • Results too close to random")
        print("   • Needs improvement before live use")
        print("   • Try: more features, better tuning, different timeframes")
    
    # Step 10: Next Steps
    print_section("PHASE 0.4 COMPLETE - WHAT'S NEXT")
    
    print("\n✅ WHAT WE VALIDATED:")
    print("   ✓ Model trained on real market data")
    print("   ✓ Walk-forward validation (realistic testing)")
    print("   ✓ Transaction costs included")
    print("   ✓ Performance metrics calculated")
    print("   ✓ We know if there's an edge or not")
    
    print("\n📋 IMPROVEMENTS TO TRY:")
    print("   1. Add options features (skew, premium, OI)")
    print("   2. Optimize confidence threshold")
    print("   3. Try different holding periods")
    print("   4. Add market regime filters")
    print("   5. Tune model hyperparameters")
    print("   6. Add more technical features")
    print("   7. Filter by volume/volatility")
    
    print("\n📱 DASHBOARD INTEGRATION:")
    print("   • Show backtest results in dashboard")
    print("   • Display equity curve chart")
    print("   • Trade history table")
    print("   • Performance metrics cards")
    
    print("\n" + "=" * 80)
    print("SYSTEM STATUS: PHASE 0.4 BACKTESTING COMPLETE ✓")
    print("=" * 80 + "\n")
    
    return backtester, metrics


if __name__ == "__main__":
    backtester, metrics = main()
