"""
Enhanced Trading Model
- Options features integration
- Trade management (position sizing, stops, exits)
- Risk-adjusted signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class EnhancedTradingModel:
    """
    Complete trading model with:
    - Options-enhanced predictions
    - Position sizing based on confidence
    - Stop loss and take profit levels
    - Risk management rules
    """
    
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.06,
                 confidence_threshold: float = 0.55):
        """
        Initialize enhanced model
        
        Args:
            max_risk_per_trade: Maximum risk per trade (2% default)
            max_portfolio_risk: Maximum total portfolio risk (6% default)
            confidence_threshold: Minimum confidence to trade
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_columns = []
        
    def add_options_features(self, 
                            price_data: pd.DataFrame,
                            options_metrics: Dict) -> pd.DataFrame:
        """
        Add options-derived features to price data
        
        Args:
            price_data: DataFrame with price/technical data
            options_metrics: Dictionary with options analytics
            
        Returns:
            DataFrame with options features added
        """
        df = price_data.copy()
        
        if not options_metrics:
            return df
        
        # IV Skew features (fear indicator)
        if 'skew_25d_put_iv' in options_metrics and 'skew_25d_call_iv' in options_metrics:
            put_iv = options_metrics['skew_25d_put_iv']
            call_iv = options_metrics['skew_25d_call_iv']
            
            # Skew ratio: >1 means puts expensive (fear)
            df['options_skew_ratio'] = put_iv / call_iv if call_iv > 0 else 1.0
            
            # Normalized skew
            df['options_skew_normalized'] = (put_iv - call_iv) / ((put_iv + call_iv) / 2) if (put_iv + call_iv) > 0 else 0
        
        # ATM IV (overall fear level)
        if 'skew_atm_put_iv' in options_metrics:
            df['options_atm_iv'] = options_metrics['skew_atm_put_iv']
        
        # Put/Call ratio features
        if 'pcr_volume' in options_metrics:
            pcr = options_metrics['pcr_volume']
            df['options_pcr_volume'] = pcr
            # PCR > 1 = bearish sentiment, < 1 = bullish
            df['options_pcr_signal'] = 1 if pcr < 0.7 else (-1 if pcr > 1.3 else 0)
        
        if 'pcr_open_interest' in options_metrics:
            df['options_pcr_oi'] = options_metrics['pcr_open_interest']
        
        # Gamma exposure (market maker positioning)
        if 'gamma_total_gamma' in options_metrics:
            df['options_gamma'] = options_metrics['gamma_total_gamma']
            # High gamma = likely to stay in range
            # Low gamma = potential for big move
        
        # Max pain (price magnet)
        if 'gamma_max_pain' in options_metrics:
            max_pain = options_metrics['gamma_max_pain']
            current_price = df['Close'].iloc[-1]
            df['options_max_pain'] = max_pain
            df['options_max_pain_distance'] = (current_price - max_pain) / current_price
        
        # Premium flow
        if 'premium_call_premium' in options_metrics and 'premium_put_premium' in options_metrics:
            call_prem = options_metrics['premium_call_premium']
            put_prem = options_metrics['premium_put_premium']
            total_prem = call_prem + put_prem
            
            if total_prem > 0:
                df['options_call_premium_ratio'] = call_prem / total_prem
                df['options_put_premium_ratio'] = put_prem / total_prem
                # Net premium flow: positive = bullish
                df['options_premium_flow'] = (call_prem - put_prem) / total_prem
        
        # Unusual activity
        if 'unusual_high_volume_strikes' in options_metrics:
            df['options_unusual_activity'] = len(options_metrics['unusual_high_volume_strikes'])
        
        return df
    
    def calculate_position_size(self,
                               capital: float,
                               entry_price: float,
                               stop_loss_price: float,
                               confidence: float) -> Dict:
        """
        Calculate position size based on risk management rules
        
        Kelly Criterion adjusted by confidence:
        Position = (Win% * AvgWin - Loss% * AvgLoss) / AvgWin
        
        Args:
            capital: Available capital
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            confidence: Model confidence (0.5-1.0)
            
        Returns:
            Dictionary with position details
        """
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_percent = risk_per_share / entry_price
        
        if risk_percent == 0:
            risk_percent = 0.02  # Default 2% stop
        
        # Adjust risk by confidence
        # Higher confidence = larger position (but capped)
        confidence_multiplier = (confidence - 0.5) * 4  # 0.5 conf = 0x, 0.75 conf = 1x
        confidence_multiplier = max(0.25, min(1.5, confidence_multiplier))
        
        # Calculate max shares based on risk
        risk_amount = capital * self.max_risk_per_trade * confidence_multiplier
        max_shares = int(risk_amount / risk_per_share)
        
        # Also limit by max position size (25% of capital)
        max_position_value = capital * 0.25
        max_shares_by_value = int(max_position_value / entry_price)
        
        shares = min(max_shares, max_shares_by_value)
        
        position_value = shares * entry_price
        position_risk = shares * risk_per_share
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_risk': position_risk,
            'risk_percent_of_capital': position_risk / capital,
            'confidence_multiplier': confidence_multiplier,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price
        }
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           atr: float,
                           direction: int,
                           method: str = 'atr') -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 1 for long, 0 for short
            method: 'atr', 'percent', or 'support'
            
        Returns:
            Stop loss price
        """
        if method == 'atr':
            # 2x ATR stop
            stop_distance = 2.0 * atr
        elif method == 'percent':
            # 2% stop
            stop_distance = entry_price * 0.02
        else:
            # Default 1.5x ATR
            stop_distance = 1.5 * atr
        
        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self,
                             entry_price: float,
                             stop_loss: float,
                             direction: int,
                             risk_reward: float = 2.0) -> float:
        """
        Calculate take profit price based on risk:reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 1 for long, 0 for short
            risk_reward: Target risk:reward ratio
            
        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward
        
        if direction == 1:  # Long
            take_profit = entry_price + reward
        else:  # Short
            take_profit = entry_price - reward
        
        return take_profit
    
    def generate_trade_signal(self,
                             prediction: int,
                             confidence: float,
                             current_price: float,
                             atr: float,
                             capital: float,
                             options_signal: Optional[int] = None) -> Dict:
        """
        Generate complete trade signal with all management parameters
        
        Args:
            prediction: Model prediction (1=up, 0=down)
            confidence: Model confidence
            current_price: Current price
            atr: Current ATR
            capital: Available capital
            options_signal: Options-based signal (-1, 0, 1)
            
        Returns:
            Complete trade signal dictionary
        """
        # Check if we should trade
        if confidence < self.confidence_threshold:
            return {
                'action': 'NO_TRADE',
                'reason': f'Confidence {confidence:.1%} below threshold {self.confidence_threshold:.1%}'
            }
        
        # Check options confirmation (if available)
        if options_signal is not None:
            if prediction == 1 and options_signal == -1:
                # Model says up, options say down
                confidence *= 0.8  # Reduce confidence
            elif prediction == 0 and options_signal == 1:
                # Model says down, options say up
                confidence *= 0.8
            elif (prediction == 1 and options_signal == 1) or (prediction == 0 and options_signal == -1):
                # Agreement - boost confidence
                confidence = min(confidence * 1.1, 0.95)
        
        # Recheck threshold after adjustment
        if confidence < self.confidence_threshold:
            return {
                'action': 'NO_TRADE',
                'reason': 'Confidence reduced by options disagreement'
            }
        
        # Calculate entry, stop, target
        direction = prediction
        entry_price = current_price
        stop_loss = self.calculate_stop_loss(entry_price, atr, direction)
        take_profit = self.calculate_take_profit(entry_price, stop_loss, direction, risk_reward=2.0)
        
        # Calculate position size
        position = self.calculate_position_size(capital, entry_price, stop_loss, confidence)
        
        # Generate signal
        signal = {
            'action': 'BUY' if direction == 1 else 'SELL',
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'shares': position['shares'],
            'position_value': position['position_value'],
            'position_risk': position['position_risk'],
            'risk_percent': position['risk_percent_of_capital'],
            'risk_reward_ratio': 2.0,
            'options_confirmed': options_signal == (1 if direction == 1 else -1) if options_signal else None
        }
        
        return signal
    
    def format_trade_plan(self, signal: Dict) -> str:
        """
        Format trade signal as readable trade plan
        
        Args:
            signal: Trade signal dictionary
            
        Returns:
            Formatted trade plan string
        """
        if signal['action'] == 'NO_TRADE':
            return f"❌ NO TRADE: {signal['reason']}"
        
        plan = f"""
╔══════════════════════════════════════════════════════════════╗
║                      TRADE PLAN                               ║
╠══════════════════════════════════════════════════════════════╣
║  Action:        {signal['action']} ({signal['direction']})
║  Confidence:    {signal['confidence']:.1%}
║  
║  ENTRY:         ${signal['entry_price']:.2f}
║  STOP LOSS:     ${signal['stop_loss']:.2f}
║  TAKE PROFIT:   ${signal['take_profit']:.2f}
║  
║  Position Size: {signal['shares']} shares
║  Position Value: ${signal['position_value']:,.0f}
║  Risk Amount:   ${signal['position_risk']:,.0f} ({signal['risk_percent']:.1%} of capital)
║  Risk:Reward:   1:{signal['risk_reward_ratio']:.1f}
║  
║  Options Confirmed: {'✓ Yes' if signal['options_confirmed'] else '✗ No' if signal['options_confirmed'] is False else 'N/A'}
╚══════════════════════════════════════════════════════════════╝
"""
        return plan


def create_trade_management_rules() -> Dict:
    """
    Return standard trade management rules
    """
    return {
        'entry_rules': [
            "Enter at market open if signal generated at close",
            "Use limit order at signal price ± 0.1%",
            "Wait for first 15 minutes to avoid opening volatility",
            "Don't enter if gap > 1% from signal price"
        ],
        'exit_rules': [
            "Exit at stop loss immediately (no exceptions)",
            "Exit at take profit (or trail stop)",
            "Exit at close if holding overnight not desired",
            "Exit if options signal reverses mid-day"
        ],
        'position_sizing': [
            "Risk maximum 2% of capital per trade",
            "Larger position for higher confidence (up to 1.5x)",
            "Smaller position for lower confidence (down to 0.25x)",
            "Never risk more than 6% total portfolio"
        ],
        'risk_management': [
            "Always use stop loss",
            "Move stop to breakeven after 1R profit",
            "Trail stop by 1 ATR after 2R profit",
            "Take partial profits at 2R (50% of position)"
        ]
    }


def main():
    """Test enhanced trading model"""
    
    print("=" * 70)
    print("ENHANCED TRADING MODEL TEST")
    print("=" * 70)
    
    # Initialize model
    model = EnhancedTradingModel(
        max_risk_per_trade=0.02,
        max_portfolio_risk=0.06,
        confidence_threshold=0.55
    )
    
    # Test signal generation
    print("\n📊 Test Signal Generation:")
    
    signal = model.generate_trade_signal(
        prediction=1,  # Model predicts UP
        confidence=0.62,  # 62% confidence
        current_price=691.66,  # Current SPY price
        atr=8.50,  # ATR
        capital=100000,  # $100k capital
        options_signal=1  # Options also bullish
    )
    
    print(model.format_trade_plan(signal))
    
    # Test with low confidence
    print("\n📊 Test Low Confidence Signal:")
    
    signal_low = model.generate_trade_signal(
        prediction=1,
        confidence=0.52,  # Below threshold
        current_price=691.66,
        atr=8.50,
        capital=100000,
        options_signal=0
    )
    
    print(model.format_trade_plan(signal_low))
    
    # Print trade management rules
    print("\n📋 TRADE MANAGEMENT RULES:")
    rules = create_trade_management_rules()
    
    for category, rule_list in rules.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for rule in rule_list:
            print(f"  • {rule}")
    
    print("\n" + "=" * 70)
    print("ENHANCED MODEL READY ✓")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    model = main()
