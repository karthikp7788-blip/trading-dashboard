"""
Trading Prediction System - Streamlit Dashboard
Phase 0.1 Demo Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import our modules
from technical_indicators import TechnicalIndicators
from options_analytics import OptionsAnalytics
from real_data_provider import RealDataProvider

# Page configuration
st.set_page_config(
    page_title="Trading Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bullish {
        color: #00ff00;
        font-weight: bold;
    }
    .bearish {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data_provider():
    """Initialize real data provider once"""
    return RealDataProvider(massive_api_key="xxu7bdpXlzs9EwWXFk_x0BDkM25FSBFg")


@st.cache_data
def load_data(ticker):
    """Load and process data"""
    provider = get_data_provider()
    
    # Get real market data (5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1260)).strftime('%Y-%m-%d')
    
    price_data = provider.get_stock_data(ticker, start_date, end_date)
    
    # Calculate indicators
    data_with_indicators = TechnicalIndicators.calculate_all_indicators(price_data)
    
    # Get real options chain
    options_data = provider.get_options_chain(ticker)
    
    # Get current price
    current_price = data_with_indicators['Close'].iloc[-1]
    
    # Calculate options metrics
    options_metrics = OptionsAnalytics.calculate_all_metrics(options_data, current_price)
    
    return data_with_indicators, options_data, options_metrics


def plot_price_chart(data, days=180):
    """Create interactive price chart with indicators"""
    recent_data = data.tail(days)
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=recent_data.index,
        open=recent_data['Open'],
        high=recent_data['High'],
        low=recent_data['Low'],
        close=recent_data['Close'],
        name='Price'
    ))
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['SMA_20'],
        name='SMA 20',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['SMA_50'],
        name='SMA 50',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['SMA_200'],
        name='SMA 200',
        line=dict(color='red', width=1)
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['BB_Upper'],
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['BB_Lower'],
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ))
    
    fig.update_layout(
        title='Price Chart with Technical Indicators',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_rsi(data, days=180):
    """Plot RSI indicator"""
    recent_data = data.tail(days)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # Overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        xaxis_title='Date',
        height=300,
        template='plotly_white'
    )
    
    return fig


def plot_macd(data, days=180):
    """Plot MACD indicator"""
    recent_data = data.tail(days)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['MACD'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['MACD_Signal'],
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    # Histogram
    colors = ['green' if val >= 0 else 'red' for val in recent_data['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=recent_data.index,
        y=recent_data['MACD_Hist'],
        name='Histogram',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        yaxis_title='MACD',
        xaxis_title='Date',
        height=300,
        template='plotly_white'
    )
    
    return fig


def plot_volume(data, days=180):
    """Plot volume"""
    recent_data = data.tail(days)
    
    fig = go.Figure()
    
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(recent_data['Close'], recent_data['Open'])]
    
    fig.add_trace(go.Bar(
        x=recent_data.index,
        y=recent_data['Volume'],
        name='Volume',
        marker_color=colors
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Volume_SMA'],
        name='Volume SMA',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Volume Analysis',
        yaxis_title='Volume',
        xaxis_title='Date',
        height=300,
        template='plotly_white'
    )
    
    return fig


def plot_iv_skew(options_data, spot_price):
    """Plot IV skew smile"""
    calls = options_data['calls']
    puts = options_data['puts']
    
    # Filter for valid IV
    calls_valid = calls[calls['impliedVolatility'] > 0].copy()
    puts_valid = puts[puts['impliedVolatility'] > 0].copy()
    
    calls_valid['moneyness'] = calls_valid['strike'] / spot_price
    puts_valid['moneyness'] = puts_valid['strike'] / spot_price
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=calls_valid['moneyness'],
        y=calls_valid['impliedVolatility'] * 100,
        mode='markers+lines',
        name='Calls',
        marker=dict(size=8, color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=puts_valid['moneyness'],
        y=puts_valid['impliedVolatility'] * 100,
        mode='markers+lines',
        name='Puts',
        marker=dict(size=8, color='red')
    ))
    
    # ATM line
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="ATM")
    
    fig.update_layout(
        title='Implied Volatility Skew',
        xaxis_title='Moneyness (Strike / Spot)',
        yaxis_title='Implied Volatility (%)',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_oi_distribution(options_data, spot_price):
    """Plot open interest distribution"""
    calls = options_data['calls']
    puts = options_data['puts']
    
    # Combine and group by strike
    all_options = pd.concat([
        calls[['strike', 'openInterest']].assign(type='Calls'),
        puts[['strike', 'openInterest']].assign(type='Puts')
    ])
    
    oi_by_strike = all_options.groupby(['strike', 'type'])['openInterest'].sum().reset_index()
    
    fig = go.Figure()
    
    # Calls
    calls_oi = oi_by_strike[oi_by_strike['type'] == 'Calls']
    fig.add_trace(go.Bar(
        x=calls_oi['strike'],
        y=calls_oi['openInterest'],
        name='Calls',
        marker_color='green',
        opacity=0.7
    ))
    
    # Puts
    puts_oi = oi_by_strike[oi_by_strike['type'] == 'Puts']
    fig.add_trace(go.Bar(
        x=puts_oi['strike'],
        y=-puts_oi['openInterest'],
        name='Puts',
        marker_color='red',
        opacity=0.7
    ))
    
    # Spot price line
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text=f"Spot: ${spot_price:.2f}")
    
    fig.update_layout(
        title='Open Interest Distribution',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=400,
        template='plotly_white',
        barmode='overlay'
    )
    
    return fig


def main():
    """Main application"""
    
    # Sidebar
    st.sidebar.title("📊 Trading System")
    st.sidebar.markdown("---")
    
    # Ticker selection
    ticker = st.sidebar.selectbox(
        "Select Ticker",
        ['SPY', 'TSLA', 'GOOGL', 'NVDA', 'AMD', 'META']
    )
    
    # Timeframe
    timeframe = st.sidebar.selectbox(
        "Chart Timeframe",
        ['6 Months', '1 Year', '2 Years', 'All'],
        index=1
    )
    
    timeframe_map = {
        '6 Months': 126,
        '1 Year': 252,
        '2 Years': 504,
        'All': 9999
    }
    days = timeframe_map[timeframe]
    
    st.sidebar.markdown("---")
    st.sidebar.success("✓ **Live Data**: Connected to real market data")
    
    # Load data
    with st.spinner('Loading data...'):
        data, options, metrics = load_data(ticker)
    
    current_price = data['Close'].iloc[-1]
    latest = data.iloc[-1]
    
    # Main header
    st.markdown(f'<div class="main-header">📈 {ticker} Trading Analysis</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "🎯 Options Intelligence", "📋 Data"])
    
    with tab1:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            daily_change = ((current_price / data['Close'].iloc[-2]) - 1) * 100
            st.metric("Daily Change", f"{daily_change:+.2f}%", delta=f"{daily_change:+.2f}%")
        
        with col3:
            st.metric("Volume", f"{int(latest['Volume']):,}")
        
        with col4:
            st.metric("ATR", f"${latest['ATR']:.2f}")
        
        st.markdown("---")
        
        # Trend Analysis
        st.subheader("🎯 Current Market Signal")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Determine signal
            rsi = latest['RSI']
            macd_hist = latest['MACD_Hist']
            ma_alignment = latest['MA_Alignment']
            
            signal = "NEUTRAL"
            signal_color = "neutral"
            
            if rsi > 70 and macd_hist < 0:
                signal = "BEARISH"
                signal_color = "bearish"
            elif rsi < 30 and macd_hist > 0:
                signal = "BULLISH"
                signal_color = "bullish"
            elif ma_alignment == 2 and macd_hist > 0:
                signal = "BULLISH"
                signal_color = "bullish"
            elif ma_alignment == 0 and macd_hist < 0:
                signal = "BEARISH"
                signal_color = "bearish"
            
            st.markdown(f"### Overall Signal: <span class='{signal_color}'>{signal}</span>", unsafe_allow_html=True)
            
            st.write("**Signal Drivers:**")
            st.write(f"- RSI: {rsi:.2f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}")
            st.write(f"- MACD Histogram: {macd_hist:.4f} {'(Bullish)' if macd_hist > 0 else '(Bearish)'}")
            st.write(f"- MA Alignment: {int(ma_alignment)}/2 {'(Strong Trend)' if ma_alignment == 2 else '(Weak Trend)' if ma_alignment == 0 else '(Mixed)'}")
            st.write(f"- Price vs SMA 20: {((current_price/latest['SMA_20']-1)*100):+.2f}%")
            st.write(f"- Price vs SMA 200: {((current_price/latest['SMA_200']-1)*100):+.2f}%")
        
        with col2:
            st.metric("RSI", f"{rsi:.2f}")
            st.metric("MACD Hist", f"{macd_hist:.4f}")
            st.metric("Volatility (20d)", f"{latest['Volatility_20d']*100:.2f}%")
            st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
        
        st.markdown("---")
        
        # Price chart
        st.plotly_chart(plot_price_chart(data, days), use_container_width=True)
        
        # Volume chart
        st.plotly_chart(plot_volume(data, days), use_container_width=True)
    
    with tab2:
        st.subheader("📈 Technical Indicators")
        
        # Moving Averages
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SMA 20", f"${latest['SMA_20']:.2f}", 
                     delta=f"{((current_price/latest['SMA_20']-1)*100):+.2f}%")
        with col2:
            st.metric("SMA 50", f"${latest['SMA_50']:.2f}",
                     delta=f"{((current_price/latest['SMA_50']-1)*100):+.2f}%")
        with col3:
            st.metric("SMA 200", f"${latest['SMA_200']:.2f}",
                     delta=f"{((current_price/latest['SMA_200']-1)*100):+.2f}%")
        
        st.markdown("---")
        
        # RSI
        st.plotly_chart(plot_rsi(data, days), use_container_width=True)
        
        # MACD
        st.plotly_chart(plot_macd(data, days), use_container_width=True)
        
        # Bollinger Bands info
        st.subheader("📏 Bollinger Bands")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Upper Band", f"${latest['BB_Upper']:.2f}")
        with col2:
            st.metric("Middle Band", f"${latest['BB_Middle']:.2f}")
        with col3:
            st.metric("Lower Band", f"${latest['BB_Lower']:.2f}")
        
        bb_position = latest['BB_Position']
        st.progress(bb_position)
        st.caption(f"Price position within bands: {bb_position*100:.1f}%")
    
    with tab3:
        st.subheader("🎯 Options Intelligence")
        
        if not metrics:
            st.warning("No options metrics available")
        else:
            # Skew metrics
            st.markdown("### IV Skew Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            if 'skew_atm_call_iv' in metrics:
                with col1:
                    st.metric("ATM Call IV", f"{metrics['skew_atm_call_iv']*100:.2f}%")
            
            if 'skew_atm_put_iv' in metrics:
                with col2:
                    st.metric("ATM Put IV", f"{metrics['skew_atm_put_iv']*100:.2f}%")
            
            if 'skew_pc_iv_skew' in metrics:
                with col3:
                    skew_value = metrics['skew_pc_iv_skew']*100
                    st.metric("Put-Call Skew", f"{skew_value:+.2f}%")
                    if skew_value > 0:
                        st.caption("🛡️ Defensive positioning")
                    else:
                        st.caption("😌 Complacent positioning")
            
            # IV Skew chart
            st.plotly_chart(plot_iv_skew(options, current_price), use_container_width=True)
            
            st.markdown("---")
            
            # Premium metrics
            st.markdown("### Premium Flow Analysis")
            
            col1, col2 = st.columns(2)
            
            if 'premium_call_put_premium_ratio' in metrics:
                with col1:
                    ratio = metrics['premium_call_put_premium_ratio']
                    st.metric("Call/Put Premium Ratio", f"{ratio:.2f}")
            
            if 'premium_premium_imbalance' in metrics:
                with col2:
                    imbalance = metrics['premium_premium_imbalance']
                    st.metric("Premium Imbalance", f"{imbalance:+.2f}")
                    
                    if imbalance > 0.2:
                        st.caption("📈 Strong bullish bias (heavy call buying)")
                    elif imbalance < -0.2:
                        st.caption("📉 Strong bearish bias (heavy put buying)")
                    else:
                        st.caption("⚖️ Neutral positioning")
            
            st.markdown("---")
            
            # Open Interest
            st.markdown("### Open Interest Analysis")
            
            if 'oi_max_pain_strike' in metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_pain = metrics['oi_max_pain_strike']
                    distance = metrics.get('oi_max_pain_distance', 0) * 100
                    st.metric("Max Pain Strike", f"${max_pain:.2f}")
                    st.caption(f"{distance:+.2f}% from spot")
                
                with col2:
                    if 'oi_total_oi' in metrics:
                        st.metric("Total OI", f"{int(metrics['oi_total_oi']):,}")
                
                with col3:
                    if 'oi_nearby_oi_concentration' in metrics:
                        concentration = metrics['oi_nearby_oi_concentration']*100
                        st.metric("OI near spot (±5%)", f"{concentration:.1f}%")
                        if concentration > 60:
                            st.caption("🎯 High gamma - range-bound risk")
            
            # OI Distribution chart
            st.plotly_chart(plot_oi_distribution(options, current_price), use_container_width=True)
            
            # Top strikes
            if 'oi_top_5_strikes' in metrics and 'oi_top_5_oi' in metrics:
                st.markdown("### Top Strikes by Open Interest")
                
                top_strikes_df = pd.DataFrame({
                    'Strike': metrics['oi_top_5_strikes'][:5],
                    'Open Interest': [int(x) for x in metrics['oi_top_5_oi'][:5]]
                })
                
                top_strikes_df['Distance from Spot'] = top_strikes_df['Strike'].apply(
                    lambda x: f"{((x/current_price-1)*100):+.2f}%"
                )
                
                st.dataframe(top_strikes_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("📋 Raw Data")
        
        # Recent price data
        st.markdown("### Recent Price Data (Last 20 Days)")
        recent = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'ATR']].tail(20)
        st.dataframe(recent, use_container_width=True)
        
        st.markdown("---")
        
        # Options data
        st.markdown("### Current Options Chain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Call Options (Top 10 by Volume)**")
            calls_display = options['calls'].nlargest(10, 'volume')[
                ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
            ].copy()
            calls_display['impliedVolatility'] = calls_display['impliedVolatility'] * 100
            st.dataframe(calls_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Put Options (Top 10 by Volume)**")
            puts_display = options['puts'].nlargest(10, 'volume')[
                ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
            ].copy()
            puts_display['impliedVolatility'] = puts_display['impliedVolatility'] * 100
            st.dataframe(puts_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # All metrics
        st.markdown("### All Calculated Metrics")
        
        if metrics:
            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in sorted(metrics.items())
                if not isinstance(v, list)
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
