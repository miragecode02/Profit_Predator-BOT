import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.test import GOOG

# --- Custom Pivot Functions ---
# pandas_ta doesn't include pivothigh/pivotlow, so we implement them manually

def pivothigh(series, left, right):
    """
    Identifies pivot highs in a price series (similar to TradingView's pivothigh).
    A pivot high occurs when the center bar's high is the highest in a window.
    
    Args:
        series: Price series (typically 'high' column)
        left: Number of bars to the left
        right: Number of bars to the right
    
    Returns: Series with pivot high values at their positions, NaN elsewhere.
    """
    pivots = pd.Series(index=series.index, dtype=float)
    
    for i in range(left, len(series) - right):
        window = series.iloc[i - left:i + right + 1]
        center_value = series.iloc[i]
        
        # Check if center is the highest in the window
        if center_value == window.max() and (window == center_value).sum() == 1:
            pivots.iloc[i] = center_value
    
    return pivots

def pivotlow(series, left, right):
    """
    Identifies pivot lows in a price series (similar to TradingView's pivotlow).
    A pivot low occurs when the center bar's low is the lowest in a window.
    
    Args:
        series: Price series (typically 'low' column)
        left: Number of bars to the left
        right: Number of bars to the right
    
    Returns: Series with pivot low values at their positions, NaN elsewhere.
    """
    pivots = pd.Series(index=series.index, dtype=float)
    
    for i in range(left, len(series) - right):
        window = series.iloc[i - left:i + right + 1]
        center_value = series.iloc[i]
        
        # Check if center is the lowest in the window
        if center_value == window.min() and (window == center_value).sum() == 1:
            pivots.iloc[i] = center_value
    
    return pivots

# --- Part 1: Data Preparation for Multi-Timeframe (MTF) Analysis ---

# Load your primary, lower-timeframe (LTF) data
data = GOOG.copy() 
# EXAMPLE: data = pd.read_csv('BTCUSD_3Min_Data.csv', index_col=0, parse_dates=True)
data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

# --- Helper function to calculate HTF bias ---
def htf_bias_indicator(df, pivots):
    """Calculates HTF pivots and determines bias."""
    pivothigh_series = pivothigh(df.high, left=pivots, right=pivots)
    pivotlow_series = pivotlow(df.low, left=pivots, right=pivots)
    
    last_high = pivothigh_series.dropna().iloc[-1] if not pivothigh_series.dropna().empty else float('nan')
    last_low = pivotlow_series.dropna().iloc[-1] if not pivotlow_series.dropna().empty else float('nan')
    
    if df.close.iloc[-1] > last_high:
        return 1  # Bullish
    if df.close.iloc[-1] < last_low:
        return -1  # Bearish
    return 0  # Neutral

# Pre-calculate all indicators
print("Preparing data and calculating indicators...")

# Calculate LTF pivots directly on the main DataFrame
data['ltf_pivothigh'] = pivothigh(data.high, left=3, right=3)
data['ltf_pivotlow'] = pivotlow(data.low, left=3, right=3)

# Resample LTF candles into 15-minute candles
htf = data.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Calculate HTF pivots
htf['pivothigh'] = pivothigh(htf['high'], left=10, right=10)
htf['pivotlow'] = pivotlow(htf['low'], left=10, right=10)

# Store latest confirmed pivot levels
htf['last_high'] = htf['pivothigh'].ffill()
htf['last_low'] = htf['pivotlow'].ffill()

# Determine HTF bias
htf['HTF_Bias'] = 0

htf.loc[
    htf['close'] > htf['last_high'],
    'HTF_Bias'
] = 1

htf.loc[
    htf['close'] < htf['last_low'],
    'HTF_Bias'
] = -1

# Map HTF bias back onto LTF candles
data['HTF_Bias'] = htf['HTF_Bias'].reindex(
    data.index,
    method='ffill'
)

data['HTF_Bias'] = data['HTF_Bias'].fillna(0)
print("Data preparation complete.")

# --- Part 2: The Backtesting Strategy Class ---

class SmcStrategy(Strategy):
    g_rr = 3.0
    
    def init(self):
        # Create references to our pre-calculated indicator columns
        self.htf_bias = self.data.HTF_Bias
        self.ltf_pivothigh = self.data.ltf_pivothigh
        self.ltf_pivotlow = self.data.ltf_pivotlow
        
        # State variables to track strategy progress
        self.poi_top = None
        self.poi_bottom = None
        self.potential_sl = None
        self.mss_confirmed = False

    def is_bullish_hammer(self, index=-1):
        """Checks if the candle at the given index is a bullish hammer."""
        candle = self.data.df.iloc[index]
        body_size = abs(candle['close'] - candle['open'])
        if body_size == 0: 
            return False
        
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        return lower_wick >= body_size * 1.0 and upper_wick <= body_size * 0.5

    def next(self):
        # This method is called for each candle in the historical data
        
        # --- Stage 1: Wait for a Market Structure Shift (MSS) ---
        if self.htf_bias[-1] == 1 and not self.trades and not self.mss_confirmed:
            # Check if a recent pivot high exists and the price has broken above it
            if not pd.isna(self.ltf_pivothigh[-2]) and self.data.close[-1] > self.ltf_pivothigh[-2]:
                self.mss_confirmed = True
                
                # The leg starts from the last pivot low before the breakout pivot high
                self.potential_sl = self.data.low[-2]  # Simplified SL point
                
                # Simplified POI is the range of the breakout candle's body
                self.poi_top = max(self.data.open[-1], self.data.close[-1])
                self.poi_bottom = min(self.data.open[-1], self.data.close[-1])
                
        # --- Stage 2: Wait for Price to Enter the POI ---
        if self.mss_confirmed and not self.trades:
            # Check if the current low has entered our Point of Interest
            if self.data.low[-1] <= self.poi_top:
                
                # --- Stage 3: Wait for Entry Confirmation (Hammer) ---
                if self.is_bullish_hammer():
                    # Place the trade
                    self.buy(
                        sl=self.potential_sl,
                        tp=self.data.close[-1] + (self.data.close[-1] - self.potential_sl) * self.g_rr
                    )
                    # Reset state after placing trade
                    self.mss_confirmed = False
                    self.poi_top = None
                    self.poi_bottom = None
            
            # Invalidate POI if price makes a new high without entering
            if self.data.high[-1] > self.data.high[-2]:
                self.mss_confirmed = False
                self.poi_top = None
                self.poi_bottom = None

# --- Part 3: Running the Backtest ---
# backtesting.py requires these exact column names
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
# Configure the backtest
bt = Backtest(
    data, 
    SmcStrategy, 
    cash=10000, 
    commission=.002  # 0.2% commission for trades
)

print("Running backtest...")
stats = bt.run()
print("Backtest finished.")

print(stats)
bt.plot()