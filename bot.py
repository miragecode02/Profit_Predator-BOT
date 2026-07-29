# --- Imports ---
import pandas as pd
import pandas_ta as ta
import alpaca_trade_api as tradeapi
import schedule
import time
import pytz
from datetime import datetime

# --- Configuration ---
# Replace with your actual Alpaca API keys
API_KEY = 'YOUR_API_KEY_ID'
API_SECRET = 'YOUR_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets' # This is the paper trading URL

# Strategy Parameters (from Pine Script)
SYMBOL = 'BTC/USD' # Alpaca uses this format for crypto
LTF_TIMEFRAME = '3Min' # Lower Timeframe
HTF_TIMEFRAME = '15Min' # Higher Timeframe
LTF_PIVOTS = 3
HTF_PIVOTS = 10
MAX_SL_PIPS = 17.0
RR_RATIO = 3.0
TRADE_QUANTITY = 0.01 # Quantity of the asset to trade

# Session Parameters
SESSION_TIMEZONE = 'Asia/Kolkata'
SESSION_START_HOUR = 12
SESSION_START_MIN = 30

# --- API Connection ---
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)

# --- Helper Functions ---
def is_in_session():
    """Checks if the current time is within the trading session."""
    tz = pytz.timezone(SESSION_TIMEZONE)
    now = datetime.now(tz)
    
    # Let's check the current time in your timezone
    print(f"Current time in {SESSION_TIMEZONE} is {now.strftime('%H:%M:%S')}")
    
    start_time = now.replace(hour=SESSION_START_HOUR, minute=SESSION_START_MIN, second=0, microsecond=0).time()
    
    # Since the end time is 23:59, we just need to check if we are after the start time.
    return now.time() >= start_time

def is_bullish_hammer(candle):
    """Checks if a candle (pandas Series) is a bullish hammer."""
    body_size = abs(candle['close'] - candle['open'])
    if body_size == 0: return False
    
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    
    # Pine Script logic translated to Python
    return lower_wick >= body_size * 1.0 and upper_wick <= body_size * 0.5

# --- Main Strategy Logic ---
def run_strategy():
    """The main function to fetch data and run the SMC logic."""
    print("\nRunning strategy check...")
    
    if not is_in_session():
        print("Outside of trading session. Waiting...")
        return

    try:
        # 1. Fetch Data
        htf_bars = api.get_crypto_bars(SYMBOL, HTF_TIMEFRAME, limit=200).df
        ltf_bars = api.get_crypto_bars(SYMBOL, LTF_TIMEFRAME, limit=200).df
        
        # 2. Calculate Indicators
        htf_bars['pivothigh'] = ta.pivothigh(htf_bars['high'], left=HTF_PIVOTS, right=HTF_PIVOTS)
        htf_bars['pivotlow'] = ta.pivotlow(htf_bars['low'], left=HTF_PIVOTS, right=HTF_PIVOTS)
        ltf_bars['pivothigh'] = ta.pivothigh(ltf_bars['high'], left=LTF_PIVOTS, right=LTF_PIVOTS)
        ltf_bars['pivotlow'] = ta.pivotlow(ltf_bars['low'], left=LTF_PIVOTS, right=LTF_PIVOTS)

        # 3. Determine HTF Bias
        htf_last_high = htf_bars['pivothigh'].dropna().iloc[-1]
        htf_last_low = htf_bars['pivotlow'].dropna().iloc[-1]
        latest_htf_close = htf_bars['close'].iloc[-1]
        htf_bias = 1 if latest_htf_close > htf_last_high else -1 if latest_htf_close < htf_last_low else 0
        
        print(f"HTF Bias: {'Bullish' if htf_bias == 1 else 'Bearish' if htf_bias == -1 else 'Neutral'}")

        # 4. Execute LTF Logic only if bias is Bullish
        if htf_bias == 1:
            ltf_last_high_price = ltf_bars['pivothigh'].dropna().iloc[-1]
            ltf_last_low_row = ltf_bars[ltf_bars['pivotlow'].notna()].iloc[-1]
            latest_ltf_close = ltf_bars['close'].iloc[-1]

            # Check for Market Structure Shift (MSS)
            if latest_ltf_close > ltf_last_high_price:
                print("Bullish MSS detected.")
                # This is a simplified POI detection (using the last swing low area)
                # A full FVG search would require a more complex loop here.
                poi_top = ltf_last_high_price
                poi_bottom = ltf_last_low_row['pivotlow']
                
                print(f"POI identified between {poi_bottom} and {poi_top}")

                # Check for entry
                latest_ltf_candle = ltf_bars.iloc[-1]
                if latest_ltf_candle['low'] <= poi_top:
                    if is_bullish_hammer(latest_ltf_candle):
                        print("Hammer candle detected in POI. Preparing to enter trade.")
                        
                        # 5. Calculate SL/TP and Place Trade
                        stop_loss = ltf_last_low_row['pivotlow']
                        take_profit = latest_ltf_close + (latest_ltf_close - stop_loss) * RR_RATIO
                        
                        # Check if a trade is already open
                        positions = api.list_positions()
                        if not any(p.symbol == SYMBOL.replace('/', '') for p in positions):
                            print(f"Placing BUY order for {SYMBOL} at {latest_ltf_close}. SL: {stop_loss}, TP: {take_profit}")
                            api.submit_order(
                                symbol=SYMBOL.replace('/', ''),
                                qty=TRADE_QUANTITY,
                                side='buy',
                                type='market',
                                time_in_force='gtc',
                                order_class='bracket',
                                stop_loss={'stop_price': stop_loss},
                                take_profit={'limit_price': take_profit}
                            )
                        else:
                            print("Position already open for this symbol. Skipping new entry.")
                            
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Scheduling ---
# Schedule the job to run every 3 minutes.
schedule.every(3).minutes.do(run_strategy)

# Run the scheduler
print("Bot started. Waiting for next scheduled run...")
run_strategy() # Run once immediately at the start
while True:
    schedule.run_pending()
    time.sleep(1)
    