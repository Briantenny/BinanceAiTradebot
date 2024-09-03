import ccxt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import ta
import asyncio
import time
import hashlib
import urllib.parse
import hmac
from telegram import Bot
from telegram.ext import Application, CommandHandler
from datetime import datetime, timedelta

print("Starting the trading bot script...")

# Binance API setup
print("Setting up Binance API connection...")
exchange = ccxt.binance({
    'apiKey': 'YOUR BINANCE APIKEY',
    'secret': 'YOUR BINANCE SECRET KEY',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})
print("Binance API connection established.")

# Telegram bot setup
print("Setting up Telegram bot...")
BOT_TOKEN = 'TELEGRAM BOT TOKEN'
bot = Bot(token=BOT_TOKEN)
chat_id = 'YOUR CHAT ID'
print("Telegram bot setup complete.")

#def create_signed_request(params):
#  query_string = urllib.parse.urlencode(params)
#  signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
#  params['signature'] = signature

#  return params

# Trading parameters
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 1000
timestamp = int(time.time() * 1000)
recv_window = 5000
params = {
  'timestamp': timestamp,
  'recvWindow': recv_window,
  # other parameters
}
#signed_params = create_signed_request(params)
#print(f"Trading parameters set: Symbol={symbol}, Timeframe={timeframe}, Limit={limit}")

# Test mode parameters
TEST_MODE = True
initial_balance = 10000  # Initial balance in USDT
trade_amount = 0.001  # Amount of BTC to trade
print(f"Test mode: {'Enabled' if TEST_MODE else 'Disabled'}")
print(f"Initial balance: {initial_balance} USDT, Trade amount: {trade_amount} BTC")

class TestExchange:
    def __init__(self, balance):
        self.balance = balance
        self.position = 0
        self.entry_price = 0
        print(f"TestExchange initialized with balance: {balance} USDT")

    def create_market_buy_order(self, symbol, amount):
        price = self.get_current_price(symbol)
        cost = price * amount
        if cost <= self.balance:
            self.balance -= cost
            self.position += amount
            self.entry_price = price
            print(f"Test buy order executed: Amount={amount}, Cost={cost}, Price={price}")
            return {"amount": amount, "cost": cost, "price": price}
        else:
            print(f"Insufficient funds for buy order. Required: {cost}, Available: {self.balance}")
            raise Exception("Insufficient funds")

    def create_market_sell_order(self, symbol, amount):
        if self.position >= amount:
            price = self.get_current_price(symbol)
            revenue = price * amount
            self.balance += revenue
            self.position -= amount
            profit = (price - self.entry_price) * amount
            print(f"Test sell order executed: Amount={amount}, Revenue={revenue}, Price={price}, Profit={profit}")
            return {"amount": amount, "revenue": revenue, "price": price, "profit": profit}
        else:
            print(f"Insufficient position for sell order. Required: {amount}, Available: {self.position}")
            raise Exception("Insufficient position")

    def get_current_price(self, symbol, price_column='close'):
    	data = get_data()
    	if price_column in data.columns:
    		price = data[price_column].iloc[-1]
    		print(f"Current {price_column} for {symbol}: {price}")
    		return price
    	else:
    		raise DataError(f"Column '{price_column}' not found in the data")

    def get_balance(self):
        current_price = self.get_current_price(symbol)
        free_balance = self.balance
        used_balance = self.position * current_price
        print(f"Current balance: Free={free_balance} USDT, Used={used_balance} USDT")
        return {"free": free_balance, "used": used_balance}

test_exchange = TestExchange(initial_balance)

def get_data():
    print(f"Fetching data for {symbol} with timeframe {timeframe}...")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        print(f"Raw data fetched. Type: {type(ohlcv)}, Length: {len(ohlcv)}")
        print(f"Sample of raw data: {ohlcv[:2]}")
        
        if not ohlcv:
            raise ValueError("No data received from the exchange")
        
        # Create DataFrame without column names first
        df = pd.DataFrame(ohlcv)
        print(f"DataFrame created without column names. Shape: {df.shape}")
        print(f"DataFrame info:\n{df.info()}")
        print(f"DataFrame head:\n{df.head()}")
        
        # Manually assign column names
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        print(f"Column names assigned: {df.columns}")
        
        # Convert timestamp and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Final DataFrame shape: {df.shape}")
        print(f"Final DataFrame columns: {df.columns}")
        print(f"Final DataFrame head:\n{df.head()}")
        print(f"Final DataFrame info:\n{df.info()}")
        
        return df
    except Exception as e:
        print(f"Error in get_data(): {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise
def add_indicators(df):
    print("Adding technical indicators...")
    print("DataFrame before adding indicators:")
    print(df.info())
    print(df.head())
    
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    df['MA_fast'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
    df['MA_slow'] = ta.trend.SMAIndicator(df['close'], window=30).sma_indicator()
    
    print("Indicators added successfully.")
    print("DataFrame after adding indicators:")
    print(df.info())
    print(df.head())
    return df

features = ['RSI', 'MACD', 'MA_fast', 'MA_slow', 'close']

def prepare_data(df):
    print("Preparing data for model training...")
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)

    features = ['RSI', 'MACD', 'MA_fast', 'MA_slow', 'close']  # Ensure 'close' is included
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data prepared. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def predict_signal(model, latest_features):
    signal = model.predict(latest_features.reshape(1, -1))[0]
    print(f"Prediction: {'Buy' if signal == 1 else 'Sell'}")
    return signal

def execute_trade(signal):
    try:
        if TEST_MODE:
            current_exchange = test_exchange
        else:
            current_exchange = exchange

        if signal == 1:  # Buy signal
            print("Executing buy order...")
            order = current_exchange.create_market_buy_order(symbol, amount=trade_amount)
            send_notification(f"Buy order executed: {order}")
        elif signal == 0:  # Sell signal
            print("Executing sell order...")
            order = current_exchange.create_market_sell_order(symbol, amount=trade_amount)
            send_notification(f"Sell order executed: {order}")
    except Exception as e:
        error_msg = f"Error executing trade: {str(e)}"
        print(error_msg)
        send_notification(error_msg)

async def send_notification(message):
    print(f"Sending Telegram notification: {message}")
    await bot.send_message(chat_id=chat_id, text=message)

async def start_bot(update, context):
    message = "Trading bot started!"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)
    print(message)

def test_strategy(model, test_data):
    print("Testing strategy on historical data...")

    # Ensure 'close' is in test_data
    if 'close' not in test_data.columns:
        raise ValueError("test_data DataFrame does not contain 'close' column")

    signals = model.predict(test_data[features])  # Use the same features as during training
    test_data['Signal'] = signals
    test_data['Returns'] = test_data['close'].pct_change()
    test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']
    cumulative_returns = (1 + test_data['Strategy_Returns']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * test_data['Strategy_Returns'].mean() / test_data['Strategy_Returns'].std()

    return total_return, sharpe_ratio

async def main():
    
    print("Starting main function...")
    try:
        # Initial setup
        df = get_data()
        if df.empty:
            raise ValueError("No data received from the exchange")
        
        print("Data fetched successfully. Adding indicators...")
        try:
            df = add_indicators(df)
        except KeyError as e:
            print(f"KeyError occurred while adding indicators: {str(e)}")
            print("DataFrame columns:", df.columns)
            print("DataFrame head:")
            print(df.head())
            raise
        
        print("Indicators added. Preparing data for model...")
        X_train, X_test, y_train, y_test = prepare_data(df)
        print("Data prepared. Training model...")
        model = train_model(X_train, y_train)
        # Test the strategy
        test_return, test_sharpe = test_strategy(model, pd.concat([X_test, y_test], axis=1))
        await send_notification(f"Strategy backtest results: Return: {test_return:.2%}, Sharpe Ratio: {test_sharpe:.2f}")

        # Telegram bot setup
        print("Setting up Telegram bot commands...")
        application = Application.builder().token(BOT_TOKEN).build()
        application.add_handler(CommandHandler('start', start_bot))
        await application.initialize()
        await application.start()
        print("Telegram bot is now listening for commands.")

        await send_notification(f"Trading bot initialized in {'TEST' if TEST_MODE else 'LIVE'} mode.")

        start_time = datetime.now()
        print(f"Starting trading loop at {start_time}")
        while (datetime.now() - start_time) < timedelta(hours=24):  # Run for 24 hours in test mode
            try:
                print("Fetching latest data...")
                latest_data = get_data().iloc[-1]
                latest_features = np.array([latest_data['RSI'], latest_data['MACD'], 
                                            latest_data['MA_fast'], latest_data['MA_slow']])
                
                print("Making prediction...")
                signal = predict_signal(model, latest_features)
                
                print("Executing trade based on prediction...")
                execute_trade(signal)
                
                if TEST_MODE:
                    balance = test_exchange.get_balance()
                    await send_notification(f"Current balance: {balance['free']} USDT, Position: {balance['used']} USDT")
                
                print("Waiting for next iteration...")
                await asyncio.sleep(3600)  # Wait for 1 hour
            
            except Exception as e:
                error_msg = f"Error in main loop: {str(e)}"
                print(error_msg)
                await send_notification(error_msg)
                await asyncio.sleep(60)  # Wait for 1 minute before retrying

        if TEST_MODE:
            final_balance = test_exchange.get_balance()
            total_value = final_balance['free'] + final_balance['used']
            profit = total_value - initial_balance
            result_msg = f"Test completed. Final balance: {total_value} USDT, Profit: {profit} USDT ({profit/initial_balance:.2%})"
            print(result_msg)
            await send_notification(result_msg)

        print("Stopping Telegram bot...")
        await application.stop()
        print("Trading bot execution completed.")

    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}"
        print(error_msg)
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        await send_notification(error_msg)

if __name__ == "__main__":
    asyncio.run(main())
