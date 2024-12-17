import pandas as pd  # Importing Pandas for data manipulation and analysis
import numpy as np  # Importing NumPy for numerical operations
import requests  # Importing requests for making HTTP requests to APIs
from sklearn.model_selection import ParameterGrid  # Importing ParameterGrid for hyperparameter tuning
from datetime import datetime, timedelta  # Importing datetime for handling date and time
import os  # Importing os for interacting with the operating system

class MACDStrategy:
    def __init__(self, symbol, interval, short_windows, long_windows, signal_windows, step_size=90):
        """
        Initializes the MACD strategy with the given parameters.

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            interval (str): The time interval for the data (e.g., '1d').
            short_windows (list): List of short window sizes for MACD.
            long_windows (list): List of long window sizes for MACD.
            signal_windows (list): List of signal window sizes for MACD.
            step_size (int): The step size for walk-forward optimization (default is 90).
        """
        self.symbol = symbol  # Trading symbol
        self.interval = interval  # Time interval for the data
        self.short_windows = short_windows  # Short window sizes for MACD
        self.long_windows = long_windows  # Long window sizes for MACD
        self.signal_windows = signal_windows  # Signal window sizes for MACD
        self.step_size = step_size  # Step size for walk-forward optimization
        self.df = pd.DataFrame()  # DataFrame to hold OHLC data
        self.best_signals = pd.DataFrame()  # DataFrame to hold generated signals
        self.best_params = None  # Variable to store the best parameters found

    def get_binance_ohlc(self, limit=1000):
        """
        Fetches OHLC data from Binance API and stores it in the DataFrame.

        Args:
            limit (int): The number of data points to fetch (default is 1000).
        """
        base_url = 'https://api.binance.com/api/v3/klines'  # Binance API endpoint for OHLC data
        params = {
            'symbol': self.symbol,  # Trading symbol
            'interval': self.interval,  # Time interval
            'limit': limit  # Limit on the number of data points
        }
        
        try:
            response = requests.get(base_url, params=params)  # Make the API request
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()  # Parse the JSON response
            
            # Create a DataFrame from the response data
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_asset_volume', 'number_of_trades',
                                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
            df.set_index('timestamp', inplace=True)  # Set timestamp as the index
            
            # Select relevant columns and convert them to float
            df = df[['high', 'low', 'close', 'volume']].astype(float)
            self.df = df  # Store the OHLC data in the instance variable
        except requests.RequestException as e:
            print(f"Request Error: {e}")  # Print any request errors

    def calculate_macd(self, df, short_window, long_window, signal_window):
        """
        Calculates the MACD and its signal line for the given DataFrame.

        Args:
            df (DataFrame): DataFrame containing OHLC data.
            short_window (int): The short window size for MACD.
            long_window (int): The long window size for MACD.
            signal_window (int): The window size for the signal line.

        Returns:
            DataFrame: DataFrame with MACD values added.
        """
        df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()  # Calculate short EMA
        df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()  # Calculate long EMA
        df['macd'] = df['ema_short'] - df['ema_long']  # Calculate MACD line
        df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()  # Calculate signal line
        df['macd_hist'] = df['macd'] - df['macd_signal']  # Calculate MACD histogram
        return df  # Return the DataFrame with MACD values

    def generate_macd_signals(self, df):
        """
        Generates buy and sell signals based on MACD values.

        Args:
            df (DataFrame): DataFrame containing MACD values.

        Returns:
            DataFrame: DataFrame with signals added.
        """
        df['buy_signal'] = df['macd'] > df['macd_signal']  # Buy signal when MACD crosses above signal line
        df['sell_signal'] = df['macd'] < df['macd_signal']  # Sell signal when MACD crosses below signal line
        df['position'] = 0  # Initialize position column
        df.loc[df['buy_signal'], 'position'] = 1  # Set position to 1 for buy signals
        df.loc[df['sell_signal'], 'position'] = -1  # Set position to -1 for sell signals
        df['daily_returns'] = df['close'].pct_change()  # Calculate daily returns
        df['strategy_returns'] = df['position'].shift(1) * df['daily_returns']  # Calculate strategy returns
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1  # Calculate cumulative returns
        df['Volume'] = np.nan  # Initialize Volume column
        df.loc[df['buy_signal'] | df['sell_signal'], 'Volume'] = df['volume']  # Set volume for signals
        return df  # Return the DataFrame with signals

    def evaluate_performance(self, df):
        """
        Evaluates the performance of the strategy based on cumulative returns.

        Args:
            df (DataFrame): DataFrame containing cumulative returns.

        Returns:
            float: Last cumulative return if available, otherwise NaN.
        """
        if 'cumulative_returns' in df.columns and not df.empty:
            last_cumulative_return = df['cumulative_returns'].iloc[-1]  # Get last cumulative return
            return last_cumulative_return  # Return the last cumulative return
        return np.nan  # Return NaN if not available

    def walk_forward_optimization(self):
        """
        Performs walk-forward optimization to find the best parameters for the strategy.

        Returns:
            DataFrame: DataFrame containing the best signals generated.
        """
        self.best_signals = pd.DataFrame()  # Initialize best signals DataFrame
        results = []  # List to store results
        num_steps = (len(self.df) - self.step_size) // self.step_size  # Calculate number of steps

        # Loop through the data for walk-forward optimization
        for i in range(num_steps):
            train_start = self.df.index[0] + timedelta(days=i * self.step_size)  # Start of training data
            train_end = train_start + timedelta(days=self.step_size - 1)  # End of training data
            test_start = train_end + timedelta(days=1)  # Start of test data
            test_end = test_start + timedelta(days=self.step_size - 1)  # End of test data

            train_df = self.df.loc[train_start:train_end]  # Training DataFrame
            test_df = self.df.loc[test_start:test_end]  # Test DataFrame

            if train_df.empty or test_df.empty:  # Skip if either DataFrame is empty
                continue

            best_return = -np.inf  # Initialize best return
            best_params = None  # Initialize best parameters

            # Iterate over all combinations of parameters
            for params in ParameterGrid({
                'short_window': self.short_windows,
                'long_window': self.long_windows,
                'signal_window': self.signal_windows
            }):
                temp_df = self.calculate_macd(train_df.copy(), params['short_window'], params['long_window'], params['signal_window'])  # Calculate MACD
                temp_df = self.generate_macd_signals(temp_df)  # Generate signals
                performance = self.evaluate_performance(temp_df)  # Evaluate performance
                
                # Update best return and parameters if a better performance is found
                if not np.isnan(performance) and performance > best_return:
                    best_return = performance  # Update best return
                    best_params = params  # Update best parameters
                    self.best_signals = test_df.copy()  # Copy test DataFrame for best signals
                    self.best_signals = self.calculate_macd(self.best_signals, params['short_window'], params['long_window'], params['signal_window'])  # Calculate MACD for best signals
                    self.best_signals = self.generate_macd_signals(self.best_signals)  # Generate signals for best signals

        if best_params is not None:
            self.best_params = best_params  # Store the best parameters

        # Deduplicate best signals DataFrame
        self.best_signals = self.best_signals.reset_index()  # Reset index
        self.best_signals = self.best_signals[['timestamp', 'buy_signal', 'sell_signal', 'Volume']]  # Select relevant columns

        # Generate Signal column based on buy_signal and sell_signal
        self.best_signals['Signal'] = np.where(self.best_signals['buy_signal'], 'Buy',
                                            np.where(self.best_signals['sell_signal'], 'Sell', ''))  # Assign signals
        
        # Keep only rows where 'Signal' is not empty
        self.best_signals = self.best_signals[['timestamp', 'Signal', 'Volume']]  # Select relevant columns
        self.best_signals = self.best_signals[self.best_signals['Signal'] != '']  # Filter out empty signals
        
        # Sort by timestamp to keep the latest signal if duplicates occur
        self.best_signals = self.best_signals.sort_values(by='timestamp')  # Sort by timestamp

        # Drop duplicate rows based on timestamp and Signal, keeping the last entry
        self.best_signals = self.best_signals.drop_duplicates(subset=['timestamp', 'Signal'], keep='last')  # Remove duplicates

        return self.best_signals  # Return the best signals DataFrame

    def run_optimization(self, save_path='SIGNALS_FILE.csv'):
        """
        Runs the optimization process and saves the generated signals.

        Args:
            save_path (str): Path to save the generated signals.

        Returns:
            DataFrame: DataFrame containing the generated signals.
        """
        self.get_binance_ohlc()  # Fetch OHLC data
        signals_df = self.walk_forward_optimization()  # Perform walk-forward optimization
        
        if signals_df.empty:  # Check if signals DataFrame is empty
            print("No results found. Check data and parameter settings.")  # Print warning
            return pd.DataFrame()  # Return empty DataFrame

        # Save signals to CSV
        signals_df.to_csv(save_path, mode='w', header=True, index=False)  # Save signals to file
        return signals_df  # Return the signals DataFrame
