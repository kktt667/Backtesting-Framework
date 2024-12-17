import requests  # Importing requests for making HTTP requests to APIs
import pandas as pd  # Importing Pandas for data manipulation and analysis
import numpy as np  # Importing NumPy for numerical operations
from sklearn.model_selection import ParameterGrid  # Importing ParameterGrid for hyperparameter tuning
import os  # Importing os for interacting with the operating system

class BBSTRAT:
    def __init__(self, symbol, interval, window_range, num_std_dev_range):
        """
        Initializes the Bollinger Bands strategy with the given parameters.

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            interval (str): The time interval for the data (e.g., '1d').
            window_range (list): List of window sizes for Bollinger Bands.
            num_std_dev_range (list): List of standard deviation multipliers for Bollinger Bands.
        """
        self.symbol = symbol  # Trading symbol
        self.interval = interval  # Time interval for the data
        self.window_range = window_range  # Window sizes for Bollinger Bands
        self.num_std_dev_range = num_std_dev_range  # Standard deviation multipliers
        self.df = pd.DataFrame()  # DataFrame to hold OHLC data
        self.best_signals = pd.DataFrame()  # DataFrame to hold generated signals
        self.best_params = {}  # Dictionary to store the best parameters found
        self.best_cumulative_return = -np.inf  # Initialize best cumulative return to negative infinity

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
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
            df.set_index('timestamp', inplace=True)  # Set timestamp as the index
            
            # Select relevant columns and convert them to float
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            self.df = df  # Store the OHLC data in the instance variable
        except requests.RequestException as e:
            print(f"Request Error: {e}")  # Print any request errors

    def calculate_bollinger_bands(self, df, window, num_std_dev):
        """
        Calculates Bollinger Bands for the given DataFrame.

        Args:
            df (DataFrame): DataFrame containing OHLC data.
            window (int): The window size for calculating the moving average.
            num_std_dev (float): The number of standard deviations for the bands.

        Returns:
            DataFrame: DataFrame with Bollinger Bands added.
        """
        df['rolling_mean'] = df['close'].rolling(window=window).mean()  # Calculate rolling mean
        df['rolling_std'] = df['close'].rolling(window=window).std()  # Calculate rolling standard deviation
        df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * num_std_dev)  # Calculate upper band
        df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * num_std_dev)  # Calculate lower band
        return df  # Return the DataFrame with Bollinger Bands

    def identify_signals(self, df):
        """
        Identifies buy and sell signals based on Bollinger Bands.

        Args:
            df (DataFrame): DataFrame containing OHLC data with Bollinger Bands.

        Returns:
            DataFrame: DataFrame with signals added.
        """
        df['buy_signal'] = df['close'] < df['lower_band']  # Buy signal when price is below lower band
        df['sell_signal'] = df['close'] > df['upper_band']  # Sell signal when price is above upper band
        df['Signal'] = np.where(df['buy_signal'], 'Buy', np.where(df['sell_signal'], 'Sell', None))  # Assign signals
        df['Volume'] = df['volume']  # Ensure volume is included
        df_signals = df[['Signal', 'Volume', 'close']].copy()  # Create a DataFrame for signals
        df_signals.reset_index(inplace=True)  # Reset index to make timestamp a column
        return df_signals  # Return the signals DataFrame

    def calculate_cumulative_returns(self, df):
        """
        Calculates cumulative returns based on the signals.

        Args:
            df (DataFrame): DataFrame containing OHLC data and signals.

        Returns:
            float: Cumulative return of the strategy.
        """
        if 'close' not in df.columns:
            print("Warning: 'close' column not found. Using 'Signal' column for returns calculation.")
            df['position'] = np.where(df['Signal'] == 'Buy', 1, np.where(df['Signal'] == 'Sell', -1, 0))  # Assign positions
            df['daily_returns'] = df['position'].diff()  # Calculate daily returns
        else:
            df['position'] = np.where(df['Signal'] == 'Buy', 1, np.where(df['Signal'] == 'Sell', -1, 0))  # Assign positions
            df['daily_returns'] = df['close'].pct_change()  # Calculate daily returns based on closing prices
        df['strategy_returns'] = df['position'].shift(1) * df['daily_returns']  # Calculate strategy returns
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1  # Calculate cumulative returns
        return df['cumulative_returns'].iloc[-1] if not df.empty else np.nan  # Return the last cumulative return

    def grid_search(self, df):
        """
        Performs grid search to find the best parameters for the strategy.

        Args:
            df (DataFrame): DataFrame containing OHLC data.

        Returns:
            tuple: Best parameters and the corresponding score.
        """
        best_score = -np.inf  # Initialize best score
        best_params = None  # Initialize best parameters
        
        # Iterate over all combinations of parameters
        for params in ParameterGrid({'window': self.window_range, 'num_std_dev': self.num_std_dev_range}):
            df_copy = df.copy()  # Create a copy of the DataFrame
            df_copy = self.calculate_bollinger_bands(df_copy, window=params['window'], num_std_dev=params['num_std_dev'])  # Calculate Bollinger Bands
            df_copy = self.identify_signals(df_copy)  # Identify signals
            total_return = self.calculate_cumulative_returns(df_copy)  # Calculate cumulative return
            
            # Update best score and parameters if a better score is found
            if total_return > best_score:
                best_score = total_return
                best_params = params
        
        return best_params, best_score  # Return best parameters and score

    def walk_forward_optimization(self, initial_train_size, test_size):
        """
        Performs walk-forward optimization to evaluate strategy performance.

        Args:
            initial_train_size (int): Size of the initial training set.
            test_size (int): Size of the test set.

        Returns:
            list: List of cumulative returns for each test.
        """
        total_size = len(self.df)  # Total size of the data
        train_end = initial_train_size  # End index for training data
        all_cumulative_returns = []  # List to store cumulative returns
        
        # Loop through the data for walk-forward optimization
        while train_end + test_size <= total_size:
            train_df = self.df[:train_end].copy()  # Training data
            test_df = self.df[train_end:train_end + test_size].copy()  # Test data
            
            best_params, _ = self.grid_search(train_df)  # Find best parameters on training data
            test_df = self.calculate_bollinger_bands(test_df, window=best_params['window'], num_std_dev=best_params['num_std_dev'])  # Calculate Bollinger Bands on test data
            test_df = self.identify_signals(test_df)  # Identify signals on test data
            
            cumulative_return = self.calculate_cumulative_returns(test_df)  # Calculate cumulative return
            all_cumulative_returns.append(cumulative_return)  # Store cumulative return
            
            train_end += test_size  # Move the training end index forward
        
        return all_cumulative_returns  # Return cumulative returns

    def run_optimization(self, initial_train_size, test_size, save_path='SIGNALS_FILE.csv'):
        """
        Runs the optimization process and saves the generated signals.

        Args:
            initial_train_size (int): Size of the initial training set.
            test_size (int): Size of the test set.
            save_path (str): Path to save the generated signals.

        Returns:
            list: List of cumulative returns from the optimization.
        """
        self.get_binance_ohlc()  # Fetch OHLC data
        
        cumulative_returns = self.walk_forward_optimization(initial_train_size, test_size)  # Perform optimization
        
        # Generate final signals based on the best parameters identified
        self.df = self.calculate_bollinger_bands(self.df, window=self.best_params['window'], num_std_dev=self.best_params['num_std_dev'])  # Calculate Bollinger Bands on full data
        self.df = self.identify_signals(self.df)  # Identify signals on full data
        
        # Prepare signals DataFrame for export
        self.best_signals = self.df[['Signal', 'Volume']].copy()  # Create a DataFrame for signals
        self.best_signals = self.best_signals[self.best_signals['Signal'] != '']  # Filter out empty signals
        
        # Reset index to match format
        self.best_signals.reset_index(inplace=True)  # Reset index to make timestamp a column
        self.best_signals.rename(columns={'index': 'timestamp'}, inplace=True)  # Rename index column
        self.best_signals.sort_values(by='timestamp', inplace=True)  # Sort by timestamp
        self.best_signals.drop_duplicates(subset=['timestamp', 'Signal'], keep='last', inplace=True)  # Remove duplicates
        
        # Save signals to CSV
        if os.path.exists(save_path):
            existing_signals = pd.read_csv(save_path)  # Read existing signals
            combined_df = pd.concat([existing_signals, self.best_signals])  # Combine with new signals
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'Signal'], keep='last')  # Remove duplicates
            combined_df.to_csv(save_path, index=False)  # Save combined signals
        else:
            self.best_signals.to_csv(save_path, index=False)  # Save new signals

        return cumulative_returns  # Return cumulative returns
