import pandas as pd  # Importing Pandas for data manipulation and analysis
import numpy as np  # Importing NumPy for numerical operations
import requests  # Importing requests for making HTTP requests to APIs
from sklearn.model_selection import ParameterGrid  # Importing ParameterGrid for hyperparameter tuning
import os  # Importing os for interacting with the operating system

class ATRSTRAT:
    def __init__(self, symbol, interval, atr_periods, multipliers):
        """
        Initializes the ATR strategy with the given parameters.

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            interval (str): The time interval for the data (e.g., '1d').
            atr_periods (list): List of ATR periods to test.
            multipliers (list): List of multipliers for signal generation.
        """
        self.symbol = symbol  # Trading symbol
        self.interval = interval  # Time interval for the data
        self.atr_periods = atr_periods  # ATR periods for optimization
        self.multipliers = multipliers  # Multipliers for signal generation
        self.df = pd.DataFrame()  # DataFrame to hold OHLC data
        self.signals_df = pd.DataFrame()  # DataFrame to hold generated signals
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
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            self.df = df  # Store the OHLC data in the instance variable
        except requests.RequestException as e:
            print(f"Request Error: {e}")  # Print any request errors

    def calculate_atr(self, df, window):
        """
        Calculates the Average True Range (ATR) for the given DataFrame.

        Args:
            df (DataFrame): DataFrame containing OHLC data.
            window (int): The window size for calculating ATR.

        Returns:
            DataFrame: DataFrame with ATR values added.
        """
        df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
        # Calculate True Range (TR)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                            np.abs(df['high'] - df['close'].shift(1)),
                            np.abs(df['low'] - df['close'].shift(1)))
        # Calculate ATR as the rolling mean of TR
        df['atr'] = df['tr'].rolling(window=window).mean()
        df.dropna(inplace=True)  # Drop rows with NaN values
        return df  # Return the DataFrame with ATR

    def identify_signals(self, df, atr_period, multiplier):
        """
        Identifies buy and sell signals based on ATR.

        Args:
            df (DataFrame): DataFrame containing OHLC data.
            atr_period (int): The ATR period to use for signal generation.
            multiplier (float): The multiplier for signal thresholds.

        Returns:
            DataFrame: DataFrame with signals added.
        """
        df = self.calculate_atr(df, window=atr_period)  # Calculate ATR
        # Generate buy and sell signals based on ATR
        df['buy_signal'] = df['close'] < df['close'].shift(1) - multiplier * df['atr']
        df['sell_signal'] = df['close'] > df['close'].shift(1) + multiplier * df['atr']
        
        df['position'] = 0  # Initialize position column
        df.loc[df['buy_signal'], 'position'] = 1  # Set position to 1 for buy signals
        df.loc[df['sell_signal'], 'position'] = -1  # Set position to -1 for sell signals
        df['position'] = df['position'].fillna(method='ffill')  # Forward fill positions
        
        # Calculate daily and strategy returns
        df['daily_returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['daily_returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1  # Calculate cumulative returns
        
        # Create a Signal column based on buy/sell signals
        df['Signal'] = np.where(df['buy_signal'], 'Buy', np.where(df['sell_signal'], 'Sell', ''))
        df['Volume'] = df['volume']  # Include volume in the DataFrame
        
        return df  # Return the DataFrame with signals

    def calculate_cumulative_returns(self, df):
        """
        Calculates the cumulative returns from the strategy.

        Args:
            df (DataFrame): DataFrame containing strategy returns.

        Returns:
            float: The cumulative return value.
        """
        return df['cumulative_returns'].iloc[-1] if not df.empty else np.nan  # Return the last cumulative return

    def grid_search(self, df):
        """
        Performs a grid search to find the best parameters for the strategy.

        Args:
            df (DataFrame): DataFrame containing OHLC data.

        Returns:
            tuple: Best parameters and the corresponding score.
        """
        best_score = -np.inf  # Initialize best score
        best_params = None  # Initialize best parameters
        
        # Iterate over all combinations of parameters
        for params in ParameterGrid({'atr_period': self.atr_periods, 'multiplier': self.multipliers}):
            df_copy = df.copy()  # Create a copy of the DataFrame
            df_copy = self.identify_signals(df_copy, params['atr_period'], params['multiplier'])  # Identify signals
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
            list: List of best parameters for each test.
        """
        total_size = len(self.df)  # Total size of the data
        train_end = initial_train_size  # End index for training data
        all_results = []  # List to store results
        all_best_params = []  # List to store best parameters
        
        # Loop through the data for walk-forward optimization
        while train_end + test_size <= total_size:
            train_df = self.df[:train_end].copy()  # Training data
            test_df = self.df[train_end:train_end + test_size].copy()  # Test data
            
            best_params, _ = self.grid_search(train_df)  # Find best parameters on training data
            test_df = self.identify_signals(test_df, best_params['atr_period'], best_params['multiplier'])  # Identify signals on test data
            
            cumulative_return = self.calculate_cumulative_returns(test_df)  # Calculate cumulative return
            all_results.append(float(cumulative_return))  # Store result
            all_best_params.append(best_params)  # Store best parameters
            
            train_end += test_size  # Move the training end index forward
        
        return all_results, all_best_params  # Return results and best parameters

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
        
        cumulative_returns, best_params = self.walk_forward_optimization(initial_train_size, test_size)  # Perform optimization
        
        if not cumulative_returns:
            print("No ATR signals generated. Check data and parameter settings.")
            return []  # Return empty if no signals were generated

        self.best_params = best_params[-1]  # Use the last best parameters
        self.df = self.identify_signals(self.df, self.best_params['atr_period'], self.best_params['multiplier'])  # Identify signals on full data
        
        # Prepare signals DataFrame for export
        self.signals_df = self.df[self.df['Signal'] != ''].reset_index()  # Filter signals
        self.signals_df = self.signals_df[['timestamp', 'Signal', 'Volume']]  # Select relevant columns
        self.signals_df.sort_values(by='timestamp', inplace=True)  # Sort by timestamp
        self.signals_df.drop_duplicates(subset=['timestamp', 'Signal'], keep='last', inplace=True)  # Remove duplicates
        
        # Save signals to CSV
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            existing_signals = pd.read_csv(save_path)  # Read existing signals
            combined_df = pd.concat([existing_signals, self.signals_df])  # Combine with new signals
            combined_df.drop_duplicates(subset=['timestamp', 'Signal'], keep='last', inplace=True)  # Remove duplicates
            combined_df.to_csv(save_path, index=False)  # Save combined signals
        else:
            self.signals_df.to_csv(save_path, index=False)  # Save new signals

        return cumulative_returns  # Return cumulative returns
