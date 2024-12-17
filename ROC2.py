import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import ParameterGrid
import os

class ROCStrategy:
    def __init__(self, symbol, interval, lookback_days=365):
        self.symbol = symbol
        self.interval = interval
        self.lookback_days = lookback_days
        self.df = pd.DataFrame()
        self.signals_df = pd.DataFrame()
        self.best_params = None

    def get_binance_ohlc(self, limit=1000):
        base_url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_asset_volume', 'number_of_trades',
                                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric columns are converted to float
            for col in ['high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.df = df[['high', 'low', 'close', 'volume']]
        except requests.RequestException as e:
            print(f"Request Error: {e}")

    def calculate_roc(self, df, period):
        df['ROC'] = df['close'].pct_change(periods=period) * 100
        return df

    def identify_signals(self, df, overbought_threshold, oversold_threshold):
        df['Signal'] = np.where(df['ROC'] > overbought_threshold, 'Sell',
                                np.where(df['ROC'] < oversold_threshold, 'Buy', ''))
        return df

    def calculate_cumulative_returns(self, df):
        df = df.copy()
        df['Signal'] = df['Signal'].map({'Buy': 1, 'Sell': -1, '': 0})
        df['strategy_returns'] = df['Signal'].shift(1) * df['close'].pct_change()
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
        return df['cumulative_returns'].iloc[-1]

    def grid_search(self, df, param_grid):
        best_score = -np.inf
        best_params = None
        for params in ParameterGrid(param_grid):
            df_copy = df.copy()
            df_copy = self.calculate_roc(df_copy, period=params['roc_period'])
            df_copy = self.identify_signals(df_copy, params['overbought_threshold'], params['oversold_threshold'])
            score = self.calculate_cumulative_returns(df_copy)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    def walk_forward_optimization(self, initial_train_size, test_size, param_grid):
        df = self.df
        total_size = len(df)
        train_end = initial_train_size
        all_results = []
        all_best_params = []
        while train_end + test_size <= total_size:
            train_df = df[:train_end].copy()
            test_df = df[train_end:train_end + test_size].copy()
            best_params, best_score = self.grid_search(train_df, param_grid)
            test_df = self.calculate_roc(test_df, period=best_params['roc_period'])
            test_df = self.identify_signals(test_df, best_params['overbought_threshold'], best_params['oversold_threshold'])
            cumulative_return = self.calculate_cumulative_returns(test_df)
            all_results.append(float(cumulative_return))
            all_best_params.append(best_params)
            train_end += test_size
        return all_results, all_best_params

    def align_signals_with_ohlc(self):
        if self.df is None or self.signals_df is None:
            raise ValueError("DataFrames must be initialized before alignment.")
        
        # Ensure 'timestamp' is in the index for both DataFrames
        if 'timestamp' in self.df.columns:
            self.df.set_index('timestamp', inplace=True)
        if 'timestamp' in self.signals_df.columns:
            self.signals_df.set_index('timestamp', inplace=True)
        
        self.df = self.df.join(self.signals_df[['Signal']], how='left')
        self.df['Signal'].fillna('', inplace=True)

    def save_signals(self, file_path):
        if self.df is not None and not self.df.empty:
            # Filter out rows where signal is 0
            filtered_df = self.df[self.df['signal'] != 0]
            
            # Include necessary columns and sort
            signals = filtered_df[['Signal', 'volume']]
            signals.reset_index(inplace=True)  # Ensure timestamp is included as a column
            
            # Save to CSV
            if os.path.exists(file_path):
                signals.to_csv(file_path, mode='a', header=False, index=False)
            else:
                signals.to_csv(file_path, mode='w', header=True, index=False)
        else:
            print("No signals to save.")

    def run_optimization(self, initial_train_size, test_size, param_grid, save_path='SIGNALS_FILE.csv'):
        self.get_binance_ohlc()
        cumulative_returns, best_params = self.walk_forward_optimization(initial_train_size, test_size, param_grid)
        self.df = self.calculate_roc(self.df, period=best_params[-1]['roc_period'])
        self.df = self.identify_signals(self.df, best_params[-1]['overbought_threshold'], best_params[-1]['oversold_threshold'])
        
        # Create signals_df with timestamp included
        self.signals_df = self.df[self.df['Signal'] != ''].reset_index()
        self.signals_df = self.signals_df[['timestamp', 'Signal', 'volume']]
        self.signals_df.sort_values(by='timestamp', inplace=True)
        self.signals_df.drop_duplicates(subset=['timestamp', 'Signal'], keep='last', inplace=True)
        
        # Save signals to CSV
        self.signals_df.to_csv(save_path, index=False)
        return cumulative_returns

    def load_signals(self):
        if os.path.exists(self.signals_file):
            df = pd.read_csv(self.signals_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                raise KeyError("The 'timestamp' column is missing from the signals file.")
            df.drop_duplicates(subset=['timestamp', 'Signal'], inplace=True)
            return df
        else:
            raise FileNotFoundError(f"{self.signals_file} not found.")
