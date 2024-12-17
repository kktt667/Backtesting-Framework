import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import os

class RSIAnalysis:
    def __init__(self, symbol, interval, rsi_windows, overbought_thresholds, oversold_thresholds):
        self.symbol = symbol
        self.interval = interval
        self.rsi_windows = rsi_windows
        self.overbought_thresholds = overbought_thresholds
        self.oversold_thresholds = oversold_thresholds
        self.df = pd.DataFrame()
        self.best_params = None
        self.signals_file = 'SIGNALS_FILE.csv'

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
            
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            self.df = df
        except requests.RequestException as e:
            print(f"Request Error: {e}")

    def calculate_rsi(self, df, period):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi
        return df

    def identify_signals(self, df, overbought_threshold, oversold_threshold):
        df['buy_signal'] = (df['RSI'] < oversold_threshold) & (df['RSI'].shift(1) >= oversold_threshold)
        df['sell_signal'] = (df['RSI'] > overbought_threshold) & (df['RSI'].shift(1) <= overbought_threshold)
        df['Signal'] = np.where(df['buy_signal'], 'Buy', np.where(df['sell_signal'], 'Sell', None))
        df['Volume'] = df['volume']  # Ensure volume is included
        return df

    def calculate_cumulative_returns(self, df):
        df['position'] = np.where(df['Signal'] == 'Buy', 1, np.where(df['Signal'] == 'Sell', -1, 0))
        df['daily_returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['daily_returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
        return df['cumulative_returns'].iloc[-1] if not df.empty else np.nan

    def grid_search(self, df):
        best_score = -np.inf
        best_params = None
        
        for params in ParameterGrid({
            'rsi_period': self.rsi_windows, 
            'overbought_threshold': self.overbought_thresholds, 
            'oversold_threshold': self.oversold_thresholds
        }):
            df_copy = df.copy()
            df_copy = self.calculate_rsi(df_copy, period=params['rsi_period'])
            df_copy = self.identify_signals(df_copy, 
                                            overbought_threshold=params['overbought_threshold'], 
                                            oversold_threshold=params['oversold_threshold'])
            total_return = self.calculate_cumulative_returns(df_copy)
            
            if total_return > best_score:
                best_score = total_return
                best_params = params
        
        return best_params, best_score

    def walk_forward_optimization(self, initial_train_size, test_size):
        total_size = len(self.df)
        train_end = initial_train_size
        all_cumulative_returns = []
        all_best_params = []
        
        while train_end + test_size <= total_size:
            train_df = self.df[:train_end].copy()
            test_df = self.df[train_end:train_end + test_size].copy()
            
            best_params, _ = self.grid_search(train_df)
            test_df = self.calculate_rsi(test_df, period=best_params['rsi_period'])
            test_df = self.identify_signals(test_df, 
                                            overbought_threshold=best_params['overbought_threshold'], 
                                            oversold_threshold=best_params['oversold_threshold'])
            
            cumulative_return = self.calculate_cumulative_returns(test_df)
            all_cumulative_returns.append(cumulative_return)
            all_best_params.append(best_params)
            
            train_end += test_size
        
            if all_best_params:
                self.best_params = all_best_params[-1]  # Use the last best parameters
            else:
                self.best_params = None  # Set to None if no parameters were found
        return all_cumulative_returns

    def run_optimization(self, initial_train_size, test_size, save_path='SIGNALS_FILE.csv'):
        self.get_binance_ohlc()
        
        cumulative_returns = self.walk_forward_optimization(initial_train_size, test_size)
        
        if not cumulative_returns or self.best_params is None:
            print("No RSI signals generated. Check data and parameter settings.")
            return []

        self.df = self.calculate_rsi(self.df, period=self.best_params['rsi_period'])
        self.df = self.identify_signals(self.df, 
                                        overbought_threshold=self.best_params['overbought_threshold'], 
                                        oversold_threshold=self.best_params['oversold_threshold'])
        
        self.signals_df = self.df[self.df['Signal'] != ''].reset_index()
        self.signals_df = self.signals_df[['timestamp', 'Signal', 'Volume']]
        self.signals_df.sort_values(by='timestamp', inplace=True)
        self.signals_df.drop_duplicates(subset=['timestamp', 'Signal'], keep='last', inplace=True)
        
        if not self.signals_df.empty:
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                existing_signals = pd.read_csv(save_path)
                combined_df = pd.concat([existing_signals, self.signals_df])
                combined_df.drop_duplicates(subset=['timestamp', 'Signal'], keep='last', inplace=True)
                combined_df.to_csv(save_path, index=False)
            else:
                self.signals_df.to_csv(save_path, index=False)
        else:
            print("No signals generated. Not saving to file.")

        return cumulative_returns
