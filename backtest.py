import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Backtester:
    def __init__(self, signals_file, symbol, interval, min_volume=None, use_vts=False,):
        self.signals_file = signals_file
        self.symbol = symbol
        self.interval = interval
        self.min_volume = min_volume
        self.use_vts = use_vts
        self.signals_df = self.load_signals()
        self.ohlc_data =  None #self.get_binance_ohlc(self, start_time, end_time) 
       
        self.results = {}  # Initialize results as empty dictionary

        if self.use_vts and self.min_volume is not None:
            self.signals_df = self.filter_signals_by_volume()

    def load_signals(self):
        if os.path.exists(self.signals_file) and os.path.getsize(self.signals_file) > 0:
            df = pd.read_csv(self.signals_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.drop_duplicates(subset=['timestamp', 'Signal'], inplace=True)
            return df
        else:
            print(f"Warning: {self.signals_file} is empty or not found.")
            return pd.DataFrame(columns=['timestamp', 'Signal', 'Volume'])

    def filter_signals_by_volume(self):
        if not self.use_vts:
            return self.signals_df
        
        if 'Volume' not in self.signals_df.columns:
            raise ValueError("Volume column not found in signals data.")
            
        return self.signals_df[self.signals_df['Volume'] > self.min_volume]

    def get_binance_ohlc(self, start_time, end_time, limit=1000):#start_time, end_time,
        base_url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
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
            df = df[['open', 'high', 'low', 'close']]  # Exclude 'volume' as it's not needed here
            df = df.astype(float)
            return df
        
        except requests.RequestException as e:
            print(f"Request Error: {e}")
            return None


    def align_signals_with_ohlc(self):
        start_time = self.signals_df['timestamp'].min()
        end_time = self.signals_df['timestamp'].max()
        
        # Fetch OHLC data
        self.ohlc_data = self.get_binance_ohlc(start_time, end_time,)
        if self.ohlc_data is None:
            raise ValueError("Failed to fetch OHLC data.")
        
        # Ensure OHLC data covers the same period as signals
        self.ohlc_data = self.ohlc_data[self.ohlc_data.index.isin(self.signals_df['timestamp'])]

        # Rename overlapping columns in signals_df to avoid conflicts
        signals_df_renamed = self.signals_df.rename(columns=lambda x: x + '_signals' if x in self.ohlc_data.columns else x)
        
        # Merge signals with OHLC data
        self.ohlc_data = self.ohlc_data.join(signals_df_renamed.set_index('timestamp'), how='left')
        
        if self.ohlc_data.empty:
            raise ValueError("No OHLC data available for the given signals.")



    def calculate_price_changes(self, df, time_deltas):
        for minutes, periods in time_deltas.items():
            df[f'price_change_{minutes}m'] = df['close'].shift(-periods) / df['close'] - 1

            buy_signals = df[df['Signal'] == 'Buy']
            sell_signals = df[df['Signal'] == 'Sell']

            if not buy_signals.empty:
                buy_changes = buy_signals[f'price_change_{minutes}m'].dropna()
                if not buy_changes.empty:
                    self.results[f'{minutes}_min_avg_price_change'] = buy_changes.mean()
                    self.results[f'{minutes}_min_median_price_change'] = buy_changes.median()
                    self.results[f'{minutes}_min_win_rate'] = (buy_changes > 0).mean()
                    self.results[f'{minutes}_min_sharpe_ratio'] = self.calculate_sharpe_ratio(buy_changes)
                    print(f"3-day Buy Changes: {buy_changes.describe()}")  # Debugging statement
            
            if not sell_signals.empty:
                sell_changes = sell_signals[f'price_change_{minutes}m'].dropna()
                if not sell_changes.empty:
                    self.results[f'{minutes}_min_avg_price_change_sell'] = sell_changes.mean()
                    self.results[f'{minutes}_min_median_price_change_sell'] = sell_changes.median()
                    self.results[f'{minutes}_min_win_rate_sell'] = (sell_changes > 0).mean()
                    self.results[f'{minutes}_min_sharpe_ratio_sell'] = self.calculate_sharpe_ratio(sell_changes)
                    print(f"3-day Sell Changes: {sell_changes.describe()}")  # Debugging statement


    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        returns = returns.dropna()
        if returns.empty:
            return np.nan
        return_ratio = returns.mean()
        return_std = returns.std()
        sharpe_ratio = (return_ratio - risk_free_rate) / return_std if return_std != 0 else np.nan
        return sharpe_ratio


    def calculate_drawdown(self, df):
        if df.empty or 'price_change_1440m' not in df.columns:
            return {'max_drawdown': np.nan, 'max_drawdown_duration': np.nan}

        df['cumulative_return'] = (1 + df['price_change_1440m']).cumprod()
        df['cumulative_peak'] = df['cumulative_return'].cummax()
        df['drawdown'] = df['cumulative_return'] - df['cumulative_peak']

        max_drawdown = df['drawdown'].min()
        drawdown_duration = df['drawdown'].groupby((df['drawdown'] == 0).cumsum()).transform('size').max()

        return {'max_drawdown': max_drawdown, 'max_drawdown_duration': drawdown_duration}

    def generate_backtest_summary(self, time_deltas):
        self.results['Total Occurrences'] = len(self.signals_df)
        
        self.calculate_price_changes(self.ohlc_data, time_deltas)
        drawdown_stats = self.calculate_drawdown(self.ohlc_data)
        self.results.update(drawdown_stats)  # Update results with drawdown statistics
        
        # Convert relevant results to percentages
        for key in self.results:
            if 'change' in key or 'rate' in key or 'drawdown' in key:
                self.results[key] = self.results[key] * 100  # Convert to percentage
        
        # Create a structured DataFrame for table format
        summary_data = {
            'Metric': ['Avg Price Change', 'Median Price Change', 'Win Rate', 'Sharpe Ratio'],
            '5 min': [],
            '15 min': [],
            '1 hour': [],
            '4 hours': [],
            '1 day': [],
            '3 days': []  # Ensure 3-day time delta is included
        }
        
        # Populate the DataFrame with results
        for minutes, label in zip([5, 15, 60, 240, 1440, 4320], ['5 min', '15 min', '1 hour', '4 hours', '1 day', '3 days']):
            summary_data[label].append(f"{self.results.get(f'{minutes}_min_avg_price_change', 0):.2f}%")
            summary_data[label].append(f"{self.results.get(f'{minutes}_min_median_price_change', 0):.2f}%")
            summary_data[label].append(f"{self.results.get(f'{minutes}_min_win_rate', 0):.2f}%")
            summary_data[label].append(f"{self.results.get(f'{minutes}_min_sharpe_ratio', 0):.2f}")
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df


    def backtest(self):
        time_deltas = {
            5: 1,      # 5 minutes
            15: 3,     # 15 minutes
            60: 12,    # 1 hour
            240: 48,   # 4 hours
            1440: 288, # 1 day
            4320: 864  # 3 days (make sure this is included)
        }
        self.align_signals_with_ohlc()
        summary = self.generate_backtest_summary(time_deltas)

        self.plot_signals(self.ohlc_data)
        self.print_results(summary)

        return summary

    def plot_signals(self, df):
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
        buy_signals = df[df['Signal'] == 'Buy']
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', alpha=1, s=100)
        sell_signals = df[df['Signal'] == 'Sell']
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', alpha=1, s=100)
        plt.title(f'{self.symbol} Price with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # def print_results(self, summary):
    #     print("\nBacktest Summary:")
        
    #     for key, value in summary.items():
    #         print(f"{key}: {value}")

# Example usage:
# backtester = Backtester('signals.csv', 'BTCUSDT', '5m', min_volume=1000, use_vts=True)
# results = backtester.backtest()
# print(results)






