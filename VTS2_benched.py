import requests
import pandas as pd

class VTSAnalyzer:
    def __init__(self, symbol, interval='1d', limit=1000):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.df = None

    def get_binance_ohlc(self):
        base_url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': self.limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            self.df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            self.df.set_index('timestamp', inplace=True)
            
            self.df = self.df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        except requests.RequestException as e:
            print(f"Request Error: {e}")
            self.df = pd.DataFrame()


    def calculate_cumulative_returns(self):
        if self.df is not None and not self.df.empty:
            self.df['return'] = self.df['close'].pct_change()
            self.df['cumulative_return'] = (1 + self.df['return']).cumprod() - 1
        else:
            print("DataFrame is empty. Cannot calculate returns.")

    def find_minimum_volume_of_max_return(self):
        if self.df is None or self.df.empty:
            print("No data available.")
            return None, None, None
        
        self.calculate_cumulative_returns()
        
        if self.df['cumulative_return'].empty:
            print("No cumulative returns available.")
            return None, None, None
        
        # Find the minimum volume associated with the highest cumulative return
        max_return_idx = self.df['cumulative_return'].idxmax()
        min_volume = self.df.loc[max_return_idx, 'volume']
        
        return self.symbol, max_return_idx, min_volume

    # def filter_signals(self, signals_df):
    #     _, max_return_idx, min_volume = self.find_minimum_volume_of_max_return()
    #     if max_return_idx is not None:
    #         # Filter signals based on the timestamp of max return and minimum volume
    #         filtered_signals = signals_df[(signals_df['timestamp'] <= max_return_idx) & (signals_df['volume'] >= min_volume)]
    #         return filtered_signals
    #     else:
    #         print("No valid signals to filter.")
    #         return pd.DataFrame()

    def filter_signals(self):
        if 'volume' not in self.df.columns:
            raise ValueError("Volume data missing from DataFrame.")

        # Calculate a 50-period moving average on volume
        self.df['volume_ma'] = self.df['volume'].rolling(window=50).mean()

        # Set dynamic threshold as 1.5 times the moving average
        dynamic_threshold = self.df['volume_ma'] * 1.5

        # Filter signals based on dynamic threshold
        return self.df[self.df['volume'] > dynamic_threshold]


