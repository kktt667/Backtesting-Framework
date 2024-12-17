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
            
            # Store necessary columns and format DataFrame
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
            # Calculate daily returns and cumulative returns
            self.df['return'] = self.df['close'].pct_change()
            self.df['cumulative_return'] = (1 + self.df['return']).cumprod() - 1
        else:
            print("DataFrame is empty. Cannot calculate returns.")

    def find_minimum_volume_of_max_return(self):
        """Finds the minimum volume corresponding to the highest cumulative return."""
        if self.df is None or self.df.empty:
            print("No data available.")
            return None, None, None
        
        # Ensure returns are calculated
        self.calculate_cumulative_returns()
        
        # Check cumulative return availability
        if self.df['cumulative_return'].empty:
            print("No cumulative returns available.")
            return None, None, None
        
        # Find minimum volume at maximum return timestamp
        max_return_idx = self.df['cumulative_return'].idxmax()
        min_volume = self.df.loc[max_return_idx, 'volume']
        
        return self.symbol, max_return_idx, min_volume

    def filter_signals(self):
        """Filters signals based on a dynamic volume threshold."""
        if self.df is None or self.df.empty:
            print("Data is missing; unable to filter signals.")
            return pd.DataFrame()
        
        if 'volume' not in self.df.columns:
            raise ValueError("Volume data missing from DataFrame.")
        
        # Calculate a 50-period moving average on volume for dynamic thresholding
        self.df['volume_ma'] = self.df['volume'].rolling(window=50).mean()
        dynamic_threshold = self.df['volume_ma'] * 1.5
        
        # Apply filter: retain rows with volume exceeding dynamic threshold
        filtered_df = self.df[self.df['volume'] > dynamic_threshold].copy()
        return filtered_df[['open', 'high', 'low', 'close', 'volume']]
