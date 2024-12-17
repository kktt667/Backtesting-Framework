import pandas as pd
import requests
import datetime
import os

# File path for cached data
BINANCE_CACHE_FILE = 'binance_assets.csv'
CACHE_EXPIRY_DAYS = 1  # Update cache every day

# Fetch data from Binance ticker price endpoint
def fetch_binance_assets():
    url = 'https://api.binance.com/api/v3/ticker/price'
    
    # Try fetching the data from Binance API
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        
        # Extract the asset symbols that trade against USDT
        assets = [item['symbol'][:-4] for item in data if item['symbol'].endswith('USDT')]
        
        # Check if any assets were found
        if not assets:
            print("No assets found trading against USDT.")
            return
        
        # Remove duplicates and sort
        assets = sorted(set(assets))
        
        # Log the number of assets found
        print(f"Found {len(assets)} assets trading against USDT.")
        
        # Create a DataFrame and save it as a CSV
        df = pd.DataFrame(assets, columns=['Asset'])
        df.to_csv(BINANCE_CACHE_FILE, index=False)
        print(f"Asset data saved to {BINANCE_CACHE_FILE}.")
        
    except requests.RequestException as e:
        print(f"Error fetching data from Binance: {e}")

# Load data from CSV
def load_csv(filename):
    return pd.read_csv(filename)

# Check if a file is empty
def is_file_empty(filename):
    return os.path.exists(filename) and os.path.getsize(filename) == 0

# Update assets if cache is outdated or file is empty
def update_cache():
    # Check if the file exists, if it's empty, or if it's older than the expiry time
    if not os.path.exists(BINANCE_CACHE_FILE) or is_file_empty(BINANCE_CACHE_FILE) or (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(BINANCE_CACHE_FILE))).days >= CACHE_EXPIRY_DAYS:
        print("Updating asset cache...")
        fetch_binance_assets()
    else:
        print("Cache is up to date and file is not empty.")

# Main function to update cache
if __name__ == "__main__":
    update_cache()
