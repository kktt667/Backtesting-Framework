import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
#Import the classes for all the seperate scripts once written
from BB2 import BBSTRAT
from ATR2 import ATRSTRAT
from MACD2 import MACDStrategy
from ROC2 import ROCStrategy
from VTS import VTSAnalyzer
from RSI2 import RSIAnalysis
from backtest import Backtester

# File path for 'cached_assets' script runs to reset every 4 hours -> integrate into appb.py
BINANCE_CACHE_FILE = 'binance_assets.csv'
# Signals file to append the generated signals
SIGNALS_FILE = 'signals_file.csv'

def get_asset_list_and_timeframes():
    # this reads the list of assets currently avaliable from csv
    if os.path.exists(BINANCE_CACHE_FILE):
        asset_list =  pd.read_csv(BINANCE_CACHE_FILE)['Asset'].tolist()
        return asset_list, ['4h', '12h', '1d'] # The avaliable time frames for each asset 

    else:
        return [], []

# Initialise streamlit app 
st.title('Backtesting Framework')

# setup sidebar
with st.sidebar:
    st.write('Dashboard')
    # define avaliable indicators
    indicators = ['ATR', 'BB', 'MACD', 'VTS', 'ROC', 'RSI']
    default_indicators = ['ROC']#defalt indicator will atomatically be in the bar when page is loaded up or reset
    selected_indicators = st.multiselect(
        'Select Indicator(s):', 
        options=indicators,
        default=default_indicators
    )
    # use function to collect relevant data for dropdowns
    assets, timeframes = get_asset_list_and_timeframes()
    
    # Error handling - we can report 'no signals to backtest later in teh code'
    if assets:
        selected_asset = st.selectbox('Select Asset:', assets)
    else:
        selected_asset = ''#will be used to pass the indicators selected on
    
    if timeframes:
        selected_timeframe = st.selectbox('Select Timeframe:', timeframes)
    else:
        selected_timeframe = ''

# Function to backtest and show summary statistics
def backtest_all_indicators(symbol, selected_timeframe):
    # Initialize Backtester Class
    backtester = Backtester(
        signals_file=SIGNALS_FILE,
        symbol=symbol,
        interval=selected_timeframe
    )
    
    # Align signals with OHLC data so it can be traced and tested
    backtester.align_signals_with_ohlc()
    
    # Backtest summary
    summary = backtester.generate_backtest_summary(time_deltas={
        5: 1,      # 5 minutes
        15: 3,     # 15 minutes
        60: 12,    # 1 hour
        240: 48,   # 4 hours
        1440: 288, # 1 day
        4320: 864, # 3 days
        10080: 2016 # 7 days
    })
    
    return summary, backtester.ohlc_data

# Function to plot the graph and return image as BytesIO 
### Import from the backtester 
def plot_signals(ohlc_data, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(ohlc_data.index, ohlc_data['close'], label='Price', color='blue', alpha=0.5)

    buy_signals = ohlc_data[ohlc_data['Signal'] == 'Buy']
    sell_signals = ohlc_data[ohlc_data['Signal'] == 'Sell']

    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', alpha=1)
#this will put signals on the graph.
    plt.title(f'{symbol} Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
# this is so we can import it from the backtesting class
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf #returned with the signals plotted on

# Main button to run the analysis
if st.sidebar.button('Go'):
    if selected_asset and selected_timeframe:
        ## Adding labels to the dropdown boxes
        st.write('Running analysis with the following settings:')
        st.write('Indicators:', selected_indicators)
        st.write('Selected Asset:', selected_asset)
        st.write('Selected Time Frame:', selected_timeframe)
        
        #Set the quote asset to usdt befoe the request is sent. 
        symbol = f"{selected_asset}USDT"  
        
        # Clear the signals file only if it exists before starting
        #This code completely removes the fille comment it to ensure that file isnt removed. 
        # if os.path.exists(SIGNALS_FILE):
        #     os.remove(SIGNALS_FILE)
    
    
        if 'ATR' in selected_indicators:
            st.write("Running ATR Optimization...")
            atr_periods = [5, 10, 14, 20, 30]
            multipliers = [1, 1.5, 2, 2.5, 3]
            
            atr_optimizer = ATRSTRAT(
                symbol=symbol,
                interval=selected_timeframe,
                atr_periods=atr_periods,
                multipliers=multipliers
            )
            
            cumulative_returns = atr_optimizer.run_optimization(
                initial_train_size=200, 
                test_size=50,
                save_path=SIGNALS_FILE
            )
            
            if cumulative_returns:
                st.success("ATR Optimization Complete!")
                st.write(f"Best Parameters: {atr_optimizer.best_params}")
                st.write(f"Average Cumulative Returns: {np.mean(cumulative_returns):.4f}")
                atr_signals_df = pd.read_csv(SIGNALS_FILE)
                st.write(f"Number of ATR signals generated: {len(atr_signals_df)}")
                st.write("Sample of ATR signals:")
                st.dataframe(atr_signals_df.head())
            else:
                st.error("No ATR signals generated.")

        if 'BB' in selected_indicators:
            st.write("Running Bollinger Bands Optimization...")
            window_range = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
            num_std_dev_range = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]

            bb_optimizer = BBSTRAT(
                symbol=symbol,
                interval=selected_timeframe,
                window_range=window_range,
                num_std_dev_range=num_std_dev_range
            )
            
            cumulative_returns = bb_optimizer.run_optimization(
                initial_train_size=200, 
                test_size=50,
                save_path=SIGNALS_FILE 
            )
            
            if cumulative_returns:
                st.success("Bollinger Bands Optimization Complete!")
                st.write(f"Best Parameters: {bb_optimizer.best_params}")
                st.write(f"Average Cumulative Returns: {np.mean(cumulative_returns):.4f}")
                bb_signals_df = pd.read_csv(SIGNALS_FILE)
                #append_signals_to_file(bb_signals_df)
            else:
                st.error("No Bollinger Bands signals generated.")

        if 'MACD' in selected_indicators:
            st.write("Running MACD Optimization...")
            short_windows = [8, 10, 12, 14, 16]
            long_windows = [20, 26, 30, 35, 40]
            signal_windows = [9, 12, 15]
            
            macd_optimizer = MACDStrategy(
                symbol=symbol,
                interval=selected_timeframe,
                short_windows=short_windows,
                long_windows=long_windows,
                signal_windows=signal_windows
            )
            macd_optimizer.get_binance_ohlc()
            macd_signals_df = macd_optimizer.run_optimization(SIGNALS_FILE)

            if isinstance(macd_signals_df, pd.DataFrame) and not macd_signals_df.empty:
                st.write(macd_signals_df.head())  # Display a sample of the signals for debugging
                st.write("Best Parameters:", macd_optimizer.best_params)
                #append_signals_to_file(macd_signals_df)
            else:
                st.error("No MACD signals generated.")
        
        if 'ROC' in selected_indicators:
            st.write("Running ROC Optimization...")
            roc_strategy = ROCStrategy(
                symbol=symbol,
                interval=selected_timeframe,
                lookback_days=365
            )
            roc_strategy.get_binance_ohlc()  # Fetch data

            # Define parameter grid for optimization
            param_grid = {
                'roc_period': [10, 14, 20],
                'overbought_threshold': [5, 10],
                'oversold_threshold': [-5, -10]
            }

            # Run the optimization
            roc_strategy.run_optimization(
                initial_train_size=200, 
                test_size=50, 
                param_grid=param_grid,
                save_path=SIGNALS_FILE
            )

            if os.path.exists(SIGNALS_FILE):
                st.write("ROC Signals saved successfully!")
                roc_signals_df = pd.read_csv(SIGNALS_FILE)
            else:
                st.error("No ROC signals generated.")

        if 'VTS' in selected_indicators:
            st.write("Running VTS (Volume Trend Signal) Analysis...")
            vts_analyzer = VTSAnalyzer(symbol=symbol, interval=selected_timeframe)
            vts_analyzer.get_binance_ohlc()  # Fetch data
            symbol, max_return_idx, min_volume = vts_analyzer.find_minimum_volume_of_max_return()

            vts_backtester = Backtester(
                signals_file=SIGNALS_FILE,
                symbol=symbol,
                interval=selected_timeframe,
                min_volume=min_volume,  # Pass the minimum volume value here
                use_vts=True
            )

            vts_backtester.align_signals_with_ohlc()
            vts_summary = vts_backtester.generate_backtest_summary(time_deltas={
                5: 1,      # 5 minutes
                15: 3,     # 15 minutes
                60: 12,    # 1 hour
                240: 48,   # 4 hours
                1440: 288, # 1 day
                4320: 864, # 3 days
                10080: 2016 # 7 days
            })

            st.write("VTS Backtest Summary:")
            st.write(vts_summary)
            
        if 'RSI' in selected_indicators:
            st.write("Running RSI Optimization...")
            rsi_windows = [5, 10, 14, 21]
            overbought_thresholds = [60, 70]
            oversold_thresholds = [30, 40]

            rsi_optimizer = RSIAnalysis(
                symbol=symbol,
                interval=selected_timeframe,
                rsi_windows=rsi_windows,
                overbought_thresholds=overbought_thresholds,
                oversold_thresholds=oversold_thresholds
            )
            
            rsi_optimizer.get_binance_ohlc()

            # Run the optimization
            cumulative_returns = rsi_optimizer.run_optimization(
                initial_train_size=200, 
                test_size=50, 
                save_path= SIGNALS_FILE 
            )
            
            if cumulative_returns:
                st.success("RSI Optimization Complete!")
                st.write("Best Parameters:", rsi_optimizer.best_params)
                st.write(f"Average Cumulative Returns: {np.mean(cumulative_returns):.4f}")
                RSI_signals_df = pd.read_csv(SIGNALS_FILE)
            else:
                st.error("No RSI signals generated.")



        # Backtest all indicators
        if os.path.exists(SIGNALS_FILE):
            try:
                saved_signals = pd.read_csv(SIGNALS_FILE)
                if not saved_signals.empty:
                    st.success(f"Total signals saved: {len(saved_signals)}")
                else:
                    st.warning("Signals file exists but is empty. No signals were generated.")
            except pd.errors.EmptyDataError:
                st.warning("Signals file is empty. No signals were generated.")
        else:
            st.warning("No signals file found. Make sure at least one strategy generated signals.")

        if os.path.exists(SIGNALS_FILE) and os.path.getsize(SIGNALS_FILE) > 0:
            st.write("Running Backtest...")
            backtest_summary, ohlc_data = backtest_all_indicators(symbol, selected_timeframe)
            
            st.write("Backtest Summary:")
            st.dataframe(backtest_summary)  # Display the DataFrame in the Streamlit app
            
            if ohlc_data is not None and not ohlc_data.empty:
                st.write("Plotting signals on price chart...")
                buf = plot_signals(ohlc_data, symbol)
                st.image(buf, use_column_width=True)
        else:
            st.error("No signals to backtest. Please adjust your strategy parameters or try different indicators.")
    else:
        st.error("Please select both an asset and a timeframe.")








