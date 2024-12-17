import streamlit as st  # Importing Streamlit for building the web application interface
import os  # Importing os for interacting with the operating system
import numpy as np  # Importing NumPy for numerical operations
import pandas as pd  # Importing Pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting graphs
from io import BytesIO  # Importing BytesIO for handling byte streams
from BB2 import BBSTRAT  # Importing Bollinger Bands strategy class
from ATR2 import ATRSTRAT  # Importing Average True Range strategy class
from MACD2 import MACDStrategy  # Importing MACD strategy class
from ROC2 import ROCStrategy  # Importing Rate of Change strategy class
from VTS import VTSAnalyzer  # Importing Volume Trend Signal analysis class
from RSI2 import RSIAnalysis  # Importing Relative Strength Index analysis class
from backtest import Backtester  # Importing Backtester class for backtesting strategies

# File path for cached Binance asset data
BINANCE_CACHE_FILE = 'binance_assets.csv'
# File path for storing generated trading signals
SIGNALS_FILE = 'signals_file.csv'

def get_asset_list_and_timeframes():
    """
    Retrieves the list of assets available for trading from a CSV file.
    Returns:
        asset_list (list): List of assets available for trading.
        timeframes (list): List of predefined timeframes for analysis.
    """
    if os.path.exists(BINANCE_CACHE_FILE):  # Check if the cache file exists
        asset_list = pd.read_csv(BINANCE_CACHE_FILE)['Asset'].tolist()  # Read asset list from CSV
        return asset_list, ['4h', '12h', '1d']  # Return asset list and available timeframes
    else:
        return [], []  # Return empty lists if the cache file does not exist

st.title('Backtesting Framework')  # Set the title of the Streamlit app

with st.sidebar:  # Create a sidebar for user inputs
    st.write('Dashboard')  # Sidebar title
    indicators = ['ATR', 'BB', 'MACD', 'VTS', 'ROC', 'RSI']  # List of available indicators
    default_indicators = ['ROC']  # Default selected indicator
    selected_indicators = st.multiselect(  # Allow user to select multiple indicators
        'Select Indicator(s):', 
        options=indicators,
        default=default_indicators
    )
    assets, timeframes = get_asset_list_and_timeframes()  # Get available assets and timeframes
    
    if assets:  # Check if assets are available
        selected_asset = st.selectbox('Select Asset:', assets)  # Dropdown for asset selection
    else:
        selected_asset = ''  # Default to empty if no assets
    
    if timeframes:  # Check if timeframes are available
        selected_timeframe = st.selectbox('Select Timeframe:', timeframes)  # Dropdown for timeframe selection
    else:
        selected_timeframe = ''  # Default to empty if no timeframes

def backtest_all_indicators(symbol, selected_timeframe):
    """
    Backtests all selected indicators for a given symbol and timeframe.
    Args:
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        selected_timeframe (str): The timeframe for the backtest (e.g., '1d').
    Returns:
        summary (DataFrame): Summary of the backtest results.
        ohlc_data (DataFrame): OHLC data used for the backtest.
    """
    backtester = Backtester(  # Initialize the Backtester with parameters
        signals_file=SIGNALS_FILE,
        symbol=symbol,
        interval=selected_timeframe
    )
    backtester.align_signals_with_ohlc()  # Align signals with OHLC data
    summary = backtester.generate_backtest_summary(time_deltas={  # Generate backtest summary
        5: 1,  # 5 minutes
        15: 3,  # 15 minutes
        60: 12,  # 1 hour
        240: 48,  # 4 hours
        1440: 288,  # 1 day
        4320: 864,  # 3 days
        10080: 2016  # 7 days
    })
    return summary, backtester.ohlc_data  # Return summary and OHLC data

def plot_signals(ohlc_data, symbol):
    """
    Plots the price data along with buy/sell signals.
    Args:
        ohlc_data (DataFrame): OHLC data containing price and signals.
        symbol (str): The trading symbol for the title of the plot.
    Returns:
        buf (BytesIO): Byte stream of the generated plot image.
    """
    plt.figure(figsize=(14, 7))  # Set the figure size for the plot
    plt.plot(ohlc_data.index, ohlc_data['close'], label='Price', color='blue', alpha=0.5)  # Plot closing price
    buy_signals = ohlc_data[ohlc_data['Signal'] == 'Buy']  # Filter buy signals
    sell_signals = ohlc_data[ohlc_data['Signal'] == 'Sell']  # Filter sell signals
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', alpha=1)  # Plot buy signals
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', alpha=1)  # Plot sell signals
    plt.title(f'{symbol} Price with Buy/Sell Signals')  # Set the title of the plot
    plt.xlabel('Date')  # Set x-axis label
    plt.ylabel('Price')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping
    buf = BytesIO()  # Create a byte stream to save the plot
    plt.savefig(buf, format='png')  # Save the plot to the byte stream
    buf.seek(0)  # Move to the beginning of the byte stream
    return buf  # Return the byte stream

if st.sidebar.button('Go'):  # Button to start the analysis
    if selected_asset and selected_timeframe:  # Check if both asset and timeframe are selected
        st.write('### Running analysis with the following settings:')  # Display analysis settings
        st.write(f"**Indicators:** {', '.join(selected_indicators)}")  # Show selected indicators
        st.write(f"**Selected Asset:** {selected_asset}")  # Show selected asset
        st.write(f"**Selected Time Frame:** {selected_timeframe}")  # Show selected timeframe
        
        symbol = f"{selected_asset}USDT"  # Construct the trading symbol
        
        if os.path.exists(SIGNALS_FILE):  # Check if signals file exists
            os.remove(SIGNALS_FILE)  # Remove existing signals file

        for indicator in selected_indicators:  # Loop through selected indicators
            with st.status(f"Running {indicator} Optimization...") as status:  # Show status message
                if indicator == 'ATR':  # Check if the indicator is ATR
                    atr_periods = [5, 10, 14, 20, 30]  # Define ATR periods
                    multipliers = [1, 1.5, 2, 2.5, 3]  # Define multipliers for ATR
                    atr_optimizer = ATRSTRAT(  # Initialize ATR strategy
                        symbol=symbol,
                        interval=selected_timeframe,
                        atr_periods=atr_periods,
                        multipliers=multipliers
                    )
                    cumulative_returns = atr_optimizer.run_optimization(  # Run optimization
                        initial_train_size=200, 
                        test_size=50,
                        save_path=SIGNALS_FILE
                    )
                    if cumulative_returns:  # Check if cumulative returns were generated
                        status.update(label="ATR Optimization Complete!", state="complete")  # Update status
                        atr_signals_df = pd.read_csv(SIGNALS_FILE)  # Read generated signals
                        st.write(f"Number of ATR signals generated: {len(atr_signals_df)}")  # Display number of signals
                        st.dataframe(atr_signals_df.head())  # Show a sample of the signals
                    else:
                        status.update(label="No ATR signals generated.", state="error")  # Update status if no signals

                elif indicator == 'BB':  # Check if the indicator is Bollinger Bands
                    window_range = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]  # Define window range
                    num_std_dev_range = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]  # Define std dev range
                    bb_optimizer = BBSTRAT(  # Initialize Bollinger Bands strategy
                        symbol=symbol,
                        interval=selected_timeframe,
                        window_range=window_range,
                        num_std_dev_range=num_std_dev_range
                    )
                    cumulative_returns = bb_optimizer.run_optimization(  # Run optimization
                        initial_train_size=200, 
                        test_size=50,
                        save_path=SIGNALS_FILE 
                    )
                    if cumulative_returns:  # Check if cumulative returns were generated
                        status.update(label="Bollinger Bands Optimization Complete!", state="complete")  # Update status
                        bb_signals_df = pd.read_csv(SIGNALS_FILE)  # Read generated signals
                        st.write(f"Number of BB signals generated: {len(bb_signals_df)}")  # Display number of signals
                    else:
                        status.update(label="No Bollinger Bands signals generated.", state="error")  # Update status if no signals

                elif indicator == 'MACD':  # Check if the indicator is MACD
                    short_windows = [8, 10, 12, 14, 16]  # Define short windows for MACD
                    long_windows = [20, 26, 30, 35, 40]  # Define long windows for MACD
                    signal_windows = [9, 12, 15]  # Define signal windows for MACD
                    macd_optimizer = MACDStrategy(  # Initialize MACD strategy
                        symbol=symbol,
                        interval=selected_timeframe,
                        short_windows=short_windows,
                        long_windows=long_windows,
                        signal_windows=signal_windows
                    )
                    macd_optimizer.get_binance_ohlc()  # Fetch OHLC data for MACD
                    macd_signals_df = macd_optimizer.run_optimization(SIGNALS_FILE)  # Run optimization
                    if isinstance(macd_signals_df, pd.DataFrame) and not macd_signals_df.empty:  # Check if signals were generated
                        status.update(label="MACD Optimization Complete!", state="complete")  # Update status
                        st.write(macd_signals_df.head())  # Show a sample of the signals
                    else:
                        status.update(label="No MACD signals generated.", state="error")  # Update status if no signals

                elif indicator == 'ROC':  # Check if the indicator is Rate of Change
                    roc_strategy = ROCStrategy(  # Initialize ROC strategy
                        symbol=symbol,
                        interval=selected_timeframe,
                        lookback_days=365  # Set lookback period for ROC
                    )
                    roc_strategy.get_binance_ohlc()  # Fetch data for ROC
                    param_grid = {  # Define parameter grid for optimization
                        'roc_period': [10, 14, 20],
                        'overbought_threshold': [5, 10],
                        'oversold_threshold': [-5, -10]
                    }
                    roc_strategy.run_optimization(  # Run optimization
                        initial_train_size=200, 
                        test_size=50, 
                        param_grid=param_grid,
                        save_path=SIGNALS_FILE
                    )
                    if os.path.exists(SIGNALS_FILE):  # Check if signals were saved
                        st.write("ROC Signals saved successfully!")  # Notify user of successful save
                        roc_signals_df = pd.read_csv(SIGNALS_FILE)  # Read generated signals
                    else:
                        st.error("No ROC signals generated.")  # Notify user if no signals were generated

                elif indicator == 'VTS':  # Check if the indicator is Volume Trend Signal
                    vts_analyzer = VTSAnalyzer(symbol=symbol, interval=selected_timeframe)  # Initialize VTS analyzer
                    vts_analyzer.get_binance_ohlc()  # Fetch data for VTS
                    symbol, max_return_idx, min_volume = vts_analyzer.find_minimum_volume_of_max_return()  # Analyze VTS
                    vts_backtester = Backtester(  # Initialize Backtester for VTS
                        signals_file=SIGNALS_FILE,
                        symbol=symbol,
                        interval=selected_timeframe,
                        min_volume=min_volume,
                        use_vts=True
                    )
                    vts_backtester.align_signals_with_ohlc()  # Align VTS signals with OHLC data
                    vts_summary = vts_backtester.generate_backtest_summary(time_deltas={  # Generate backtest summary
                        5: 1,
                        15: 3,
                        60: 12,
                        240: 48,
                        1440: 288,
                        4320: 864,
                        10080: 2016
                    })
                    st.write("VTS Backtest Summary:")  # Display VTS backtest summary
                    st.write(vts_summary)  # Show the summary

                elif indicator == 'RSI':  # Check if the indicator is RSI
                    rsi_windows = [5, 10, 14, 21]  # Define RSI windows
                    overbought_thresholds = [60, 70]  # Define overbought thresholds
                    oversold_thresholds = [30, 40]  # Define oversold thresholds
                    rsi_optimizer = RSIAnalysis(  # Initialize RSI analysis
                        symbol=symbol,
                        interval=selected_timeframe,
                        rsi_windows=rsi_windows,
                        overbought_thresholds=overbought_thresholds,
                        oversold_thresholds=oversold_thresholds
                    )
                    rsi_optimizer.get_binance_ohlc()  # Fetch data for RSI
                    cumulative_returns = rsi_optimizer.run_optimization(  # Run optimization
                        initial_train_size=200, 
                        test_size=50, 
                        save_path=SIGNALS_FILE 
                    )
                    if cumulative_returns:  # Check if cumulative returns were generated
                        status.update(label="RSI Optimization Complete!", state="complete")  # Update status
                        st.write("Best Parameters:", rsi_optimizer.best_params)  # Show best parameters
                        st.write(f"Average Cumulative Returns: {np.mean(cumulative_returns):.4f}")  # Show average returns
                        RSI_signals_df = pd.read_csv(SIGNALS_FILE)  # Read generated signals
                    else:
                        status.update(label="No RSI signals generated.", state="error")  # Update status if no signals

        # Backtest all indicators
        if os.path.exists(SIGNALS_FILE):  # Check if signals file exists
            try:
                saved_signals = pd.read_csv(SIGNALS_FILE)  # Read saved signals from file
                if not saved_signals.empty:  # Check if signals are available
                    st.success(f"Total signals saved: {len(saved_signals)}")  # Notify user of total signals
                    st.write("Running Backtest...")  # Notify user that backtesting is starting
                    backtest_summary, ohlc_data = backtest_all_indicators(symbol, selected_timeframe)  # Perform backtesting
                    st.write("Backtest Summary:")  # Display the backtest summary
                    st.dataframe(backtest_summary)  # Show the summary DataFrame in the Streamlit app
                    if ohlc_data is not None and not ohlc_data.empty:  # Check if OHLC data is available
                        st.write("Plotting signals on price chart...")  # Notify user that plotting is starting
                        buf = plot_signals(ohlc_data, symbol)  # Generate the plot
                        st.image(buf, use_column_width=True)  # Display the plot in the Streamlit app
                else:
                    st.warning("Signals file exists but is empty. No signals were generated.")  # Warn if empty
            except pd.errors.EmptyDataError:
                st.warning("Signals file is empty. No signals were generated.")  # Warn if file is empty
        else:
            st.warning("No signals file found. Make sure at least one strategy generated signals.")  # Warn if file not found
    else:
        st.error("Please select both an asset and a timeframe.")  # Notify user to select asset and timeframe