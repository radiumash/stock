from tvDatafeed import TvDatafeed, Interval
import pandas as pd
#import numpy as np
import cufflinks as cf
import matplotlib.dates as mdates
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
from datetime import datetime, timedelta
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# import math
#from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter("ignore")

username = 'shenaal1992@gmail.com'
password = 'Menushiiloveu22'

exchange = "NASDAQ"

# Define Intervals
def interval(i):
  if i == 1:
    interval=Interval.in_1_hour

  elif i == 4:
    interval=Interval.in_4_hour

  elif i == 24:
    interval=Interval.in_daily

  return interval

def bollingerLive(symbol,exchange):
    pass

# Create Bollinger Bands
def bollingerStd(symbol, exchange, moving_avg_size, no_steps_from_final_steps, interval_length, number_of_bars):

    tv = TvDatafeed(username, password)
    index_data = tv.get_hist(symbol=symbol,exchange=exchange,interval=interval(interval_length),n_bars=number_of_bars)

    data = index_data.tail(no_steps_from_final_steps)
    if not data.empty:
        # Calculate Bollinger Bands
        data['_MA'] = data['close'].rolling(window=moving_avg_size).mean()
        data['Upper_Band'] = data['_MA'] + 3 * data['close'].rolling(window=moving_avg_size).std()
        data['Lower_Band'] = data['_MA'] - 3 * data['close'].rolling(window=moving_avg_size).std()

        # Mark values above Upper Bollinger Band as green, and below Lower Bollinger Band as red
        above_upper_band = data[data['close'] > data['Upper_Band']]
        below_lower_band = data[data['close'] < data['Lower_Band']]

        # Create a new DataFrame for scatter plot data
        scatter_data = pd.DataFrame(index=data.index)
        scatter_data['AboveUpperBand'] = above_upper_band['close']
        scatter_data['BelowLowerBand'] = below_lower_band['close']

        # use dark background for plots
        style.use('dark_background')

        # Plotting
        add_bollinger = [
        mpf.make_addplot(data['Upper_Band'], color='b'),
        mpf.make_addplot(data['Lower_Band'], color='b'),
        mpf.make_addplot(scatter_data['AboveUpperBand'], type='scatter', color='g', marker='^'),
        mpf.make_addplot(scatter_data['BelowLowerBand'], type='scatter', color='r', marker='v'),
        ]

        fig, ax = mpf.plot(data, type='candle', addplot=add_bollinger, figratio=(10, 6), volume=True, style='nightclouds', returnfig=True)
        return fig


def bollingerPeriod(symbol, exchange, no_steps_from_final_steps, interval_length, number_of_bars):

    tv = TvDatafeed(username, password)
    index_data = tv.get_hist(symbol=symbol,exchange=exchange,interval=interval(interval_length),n_bars=number_of_bars)
    data = index_data.tail(no_steps_from_final_steps)

    if not data.empty:
        # Calculate Bollinger Bands
        data['20_MA'] = data['close'].rolling(window=20).mean()
        data['Upper_Band'] = data['20_MA'] + 2 * data['close'].rolling(window=20).std()
        data['Lower_Band'] = data['20_MA'] - 2 * data['close'].rolling(window=20).std()

        # Calculate Moving Averages
        data['MA_7'] = data['close'].rolling(window=7).mean()
        data['MA_20'] = data['close'].rolling(window=21).mean()
        data['MA_100'] = data['close'].rolling(window=99).mean()
        data['MA_500'] = data['close'].rolling(window=200).mean()

        # Mark values below all three moving averages as orange, and above all as black
        below_all_moving_averages = data[(data['close'] < data['MA_7']) & (data['close'] < data['MA_20']) & (data['close'] < data['MA_100'])& (data['close'] < data['MA_500'])]
        above_all_moving_averages = data[(data['close'] > data['MA_7']) & (data['close'] > data['MA_20']) & (data['close'] > data['MA_100'])& (data['close'] > data['MA_500'])]

        # Create a new DataFrame for scatter plot data
        scatter_data = pd.DataFrame(index=data.index)
        scatter_data['AboveAllMA'] = above_all_moving_averages['close']
        scatter_data['BelowAllMA'] = below_all_moving_averages['close']

        # Plotting
        add_bollinger = [
            mpf.make_addplot(data['Upper_Band'], color='b'),
            mpf.make_addplot(data['Lower_Band'], color='b'),
            mpf.make_addplot(data['MA_7'], color='orange'),
            mpf.make_addplot(data['MA_20'], color='green'),
            mpf.make_addplot(data['MA_100'], color='red'),
            mpf.make_addplot(data['MA_500'], color='yellow'),
            mpf.make_addplot(scatter_data['AboveAllMA'], type='scatter', color='g', marker='^'),
            mpf.make_addplot(scatter_data['BelowAllMA'], type='scatter', color='r', marker='v'),
        ]

        fig, ax = mpf.plot(data, type='candle', addplot=add_bollinger, figratio=(10, 6), volume=True, style = 'nightclouds', returnfig=True)
        return fig    

def plot_data(symbol, exchange, tail, interval_length, number_of_bars):

    tv = TvDatafeed(username, password)
    index_data = tv.get_hist(symbol=symbol,exchange=exchange,interval=interval(interval_length),n_bars=number_of_bars)
    # Consider the last 100 data for better view.
    df = index_data.tail(tail)
    if not df.empty:
        df = df.reset_index()

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].apply(mdates.date2num)

        df['20_MA'] = df['close'].rolling(window=20).mean()
        df['BB_upper_2'] = df['20_MA'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower_2'] = df['20_MA'] - 2 * df['close'].rolling(window=20).std()

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot candlestick chart
        #candlestick_ohlc(ax, df.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

        # Plot the close_price, bb_upper, and bb_lower
        ax.plot(df.index, df['close'], label='close')
        ax.plot(df.index, df['BB_upper_2'], label='BB_upper_2')
        ax.plot(df.index, df['BB_lower_2'], label='BB_lower_2')

        # Color the points based on the conditions
        for idx in df.index:
            if df.loc[idx, 'close'] < df.loc[idx, 'BB_lower_2']:
                ax.scatter(idx, df.loc[idx, 'close'], color='red')
            elif df.loc[idx, 'close'] > df.loc[idx, 'BB_upper_2']:
                ax.scatter(idx, df.loc[idx, 'close'], color='green')
            else:
                ax.scatter(idx, df.loc[idx, 'close'], color='blue')

        # Add a legend and title
        ax.legend()
        ax.set_title('Close Price with Bollinger Bands')

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)
        plt.show()
        return fig
        

# Class LSTM
class LSTM:
    def __init__(self,df):
        self.df = df
        self.model

    def createModel(self):
        pass


# Layout Width
st. set_page_config(layout="wide")
# Sidebar
st.sidebar.subheader('Filter Parameters')

# Retrieving tickers data
ticker_list = pd.read_csv('symbols.txt', header=None)
values = ticker_list[0].to_list()
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list, index=values.index('AAPL')) # Select ticker symbol
st.sidebar.markdown("""---""")
st.sidebar.subheader('Choose Dates -')

td = datetime.now()
ed = datetime.now() - timedelta(days=3*365)
start_date = st.sidebar.date_input("Start date", ed )
end_date = st.sidebar.date_input("End date", td)
st.sidebar.header('OR')
st.sidebar.subheader('Choose Period -')
period = st.sidebar.selectbox(
    'Intervals:',
    ('Please Choose','1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo', '6mo', '9mo', '1y', '3y', '6y'))

if tickerSymbol:
    tickerData = yf.Ticker(tickerSymbol) # Get ticker data
    # Ticker information
    if 'logo_url' in tickerData.info:
        string_logo = '<img src=%s>' % tickerData.info['logo_url']
        string_logo = ''
        st.markdown(string_logo, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Company Overview", "Founders/Directors", "Statistics", "Contact"])

    with tab1:
        if 'longName' in tickerData.info:
            string_name = tickerData.info['longName']
            st.header('**%s**' % string_name, divider='rainbow')

        if 'longBusinessSummary' in tickerData.info:
            string_summary = tickerData.info['longBusinessSummary']
            st.info(string_summary)

    with tab2:
        st.header("Founders/Directors", divider='rainbow')
        if 'companyOfficers' in tickerData.info:
            i = 0
            col1, col2 = st.columns(2)   
            for officer in tickerData.info['companyOfficers']:
                if i%2 == 0:            
                    col1.subheader(officer["name"])
                    for key,val in officer.items():
                        ex = ["name","maxAge"]           
                        if key not in ex:
                            col1.text(key.title() + ": " + str(val))
                else:
                    col2.subheader(officer["name"])
                    for key,val in officer.items():
                        ex = ["name","maxAge"]           
                        if key not in ex:
                            col2.text(key.title() + ": " + str(val))   
                i+=1                     

            
    with tab3:
        st.header('Statistics', divider='rainbow')
        if tickerData.info:
            i = 0
            col1, col2 = st.columns(2) 
            for key,val in tickerData.info.items():
                ex = ["address1","address2","city","state","zip","country","phone","fax","website","city","address1","companyOfficers","longBusinessSummary","longName"]
                if key not in ex:
                    if i%2 == 0: 
                        col1.text(key.title() + ": " + str(val))
                    else:
                        col2.text(key.title() + ": " + str(val))
                    i+=1

    with tab4:
        st.header('Address', divider='rainbow')
        address = ""
        if 'address1' in tickerData.info:
            address = tickerData.info['address1'] + ',\n' 
        if 'address2' in tickerData.info:
            address = tickerData.info['address2'] + ',\n'         
        if 'city' in tickerData.info:        
            address += tickerData.info['city'] + ',\n' 
        if 'state' in tickerData.info:        
            address += tickerData.info['state'] + ',\n' 
        if 'zip' in tickerData.info:        
            address += tickerData.info['zip'] + ',\n' 
        if 'country' in tickerData.info:        
            address += tickerData.info['country'] + ',\n' 
        if 'phone' in tickerData.info:        
            address += "Phone: " + tickerData.info['phone'] + ',\n' 
        if 'fax' in tickerData.info:        
            address += "Fax:" + tickerData.info['fax'] + ',\n'         
        if 'website' in tickerData.info:        
            address += tickerData.info['website']
        st.text(address)   


    # Ticker data
    tv = TvDatafeed(username, password)
    if period != 'Please Choose':
        if period in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d']:
            tickerDf = tickerData.history(period=period, interval="1m") #get the historical prices for this ticker
            #tickerDf = tv.get_hist(symbol=tickerSymbol,exchange=exchange,interval=interval(4),n_bars=10000)
        elif period in ['5d', '1wk']:
            tickerDf = tickerData.history(period=period, interval="1h") #get the historical prices for this ticker
            #tickerDf = tv.get_hist(symbol=tickerSymbol,exchange=exchange,interval=interval(4),n_bars=10000)
        else:
            tickerDf = tickerData.history(period=period, interval="1d") #get the historical prices for this ticker
            #tickerDf = tv.get_hist(symbol=tickerSymbol,exchange=exchange,interval=interval(4),n_bars=10000)
    else:
        tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker
        #tickerDf = tv.get_hist(symbol=tickerSymbol,exchange=exchange,interval=interval(1),n_bars=10000)

    if not tickerDf.empty:
        st.header('Ticker data', divider='rainbow')
        st.dataframe(tickerDf,use_container_width=True)

        # Bollinger bands
        st.header('Bollinger Bands', divider='rainbow')
        tab00, tab11, tab12 = st.tabs(["Live Stream", "STD 2", "STD 3"])
        with tab00:
            pass

        with tab11:
            # fig = bollingerStd(tickerSymbol,exchange,20,1000,24,10000)
            # st.pyplot(fig)
            qf = cf.QuantFig(tickerDf,title='STD-2',legend='top',name='GS',up_color='green', down_color='red')
            qf.add_bollinger_bands(periods=20, boll_std=2, colors=['cyan','grey'], fill=True)
            qf.add_volume(name='Volume',up_color='green', down_color='red')
            fig = qf.iplot(asFigure=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab12:
            qf = cf.QuantFig(tickerDf,title='STD-3',legend='top',name='GS',up_color='green', down_color='red')
            qf.add_bollinger_bands(periods=30, boll_std=3, colors=['cyan','grey'], fill=True)
            qf.add_volume(name='Volume',up_color='green', down_color='red')
            fig = qf.iplot(asFigure=True)
            st.plotly_chart(fig, use_container_width=True)


        # Bollinger bands
        st.header('Models, Accuracy and Prediction', divider='rainbow')
        #tab21, tab22, tab23, tab24 = st.tabs(["LSTM", "Chart 1", "Chart 2", "Chart 3"])
        tab21, tab22, tab23 = st.tabs(["LSTM", "Accuracy", "Prediction"])
        with tab21:
            pass

    else:
        st.write('Unable to find!')
    ####
    # st.write('---')
    # st.write(tickerData.info)
else:
    st.write('Please select ticker symbol from the dropdown!')