import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.graph_objs as go
import streamlit as st

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
from keras.models import load_model
import warnings
warnings.simplefilter("ignore")

exchange = "NASDAQ"

def create_dataset(dataset, time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(dataset[i + time_step,0])
    return np.array(dataX),np.array(dataY)

def bollingerLive(tickerData, moving_avg_size, period, interval_length):

    data = tickerData.history(period=period,interval=interval_length)
    data['Middle Band'] = data['Close'].rolling(window=moving_avg_size).mean()
    data['Upper Band'] = data['Middle Band'] + 1.96 * data['Close'].rolling(window=moving_avg_size).std()
    data['Lower Band'] = data['Middle Band'] - 1.96 * data['Close'].rolling(window=moving_avg_size).std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Middle Band'], line=dict(color='gray', width=.7), name='Middle Band'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper Band'], line=dict(color='red', width=1.5), name='Upper Band(Sell)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower Band'], line=dict(color='green', width=1.5), name='Lower Band(Buy)'))

    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Market Data'))
    if 'longName' in tickerData.info:
        string_name = tickerData.info['longName']

    fig.update_layout(
        title=string_name + ' Live Share Price',
        yaxis_title = 'Stock Price')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label='15m', step='minute', stepmode='backward'),
                dict(count=45, label='45m', step='minute', stepmode='backward'),
                dict(count=1, label='HTD', step='hour', stepmode='todate'),
                dict(count=3, label='3h', step='hour', stepmode='backward'),
                dict(step='all')
            ])
        )
    )

    return fig


# Create Bollinger Bands
def bollingerStd(data, moving_avg_size):

    data['Middle Band'] = data['Close'].rolling(window=moving_avg_size).mean()
    data['Upper Band'] = data['Middle Band'] + 1.96 * data['Close'].rolling(window=moving_avg_size).std()
    data['Lower Band'] = data['Middle Band'] - 1.96 * data['Close'].rolling(window=moving_avg_size).std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Middle Band'], line=dict(color='gray', width=.7), name='Middle Band'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper Band'], line=dict(color='red', width=1.5), name='Upper Band(Sell)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower Band'], line=dict(color='green', width=1.5), name='Lower Band(Buy)'))

    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Market Data'))
    if 'longName' in tickerData.info:
        string_name = tickerData.info['longName']

    fig.update_layout(
        title=string_name + ' Live Share Price',
        yaxis_title = 'Stock Price')

    return fig

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
        tab11, tab12 = st.tabs(["Live Stream", "Periodic"])

        with tab11:
            fig = bollingerLive(tickerData,21,"1d","1m")
            st.plotly_chart(fig, use_container_width=True)

        with tab12:
            fig = bollingerStd(tickerDf,21)
            st.plotly_chart(fig, use_container_width=True)


        # Bollinger bands
        st.header('Models and Prediction', divider='rainbow')
        #tab21, tab22, tab23, tab24 = st.tabs(["LSTM", "Chart 1", "Chart 2", "Chart 3"])
        model1 = load_model('LSTM_model.keras')
        model2 = load_model('Bi_LSTM_model.keras')

        y = tickerDf['Close'].fillna(method='ffill')
        y = y.values.reshape(-1, 1)

        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)

        # generate the input and output sequences
        n_lookback = 7  # length of input sequences (lookback period)
        n_forecast = 7  # length of output sequences (forecast period)

        X = []
        Y = []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        X = np.array(X)
        Y = np.array(Y)
        # generate the forecasts
        X_ = y[- n_lookback:]  # last available input sequence
        X_ = X_.reshape(1, n_lookback, 1)

        Y_ = model1.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)
        st.header('LSTM')
        st.write("Stock Price Prediction:")
        st.title(Y_[0][0])   

        Y_ = model2.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)     
        st.header('Bi-LSTM')
        st.write("Stock Price Prediction:")
        st.title(Y_[0][0]) 

    else:
        st.write('Unable to find!')
    ####
    # st.write('---')
    # st.write(tickerData.info)
else:
    st.write('Please select ticker symbol from the dropdown!')