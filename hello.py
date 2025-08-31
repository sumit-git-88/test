import talib as ta         # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/doc_index.md
import pandas_ta as pta
#import vectorbt as vbt
import io
import sys
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import streamlit as st
import requests
from tradingview_screener import Query, Column
from tradingview_ta import *
from tradingview_scraper.symbols.ideas import Ideas
from tti.indicators import *
from tradingview_scraper.symbols.news import NewsScraper
from tradingview_scraper.symbols.technicals import Indicators
from nsepythonserver import *
import opstrat as op
import autots
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from prophet import Prophet
from orbit.models.dlt import DLT
from orbit.models.ktr import KTR
from orbit.models.ets import ETS
from bs4 import BeautifulSoup
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover
# import tensorflow
import torch




print("hello world")


# streamlit

st.set_page_config(page_title="Daily / Weekly / Monthly", layout="wide")
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Ticker", "SBIN.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-08-15"))
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
button_clicked = st.sidebar.button("Plot")

value = 30



# st.sidebar.header("Indicators")
# show_jma = st.sidebar.checkbox("JMA (7)", value=True)

# st.sidebar.header("Momentum Indicators")
# show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
# show_macd = st.sidebar.checkbox("MACD", value=True)

# indicator
def value_chart(openPrice, highPrice, lowPrice, closePrice, period):
  float_axis = ((highPrice + lowPrice) / 2).rolling(window=period).mean()
  vol_unit = (highPrice - lowPrice).rolling(window=period).mean() * 0.2
  value_chart_high = pd.Series((highPrice - float_axis) / vol_unit, name="VC_high")
  value_chart_low = pd.Series((lowPrice - float_axis) / vol_unit, name="VC_low")
  value_chart_close = pd.Series((closePrice - float_axis) / vol_unit, name="VC_close")
  value_chart_open = pd.Series((openPrice - float_axis) / vol_unit, name="VC_open")
  return pd.concat([value_chart_high, value_chart_low, value_chart_close, value_chart_open], axis=1)


# data

#ticker = "SBIN.NS"
#interval= "1D"

ydf = yf.download(ticker, period="max", interval=interval, start=start_date, end=end_date)
if ydf.empty:
    st.error("No data found. Check ticker or interval.")
    st.stop()

ydf.columns=[multicols[0] for multicols in ydf.columns]
ydf['JMA_7'] = pta.jma(ydf.Close, length=7)


c = np.random.randn(100)
for i in [5, 8, 13, 21]:
    ydf['ema_' +str(i)] = ta.EMA(ydf['Close'].values, timeperiod=i)
ydf['rsi'] = ta.RSI(ydf['Close'].values, timeperiod=14)
ydf['atr_14'] = ta.ATR(ydf['High'].values, ydf['Low'].values, ydf['Close'].values, timeperiod=14)
ydf['macd'], ydf['macdsignal'], ydf['macdhist'] = ta.MACD(ydf['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)

VC = value_chart(ydf.Open, ydf.High, ydf.Low, ydf.Close, period=5)
ydf[VC.columns] = VC
print(ydf)

ydf = ydf.tail(50)


# plot

fig = go.Figure()
#fig = make_subplots(rows = 2,cols = 2)
fig = make_subplots(
    rows=2, cols=3,
    specs=[
            [{"colspan": 3}, None, None],   # First row: one full-width subplot
            [{}, {}, {}]                  # Second row: two separate subplots (RSI, MACD)
    ],
    subplot_titles=(f"{ticker} with Momentum", "RSI", "MACD", "Bar")
)

def bar_colors(column_name) :
    bar_colors = ['green' if val >= 0 else 'red' for val in column_name]
    return bar_colors

macd_bar_colors = bar_colors(ydf['macdhist'])



candle = go.Candlestick(x=ydf.index, open=ydf['Open'], high=ydf['High'], low=ydf['Low'], close=ydf['Close'], name='Candlestick')

JMA_7 = go.Scatter(x=ydf.index, y=ydf['JMA_7'], name='JMA', line=dict(color='green'))
RSI = go.Scatter(x=ydf.index, y=ydf['rsi'], name='RSI', line=dict(color='red'))
MACD = go.Scatter(x=ydf.index, y=ydf['macd'], name='MACD', line=dict(color='green'))
MACD_signal = go.Scatter(x=ydf.index, y=ydf['macdsignal'], name='MACD Signal', line=dict(color='orange'))

macd_hist = go.Bar(x=ydf.index, y=ydf['macdhist'], marker_color=macd_bar_colors)

fig.add_trace(candle, row = 1,col = 1)
fig.add_trace(JMA_7, row = 1,col = 1)
fig.add_trace(RSI, row = 2,col = 1)
fig.add_trace(MACD, row = 2,col = 2)
fig.add_trace(MACD_signal, row = 2,col = 2)
fig.add_trace(macd_hist, row = 2,col = 3)

fig.update_layout(height=700, showlegend=False, xaxis_rangeslider_visible=False, template='plotly_dark', 
                  plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
#fig.show()

# mplfinance

my_style = mpf.make_mpf_style(
    base_mpf_style='yahoo',         # Start from an existing style
    facecolor='black',              # Chart background
    edgecolor='black',              # Outer edge
    gridcolor='gray',               # Grid lines
    figcolor='black',               # Figure background
    rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
)
candle_mpf_plot, axlist  = mpf.plot(ydf, type='candle', style=my_style, title='Candle', 
                                    addplot=[
                                                    mpf.make_addplot(ydf['ema_5'], color='green', width=1),
                                                    mpf.make_addplot(ydf['ema_13'], color='orange', width=1),
                                                    mpf.make_addplot(ydf['ema_21'], color='blue', width=1)
                                                ],
                                    figratio=(10,4),figscale=0.5, returnfig=True)

ohlc_mpf_plot, axlist  = mpf.plot(ydf, type='ohlc', style=my_style, title='Bar', 
                                  addplot=[
                                                    mpf.make_addplot(ydf['ema_5'], color='green', width=1),
                                                    mpf.make_addplot(ydf['ema_13'], color='orange', width=1),
                                                    mpf.make_addplot(ydf['ema_21'], color='blue', width=1)
                                                ],
                                    figratio=(10,4),figscale=0.5, returnfig=True)

ydf_vc = ydf[['VC_open','VC_high','VC_low','VC_close']]
ydf_vc.columns = ['Open', 'High', 'Low', 'Close']
vc_mpf_plot, axlist = mpf.plot(ydf_vc, type='ohlc', style=my_style, title='Value Chart', 
                               addplot=[
                                                    mpf.make_addplot(ydf['ema_5'], color='green', width=1),
                                                    mpf.make_addplot(ydf['ema_13'], color='orange', width=1),
                                                    mpf.make_addplot(ydf['ema_21'], color='blue', width=1)
                                                ],
                               figratio=(10,5),figscale=0.5,  returnfig=True)

bricks_14 = round(ydf["atr_14"].iloc[-1], 0)
renko14_mpf_plot, axlist  = mpf.plot(ydf, type='renko', renko_params=dict(brick_size=bricks_14, atr_length=14), style=my_style, 
                                    title='Renko ATR 14', figratio=(10,4),figscale=0.5, returnfig=True)



# streamlit
tab_chart, tab_momentum, tab_df, volatality, tab4, tab5 = st.tabs(["Chart", "Momentum", "DF", "Volatitly", "Histogram",
                                        "Portfolio Plot"])
if button_clicked :

    with tab_chart:        
        # st.pyplot(candle_mpf_plot, use_container_width=True)
        # st.pyplot(ohlc_mpf_plot, use_container_width=True)
        # st.pyplot(vc_mpf_plot, use_container_width=True)
        # st.pyplot(renko7_mpf_plot, use_container_width=True)
        # st.pyplot(renko14_mpf_plot, use_container_width=True)

        # First row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Candlestick**")
            st.pyplot(candle_mpf_plot, use_container_width=True)
        with col2:
            st.markdown("**Bar Chart**")
            st.pyplot(ohlc_mpf_plot, use_container_width=True)

        # Second row
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Value Chart**")
            st.pyplot(vc_mpf_plot, use_container_width=True)
        with col4:
            st.markdown("**Renko (ATR 14 Brick Size)**")
            st.pyplot(renko14_mpf_plot, use_container_width=True)



    with tab_momentum:
        st.plotly_chart(fig, use_container_width=True)

    with tab_df:   

        #st.write(f"value is : {value}")
        st.code(f"value is : {value}", language="text")      
        
        # st.subheader("DataFrame 1")
        # st.dataframe(ydf)

        # st.subheader("DataFrame 2")
        # st.dataframe(ydf)                      

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DataFrame1**")
            st.dataframe(ydf)
        with col2:
            st.markdown("**DataFrame2**")
            st.dataframe(ydf)

    
#print(rsi)
#print(help(pta.squeeze_pro))



