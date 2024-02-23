import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ten_tickers_in_nse = ['SBIN.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', '^NSEI'] 
nse_non_index = ['ASTRAL.NS', 'CROMPTON.NS', 'DIVISLAB.NS', 'TATACOMM.NS', 'LTIM.NS', 'GAIL.NS', 'LTTS.NS', 'BLUEDART.NS', 'ESCORTS.NS', 'GLENMARK.NS']
bse_non_index = ['BEL.BO', 'BLUEDART.BO', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'GAIL.BO', 'LTTS.BO', 'ASTRAL.BO', 'CROMPTON.BO', 'DIVISLAB.BO'] 
ten_tickers_in_bse = ['SBIN.BO', 'TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'AXISBANK.BO', 'BAJFINANCE.BO', '^BSESN']

nseData = pd.read_csv('nsedata1.csv', parse_dates=['Date'])
nseData.set_index('Date', inplace=True)
bseData = pd.read_csv('bsedata1.csv', parse_dates=['Date'])
bseData.set_index('Date', inplace=True)

def plot_ticker(ticker, data, title):
    f, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 5))
    f.suptitle(ticker + ' - ' + title)
    axarr[0].plot(data[ticker].resample('D').mean())
    axarr[0].set_title('Daily')
    axarr[1].plot(data[ticker].resample('W').mean())
    axarr[1].set_title('Weekly')
    axarr[2].plot(data[ticker].resample('M').mean())
    axarr[2].set_title('Monthly')
    plt.savefig(f'./fig/1/{ticker}.png')
    plt.close()

def plot_tickers(tickers, data, title):
    for ticker in tickers:
        plot_ticker(ticker, data, title)

plot_tickers(ten_tickers_in_nse, nseData, 'NSE-NIFTY')
plot_tickers(nse_non_index, nseData, 'NSE')
plot_tickers(ten_tickers_in_bse, bseData, 'BSE-SENSEX')
plot_tickers(bse_non_index, bseData, 'BSE')