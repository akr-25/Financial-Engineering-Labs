import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

ten_tickers_in_nse = ['SBIN.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', '^NSEI'] 
nse_non_index = ['ASTRAL.NS', 'CROMPTON.NS', 'DIVISLAB.NS', 'TATACOMM.NS', 'LTIM.NS', 'GAIL.NS', 'LTTS.NS', 'BLUEDART.NS', 'ESCORTS.NS', 'GLENMARK.NS']
bse_non_index = ['BEL.BO', 'BLUEDART.BO', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'GAIL.BO', 'LTTS.BO', 'ASTRAL.BO', 'CROMPTON.BO', 'DIVISLAB.BO'] 
ten_tickers_in_bse = ['SBIN.BO', 'TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'AXISBANK.BO', 'BAJFINANCE.BO', '^BSESN']

nseData = pd.read_csv('nsedata1.csv', parse_dates=['Date'], dayfirst=True)
nseData.set_index('Date', inplace=True)
bseData = pd.read_csv('bsedata1.csv', parse_dates=['Date'], dayfirst=True)
bseData.set_index('Date', inplace=True)

def normal_func(x):
    return stats.norm.pdf(x, 0, 1)

def GBM(mu, sigma, S0, T, N):
    dt = T/(N-1)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) 
    X = mu*t + sigma*W 
    S = S0*np.exp(X) 
    return S

def GBM_with_X(mu, sigma, S0, T, N, X, noise=0.9):
    dt = T/(N-1)
    t = np.linspace(0, T, N)
    W = X + np.random.standard_normal(size = N)*noise
    W = np.cumsum(W)*np.sqrt(dt) 
    X = mu*t + sigma*W
    S = S0*np.exp(X) 
    return S

completeNse = nseData
nseData = nseData[:-248]
completeBse = bseData
bseData = bseData[:-248]

def plot_data(completeData, data, tickers, title):
    for ticker in tickers:

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
       
        ReturnWeekly = np.log(data[ticker].resample('W').last()/data[ticker].resample('W').last().shift(1))[1:]/7
        ReturnMean = ReturnWeekly.mean()
        ReturnStd = ReturnWeekly.std()
        ReturnWeeklyNormalised = (ReturnWeekly - ReturnMean)/ReturnStd

        tickerModel = GBM(ReturnMean, ReturnStd, data[ticker][-1], 365, 365//7)
        time = pd.date_range("2022-01-01", periods=365//7, freq="W")

        ax[0].plot(time, tickerModel, label='Model')
        ax[0].plot(completeData[ticker].resample('W').last()[:-1], label='Actual')
        ax[0].legend()
        ax[0].set_title(title + ' -Daily- ' + ticker + ' -Weekly')

        ReturnMonthly = np.log(data[ticker].resample('M').last()/data[ticker].resample('M').last().shift(1))[1:]/30
        ReturnMean = ReturnMonthly.mean()
        ReturnStd = ReturnMonthly.std()
        ReturnMonthlyNormalised = (ReturnMonthly - ReturnMean)/ReturnStd

        tickerModel = GBM(ReturnMean, ReturnStd, data[ticker][-1], 365, 365//30)
        time = pd.date_range("2022-01-01", periods=365//30, freq="M")

        ax[1].plot(time, tickerModel, label='Model')
        ax[1].plot(completeData[ticker].resample('M').last()[:-1], label='Actual')
        ax[1].legend()
        ax[1].set_title(title + ' ' + ticker + ' -Monthly')

        plt.savefig(f'./fig/5/{ticker}.png')
        plt.close()


plot_data(completeNse, nseData, ten_tickers_in_nse, 'NSE_NIFTY')
plot_data(completeNse, nseData, nse_non_index, 'NSE')
plot_data(completeBse, bseData, ten_tickers_in_bse, 'BSE_SENSEX')
plot_data(completeBse, bseData, bse_non_index, 'BSE')

