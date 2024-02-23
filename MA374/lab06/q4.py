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


def get_data(tickers, folder_name):
    data = pd.DataFrame()
    for ticker in tickers:
        df = pd.read_csv(folder_name + '/' + ticker + '.csv', parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        df.rename(columns={'Close': ticker}, inplace=True)
        df = df[[ticker]]
        if data.empty:
            data = df
        else:
            data = data.join(df, how='outer')
    return data

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

def plot_data(completeData, data,tickers, title):
    for ticker in tickers:
       
        ReturnDaily = np.log(data[ticker]/data[ticker].shift(1))[1:]
        ReturnMean = ReturnDaily.mean()
        ReturnStd = ReturnDaily.std()
        ReturnDailyNormalised = (ReturnDaily - ReturnMean)/ReturnStd

        tickerModel = GBM(ReturnMean, ReturnStd, data[ticker][-1], 365, 365)
        time = pd.date_range("2022-01-01", periods=365, freq="D")

        plt.figure(figsize=(10, 5))
        plt.plot(completeData[ticker][:-1], label='Actual')
        plt.plot(time, tickerModel, label='Model')
        plt.legend()
        plt.title(title + ' ' + ticker + ' Daily')
        plt.savefig(f'./fig/4/{ticker}.png')
        plt.close()




plot_data(completeNse, nseData, ten_tickers_in_nse, 'NSE_NIFTY')
plot_data(completeNse, nseData, nse_non_index, 'NSE')
plot_data(completeBse, bseData, ten_tickers_in_bse, 'BSE_SENSEX')
plot_data(completeBse, bseData, bse_non_index, 'BSE')

