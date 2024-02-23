from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

ten_tickers_in_nse = ['SBIN.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', '^NSEI'] 
nse_non_index = ['ASTRAL.NS', 'CROMPTON.NS', 'DIVISLAB.NS', 'TATACOMM.NS', 'LTIM.NS', 'GAIL.NS', 'LTTS.NS', 'BLUEDART.NS', 'ESCORTS.NS', 'GLENMARK.NS']
bse_non_index = ['BEL.BO', 'BLUEDART.BO', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'GAIL.BO', 'LTTS.BO', 'ASTRAL.BO', 'CROMPTON.BO', 'DIVISLAB.BO'] 
ten_tickers_in_bse = ['SBIN.BO', 'TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'AXISBANK.BO', 'BAJFINANCE.BO', '^BSESN']

nseData = pd.read_csv('nsedata1.csv', parse_dates=['Date'])
nseData.set_index('Date', inplace=True)
bseData = pd.read_csv('bsedata1.csv', parse_dates=['Date'])
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

def plot_data(data,tickers, title):
    X = np.linspace(-6, 6, 100)
    Y = normal_func(X)
    for ticker in tickers:
        fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        fig.suptitle(f'{title}: {ticker} - Histogram', fontsize=12)
      
        ReturnDaily = np.log(data[ticker]/data[ticker].shift(1))[1:]
        ReturnMean = ReturnDaily.mean()
        ReturnStd = ReturnDaily.std()
        ReturnDailyNormalized = (ReturnDaily - ReturnMean) / ReturnStd
        axarr[0].hist(ReturnDailyNormalized, bins = 20, rwidth = 0.85, density=True)
        axarr[0].plot(X, Y)
        axarr[0].set_title('Daily')
       

        ReturnWeekly = np.log(data[ticker].resample('W').last()/data[ticker].resample('W').last().shift(1))[1:]
        ReturnMean = ReturnWeekly.mean()
        ReturnStd = ReturnWeekly.std()
        ReturnWeeklyNormalized = (ReturnWeekly - ReturnMean) / ReturnStd
        axarr[1].hist(ReturnWeeklyNormalized, bins = 20, rwidth = 0.85, density=True)
        axarr[1].plot(X, Y)
        axarr[1].set_title('Weekly')
       


        ReturnMonthly = np.log(data[ticker].resample('M').last()[1:]/data[ticker].resample('M').last().shift(1))[1:]
        ReturnMean = ReturnMonthly.mean()
        ReturnStd = ReturnMonthly.std()
        ReturnMonthlyNormalized = (ReturnMonthly - ReturnMean) / ReturnStd
        axarr[2].hist(ReturnMonthlyNormalized, bins = 20, rwidth = 0.85, density=True)
        axarr[2].plot(X, Y)
        axarr[2].set_title('Monthly')
       
        plt.savefig(f'./fig/3/{ticker}_hist.png')
        plt.close()
        
        figB, axarrB = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        figB.suptitle(f'{title}: {ticker} - BoxPlot', fontsize=12)
        axarrB[0].boxplot(ReturnDailyNormalized)
        axarrB[0].set_title('Daily')
        axarrB[2].boxplot(ReturnMonthlyNormalized)
        axarrB[2].set_title('Monthly')
        axarrB[1].boxplot(ReturnWeeklyNormalized)
        axarrB[1].set_title('Weekly')

        plt.savefig(f'./fig/3/{ticker}_box.png')
        plt.close()

        figQ, axarrQ = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        figQ.suptitle(f'{title}: {ticker} - QQPlot', fontsize=12)
        stats.probplot(ReturnWeeklyNormalized, plot=axarrQ[1])
        axarrQ[1].set_title('Weekly')
        stats.probplot(ReturnMonthlyNormalized, plot=axarrQ[2])
        axarrQ[2].set_title('Monthly')
        stats.probplot(ReturnDailyNormalized, plot=axarrQ[0])
        axarrQ[0].set_title('Daily')

        plt.savefig(f'./fig/3/{ticker}_qq.png')
        plt.close()

plot_data(nseData, ten_tickers_in_nse, 'NSE_NIFTY')
plot_data(nseData, nse_non_index, 'NSE')
plot_data(bseData, ten_tickers_in_bse, 'BSE_SENSEX')
plot_data(bseData, bse_non_index, 'BSE')

