import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def find_market_portfolio(filename):
    # Load data from the specified CSV file and set the index
    market_data = pd.read_csv(filename)
    market_data.set_index('Date', inplace=True)

    # Compute the daily returns based on the opening and closing prices
    daily_returns = (market_data['Open'] - market_data['Close']) / market_data['Open']

    # Convert the daily returns to a NumPy array and reshape it to a column vector
    daily_returns = np.array(daily_returns)

    # Convert the daily returns to a Pandas DataFrame
    daily_returns = pd.DataFrame(np.transpose(daily_returns))

    # Compute the market mean and standard deviation based on the daily returns
    market_mean, market_std = np.mean(daily_returns, axis = 0) * len(daily_returns) / 5, daily_returns.std()
    # Extract the market return and risk
    market_return = market_mean[0]
    market_risk = market_std[0]

    # Print the market statistics
    print("Market return = {:.2%}".format(market_return))
    print("Market risk = {:.2%}".format(market_risk))

    # Return the market return and risk
    return market_return, market_risk


def compute_weights(mean_returns, cov_matrix, target_return):
    inverse_cov_matrix = np.linalg.inv(cov_matrix)
    ones_vector = np.ones(len(mean_returns))

    p = [[1, ones_vector @ inverse_cov_matrix @ np.transpose(mean_returns)], [target_return, mean_returns @ inverse_cov_matrix @ np.transpose(mean_returns)]]
    q = [[ones_vector @ inverse_cov_matrix @ np.transpose(ones_vector), 1], [mean_returns @ inverse_cov_matrix @ np.transpose(ones_vector), target_return]]
    r = [[ones_vector @ inverse_cov_matrix @ np.transpose(ones_vector), ones_vector @ inverse_cov_matrix @ np.transpose(mean_returns)], [mean_returns @ inverse_cov_matrix @ np.transpose(ones_vector), mean_returns @ inverse_cov_matrix @ np.transpose(mean_returns)]]

    det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)
    det_p /= det_r
    det_q /= det_r

    w = det_p * (ones_vector @ inverse_cov_matrix) + det_q * (mean_returns @ inverse_cov_matrix)

    return w


def construct_efficient_frontier(returns, cov_matrix, risk_free_rate):
    # Calculate efficient frontier
    num_assets = len(returns)
    ones = np.ones(num_assets)
    risks = []
    for exp_return in returns:
        weights = compute_weights(returns, cov_matrix, exp_return)
        risk = math.sqrt(weights @ cov_matrix @ weights.T)
        risks.append(risk)

    # Calculate minimum variance portfolio
    min_var_weights = ones @ np.linalg.inv(cov_matrix) / (ones @ np.linalg.inv(cov_matrix) @ ones.T)
    min_var_return = min_var_weights @ returns
    min_var_risk = math.sqrt(min_var_weights @ cov_matrix @ min_var_weights.T)

    # Split efficient frontier into two parts (before and after minimum variance point)
    ef_returns1, ef_risks1, ef_returns2, ef_risks2 = [], [], [], []
    for i in range(len(returns)):
        if returns[i] >= min_var_return:
            ef_returns1.append(returns[i])
            ef_risks1.append(risks[i])
        else:
            ef_returns2.append(returns[i])
            ef_risks2.append(risks[i])

    # Calculate market portfolio
    market_weights = (returns - risk_free_rate * ones) @ np.linalg.inv(cov_matrix) / ((returns - risk_free_rate * ones) @ np.linalg.inv(cov_matrix) @ ones.T)
    market_return = market_weights @ returns
    market_risk = math.sqrt(market_weights @ cov_matrix @ market_weights.T)

    # Plot efficient frontier, minimum variance portfolio, and market portfolio
    plt.plot(ef_risks1, ef_returns1, color='yellow', label='Efficient frontier')
    plt.plot(ef_risks2, ef_returns2, color='blue')
    plt.plot(min_var_risk, min_var_return, color='green', marker='o')
    plt.annotate(f'Minimum Variance Portfolio ({round(min_var_risk, 4)}, {round(min_var_return, 4)})', xy=(min_var_risk, min_var_return), xytext=(min_var_risk, -0.6))
    plt.plot(market_risk, market_return, color='green', marker='o')
    plt.annotate(f'Market Portfolio ({round(market_risk, 4)}, {round(market_return, 4)})', xy=(market_risk, market_return), xytext=(0.012, 0.8))
    plt.xlabel('Risk (sigma)')
    plt.ylabel('Returns')
    plt.title('Minimum Variance Curve & Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print market portfolio details
    print('Market Portfolio Weights =', market_weights)
    print('Expected Return =', market_return)
    print('Risk =', round(market_risk * 100, 2), '%')

    return market_return, market_risk

def plot_capital_market_line(M, C, mu_rf, mu_market, risk_market):
    # Define a range of returns
    returns = np.linspace(-2, 5, num=2000)

    # Create an empty list to store the calculated risks
    risks = []

    # Calculate the risk for each return in the range
    for mu in returns:
        weights = compute_weights(M, C, mu)
        risk = math.sqrt(weights @ C @ np.transpose(weights))
        risks.append(risk)

    # Define a range of risks to use for the capital market line
    cml_risks = np.linspace(0, 0.25, num=2000)

    # Calculate the corresponding returns for the capital market line
    cml_returns = []
    for i in cml_risks:
        cml_returns.append(mu_rf + (mu_market - mu_rf) * i / risk_market)

    # Calculate the slope and intercept of the capital market line
    slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf

    # Print the equation of the capital market line
    print("The equation of the Capital Market Line is:")
    print("y = {:.4f} x + {:.4f}\n".format(slope, intercept))

    # Plot the capital market line and the minimum variance line
    plt.plot(risks, returns, label='Minimum Variance Line')
    plt.plot(cml_risks, cml_returns, label='Capital Market Line')
    plt.title('Capital Market Line with Minimum Variance Line')
    plt.xlabel('Risk (sigma)')
    plt.ylabel('Return')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the capital market line by itself
    plt.plot(cml_risks, cml_returns)
    plt.title('Capital Market Line')
    plt.xlabel('Risk (sigma)')
    plt.ylabel('Return')
    plt.grid(True)
    plt.show()

def plot_sml(mu_rf, mu_market):
    # Create a range of beta values
    beta_values = np.linspace(-1, 1, 2000)
    
    # Calculate the corresponding expected return values for each beta value
    expected_returns = mu_rf + (mu_market - mu_rf) * beta_values
    
    # Plot the Security Market Line
    plt.plot(beta_values, expected_returns)
    
    # Print the equation of the Security Market Line
    slope = mu_market - mu_rf
    intercept = mu_rf
    print(f"Equation of Security Market Line is: mu = {slope:.2f}beta + {intercept:.2f}")
    
    # Set plot title, axis labels, and display the plot
    plt.title('Security Market Line')
    plt.xlabel('Beta')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.show()

def beta_model(stocks_name, market_type, mu_market_index, risk_market_index, beta):
  # Collect the daily returns for each stock into a list
  daily_returns_list = []
  for stock_name in stocks_name:
      # Load the CSV file for the given stock
      filename = f'./data/{market_type}/{stock_name}.csv'
      df = pd.read_csv(filename)
      df.set_index('Date', inplace=True)

      # Calculate the daily returns and add them to the list
      daily_returns = df['Open'].pct_change()
      daily_returns_list.append(daily_returns)

  # Convert the list of daily returns to a numpy array and compute the mean and covariance
  daily_returns_array = np.array(daily_returns_list).T
  stock_returns_df = pd.DataFrame(daily_returns_array, columns=stocks_name)
  M = np.mean(stock_returns_df, axis=0) * len(stock_returns_df) / 5
  C = stock_returns_df.cov()
  
  print("\n\nStocks Name\t\t\tActual Return\t\t\tExpected Return\n")
  for i in range(len(M)):
    print("{}\t\t\t{}\t\t{}".format(stocks_name[i], M[i], beta[i] * (mu_market_index - 0.05) + 0.05))

def model(stocks_name, market_type, mu_market_index, risk_market_index):
    # Collect the daily returns for each stock into a list
    daily_returns_list = []
    for stock_name in stocks_name:
        # Load the CSV file for the given stock
        filename = f'./data/{market_type}/{stock_name}.csv'
        df = pd.read_csv(filename)
        df.set_index('Date', inplace=True)

        # Calculate the daily returns and add them to the list
        daily_returns = df['Open'].pct_change()
        daily_returns_list.append(daily_returns)

    # Convert the list of daily returns to a numpy array and compute the mean and covariance
    daily_returns_array = np.array(daily_returns_list).T
    stock_returns_df = pd.DataFrame(daily_returns_array, columns=stocks_name)
    M = np.mean(stock_returns_df, axis=0) * len(stock_returns_df) / 5
    C = stock_returns_df.cov()

    # Compute the efficient frontier and plot the capital market line
    mu_market, risk_market = construct_efficient_frontier(M, C, 0.05)
    plot_capital_market_line(M, C, 0.05, mu_market, risk_market)

    # If the market type is BSE or NSE, plot the security market line with a fixed market return
    if market_type == 'BSE' or market_type == 'NSE':
        plot_sml(0.05, mu_market_index)
    # Otherwise, plot the security market line with the market return obtained from the efficient frontier
    else:
        plot_sml(0.05, mu_market)


def compute_beta(stocks_name, main_filename, index_type):
  df = pd.read_csv(main_filename)
  df.set_index('Date', inplace=True)
  daily_returns = (df['Open'] - df['Close'])/df['Open']

  daily_returns_stocks = []
    
  for i in range(len(stocks_name)):
    if index_type == 'Non-index':
      filename = './data/Non-index stocks/' + stocks_name[i] + '.csv'
    else:
      filename = './data/' + index_type[:3] + '/' + stocks_name[i] + '.csv'
    df_stocks = pd.read_csv(filename)
    df_stocks.set_index('Date', inplace=True)

    daily_returns_stocks.append((df_stocks['Open'] - df_stocks['Close'])/df_stocks['Open'])
    

  beta_values = []
  for i in range(len(stocks_name)):
    df_combined = pd.concat([daily_returns_stocks[i], daily_returns], axis = 1, keys = [stocks_name[i], index_type])
    C = df_combined.cov()

    beta = C[index_type][stocks_name[i]]/C[index_type][index_type]
    beta_values.append(beta)

  return beta_values


def LAB04_repeat():
  print("Market portfolio for BSE using Index ")
  mu_market_BSE, risk_market_BSE = find_market_portfolio('./data/BSE/BSESN.csv')

  print("\n\n********************  Market portfolio for NSE using Index  ********************")
  mu_market_NSE, risk_market_NSE = find_market_portfolio('./data/NSE/NSEI.csv')
  
  print("\n\n********************  10 stocks from the BSE Index  ********************")

  stocks_name = ['SBIN.BO', 'TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'AXISBANK.BO', 'BAJFINANCE.BO', 'BSESN']
  model(stocks_name, 'BSE', mu_market_BSE, risk_market_BSE)

  print("\n\n********************  10 stocks from the NSE Index  ********************")

  stocks_name = ['SBIN.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'NSEI'] 
  model(stocks_name, 'NSE', mu_market_NSE, risk_market_NSE)

  print("\n\n********************  10 stocks not from any Index  ********************")

  stocks_name = ['BEL.BO', 'ASTRAL.NS', 'BLUEDART.BO', 'CROMPTON.NS', 'DIVISLAB.NS', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'TATACOMM.NS', 'SVT1.SG'] 
  model(stocks_name, 'Non-index stocks', -1, -1)


def draw_inference():
  print("**********  Inference about stocks taken from BSE  **********")
  stocks_name_BSE = ['SBIN.BO', 'TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'AXISBANK.BO', 'BAJFINANCE.BO', 'BSESN']
  beta_BSE = compute_beta(stocks_name_BSE, './data/BSE/BSESN.csv', 'BSE Index')
  mu_market_BSE, risk_market_BSE = find_market_portfolio('./data/BSE/BSESN.csv')
  beta_model(stocks_name_BSE, 'BSE', mu_market_BSE, risk_market_BSE, beta_BSE)


  print("\n\n**********  Inference about stocks taken from NSE  **********")
  stocks_name_NSE = ['SBIN.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'NSEI'] 
  beta_NSE = compute_beta(stocks_name_NSE, './data/NSE/NSEI.csv', 'NSE Index')
  mu_market_NSE, risk_market_NSE = find_market_portfolio('./data/NSE/NSEI.csv')
  beta_model(stocks_name_NSE, 'NSE', mu_market_NSE, risk_market_NSE, beta_NSE) 
    
  print("\n\n**********  Inference about stocks not taken from any index  with index taken from BSE values**********")
  stocks_name_non = ['BEL.BO', 'ASTRAL.NS', 'BLUEDART.BO', 'CROMPTON.NS', 'DIVISLAB.NS', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'TATACOMM.NS', 'SVT1.SG'] 
  beta_non_index_BSE = compute_beta(stocks_name_non, './data/BSE/BSESN.csv', 'Non-index')
  beta_model(stocks_name_non, 'Non-index stocks', mu_market_BSE, risk_market_BSE, beta_non_index_BSE) 

  print("\n\n**********  Inference about stocks not taken from any index  with index taken from NSE values**********")
  beta_non_index_NSE = compute_beta(stocks_name_non, './data/NSE/NSEI.csv', 'Non-index')
  beta_model(stocks_name_non, 'Non-index stocks', mu_market_NSE, risk_market_NSE, beta_non_index_NSE) 

def compare_beta():
  print("**********  Beta for securities in BSE  **********")
  stocks_name_BSE = ['SBIN.BO', 'TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'AXISBANK.BO', 'BAJFINANCE.BO', 'BSESN']
  beta_BSE = compute_beta(stocks_name_BSE, './data/BSE/BSESN.csv', 'BSE Index')

  for i in range(len(beta_BSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_BSE[i], beta_BSE[i]))

  print("\n\n**********  Beta for securities in NSE  **********")
  stocks_name_NSE = ['SBIN.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'NSEI'] 
  beta_NSE = compute_beta(stocks_name_NSE, './data/NSE/NSEI.csv', 'NSE Index')
  
  for i in range(len(beta_NSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_NSE[i], beta_NSE[i]))

  print("\n\n**********  Beta for securities in non-index using BSE Index  **********")
  stocks_name_non = ['BEL.BO', 'ASTRAL.NS', 'BLUEDART.BO', 'CROMPTON.NS', 'DIVISLAB.NS', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'TATACOMM.NS', 'SVT1.SG'] 
  beta_non_BSE = compute_beta(stocks_name_non, './data/BSE/BSESN.csv', 'Non-index')
  
  for i in range(len(beta_non_BSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_non[i], beta_non_BSE[i]))
    
  print("\n\n**********  Beta for securities in non-index using NSE Index  **********")
  stocks_name_non = ['BEL.BO', 'ASTRAL.NS', 'BLUEDART.BO', 'CROMPTON.NS', 'DIVISLAB.NS', 'ESCORTS.BO', 'GLENMARK.NS','PAGEIND.BO', 'TATACOMM.NS', 'SVT1.SG'] 
  beta_non_NSE = compute_beta(stocks_name_non, './data/NSE/NSEI.csv', 'Non-index')
  
  for i in range(len(beta_non_NSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_non[i], beta_non_NSE[i]))


LAB04_repeat()
draw_inference()
compare_beta()
