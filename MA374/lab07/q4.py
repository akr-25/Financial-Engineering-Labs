import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import tabulate as tab


def BSM(S, K, T, r, t, sigma, option='both', q=0):
    if np.max(t) > T:
        raise ValueError('t must be less than T')

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    callPrice = S * np.exp(-q * (T-t)) * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    putPrice = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S * np.exp(-q * (T-t)) * norm.cdf(-d1)

    if option == 'call':
        return callPrice
    elif option == 'put':
        return putPrice
    else:
        return callPrice, putPrice


def defaultParams():
    return {
        'S': 100,
        'K': 100,
        'T': 1,
        'r': 0.05,
        't': 0,
        'sigma': 0.2,
        'option': 'call',
        'q': 0
    }

paramsName = {
    'S': 'Stock Price',
    'K': 'Strike Price',
    'r': 'Risk-free Rate',
    't': 'Time',
    'sigma': 'Volatility',
}

def SensitivityAnalysis(paramToChange, range):
    params = defaultParams()
    params[paramToChange] = range
    params['option'] = 'both'
    callPrice, putPrice = BSM(**params)
    plt.plot(range, callPrice, label='Call Price')
    plt.plot(range, putPrice, label='Put Price')
    plt.legend()
    plt.xlabel(paramsName[paramToChange])
    plt.ylabel('Option Price')
    plt.title('Sensitivity Analysis of ' + paramsName[paramToChange])
    plt.savefig('image/q4/Sensitivity Analysis of ' + paramsName[paramToChange])
    plt.close()
    return callPrice, putPrice


def SensitivityAnalysis3D(paramToChange1, range1, paramToChange2, range2):
    range1, range2 = np.meshgrid(range1, range2)

    params = defaultParams()
    params[paramToChange2] = range2
    params[paramToChange1] = range1
    params['option'] = 'both'
    callPrice, putPrice = BSM(**params)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(range1, range2, callPrice, cmap='viridis', edgecolor='none', label='Call Price')
    ax.plot_surface(range1, range2, putPrice, cmap='plasma' , edgecolor='none', label='Put Price')
    ax.set_xlabel(paramToChange1)
    ax.set_ylabel(paramToChange2)
    ax.set_zlabel('Option Price')
    ax.set_title('Sensitivity Analysis of ' + paramToChange1 + ' and ' + paramToChange2)
    plt.savefig('image/q4/Sensitivity Analysis of ' + paramToChange1 + ' and ' + paramToChange2 + '.png')
    plt.close()
    return callPrice, putPrice

rangeForParams = {
    'S': np.linspace(50, 200, 100),
    'K': np.linspace(50, 200, 100),
    'r': np.linspace(0, 0.1, 100),
    't': np.linspace(0, 1, 100),
    'sigma': np.linspace(0, 1, 100),
    'q': np.linspace(0, 0.1, 100)
}

def plot_sens2D():
    for param in paramsName.keys():
        callPrice, putPrice = SensitivityAnalysis(param, rangeForParams[param])
        leng = len(callPrice)
        randomIndex = np.random.randint(0, leng, 10)
        randomIndex = np.argsort(rangeForParams[param][randomIndex])
        table = tab.tabulate(np.array([rangeForParams[param][randomIndex], callPrice[randomIndex], putPrice[randomIndex]]).T, headers=[paramsName[param], 'Call Price', 'Put Price'], tablefmt='orgtbl') 
        print('Sensitivity Analysis of ' + paramsName[param])
        print(table)
        print('\n\n\n')


        


def plot_sens3D():
    length = len(paramsName.keys())
    for i in range(length):
        for j in range(i+1, length):
            param1 = list(paramsName.keys())[i]
            param2 = list(paramsName.keys())[j]
            callPrice, putPrice = SensitivityAnalysis3D(param1, rangeForParams[param1], param2, rangeForParams[param2])
            # Tabulate some of the results for the report (max rows and columns are set to 10)
            leng = len(callPrice)
            randomIndex1 = np.random.randint(0, leng, 10)
            randomIndex2 = np.random.randint(0, leng, 10)
            randomIndex1 = np.argsort(rangeForParams[param1][randomIndex1])
            randomIndex2 = np.argsort(rangeForParams[param2][randomIndex2])

            table = tab.tabulate(np.array([rangeForParams[param1][randomIndex1], rangeForParams[param2][randomIndex2], callPrice[randomIndex1, randomIndex2], putPrice[randomIndex1, randomIndex2]]).T, headers=[param1, param2, 'Call Price', 'Put Price'], tablefmt='orgtbl')
            print('\n\n\n')

            print('Sensitivity Analysis of ' + param1 + ' and ' + param2)
            print(table)

            
if __name__ == '__main__':
    plot_sens2D()
    plot_sens3D()