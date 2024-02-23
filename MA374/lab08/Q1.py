import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm 
import numpy as np
import math
from tabulate import tabulate
from datetime import datetime





df=pd.read_csv('nsedata1.csv')
date=df['Date'].to_numpy()


monthly_values = []
temp = -1
for i in date:
    date1 = datetime.strptime(i, '%d-%m-%Y')
    if date1.month != temp:
        monthly_values.append(np.where(date==i)[0][0])
        temp=date1.month

NSE_stock_name=['ASIANPAINT.NS','BAJFINANCE.NS','HDFC.NS','KOTAKBANK.NS','MARUTI.NS','RELIANCE.NS','TCS.NS','TITAN.NS','UPL.NS','WIPRO.NS','NSE Index']
NSE_stock=np.zeros((11,1236))
j=0
for i in NSE_stock_name:
    NSE_stock[j]=df[i].to_numpy()
    j=j+1

df=pd.read_csv('bsedata1.csv')
BSE_stock_name=['AXISBANK.BO','BEL.BO','FINCABLES.BO','HDFC.BO','HDFCBANK.BO','RELIANCE.BO','SBILIFE.BO','SKFINDIA.BO','SPICEJET.BO','TCS.BO','BSE Index']
BSE_stock=np.zeros((11,1236))
j=0
for i in BSE_stock_name:
    BSE_stock[j]=df[i].to_numpy()
    j=j+1

def solve(start,end,check):
    print('\n\n\n')
    print('For NSE Stock:')
    print('Stock Name       ','Volatality   ')
    NSE_volatality=np.zeros(11)
    for i in range(11):
        temp=np.array(NSE_stock[i])
        NSE_volatality[i]=math.sqrt(252*np.var(temp[start:end]))/temp[end-1]
        print(NSE_stock_name[i],'       ',NSE_volatality[i])
    print('\n\n\n')
    print('For BSE Stock   :')
    print('Stock Name       ','Volatality   ')
    BSE_volatality=np.zeros(11)
    for i in range(11):
        temp=np.array(BSE_stock[i])
        BSE_volatality[i]=math.sqrt(252*np.var(temp[start:end]))/temp[end-1]
        print(BSE_stock_name[i],'       ',BSE_volatality[i])

    def BSM(K,r,T,sigma,S):
        d1=(math.log(S/K)+(r+sigma*sigma/2)*(T))/(sigma*math.sqrt(T))
        d2=d1-sigma*math.sqrt(T)
        C=norm.cdf(d1)*S-norm.cdf(d2)*K*math.exp(-1*r*(T))
        P=norm.cdf(-1*d2)*K*math.exp(-1*r*(T))-norm.cdf(-1*d1)*S
        return C,P
    
    NSE_Call_price=np.zeros((4,11))
    BSE_Call_price=np.zeros((4,11))
    NSE_PUT_price=np.zeros((4,11))
    BSE_PUT_price=np.zeros((4,11))
    A=[1,0.5,0.1,1.5]
    for i in range(4):
        r=0.05
        T=0.5
        for j in range(11):
            temp=NSE_stock[j]
            NSE_Call_price[i][j],NSE_PUT_price[i][j]=BSM(A[i]*temp[end-1],r,T,NSE_volatality[j],temp[end-1])
            temp=BSE_stock[j]
            BSE_Call_price[i][j],BSE_PUT_price[i][j]=BSM(A[i]*temp[end-1],r,T,BSE_volatality[j],temp[end-1])
    if check==0:
        print('\n\n\n')
        print("For NSE Stocks : ")
        head=np.array(['A','Option type'])
        head=np.concatenate((head,NSE_stock_name))
        temp1=np.array([A[0],'Call Price'])
        temp1=np.concatenate((temp1,NSE_Call_price[0]))
        temp2=np.array([A[1],'Call Price'])
        temp2=np.concatenate((temp2,NSE_Call_price[1]))
        temp3=np.array([A[2],'Call Price'])
        temp3=np.concatenate((temp3,NSE_Call_price[2]))
        temp4=np.array([A[3],'Call Price'])
        temp4=np.concatenate((temp4,NSE_Call_price[3]))
        temp5=np.array([A[0],'Put Price'])
        temp5=np.concatenate((temp5,NSE_PUT_price[0]))
        temp6=np.array([A[1],'Put Price'])
        temp6=np.concatenate((temp6,NSE_PUT_price[1]))
        temp7=np.array([A[2],'Put Price'])
        temp7=np.concatenate((temp7,NSE_PUT_price[2]))
        temp8=np.array([A[3],'Put Price'])
        temp8=np.concatenate((temp8,NSE_PUT_price[3]))
        data=np.array((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8))
        print(tabulate(data,headers=head,tablefmt='fancy_grid'))
        print('\n\n\n')
        print("For BSE Stocks : ")
        head=np.array(['A','Option type'])
        head=np.concatenate((head,BSE_stock_name))
        temp1=np.array([A[0],'Call Price'])
        temp1=np.concatenate((temp1,BSE_Call_price[0]))
        temp2=np.array([A[1],'Call Price'])
        temp2=np.concatenate((temp2,BSE_Call_price[1]))
        temp3=np.array([A[2],'Call Price'])
        temp3=np.concatenate((temp3,BSE_Call_price[2]))
        temp4=np.array([A[3],'Call Price'])
        temp4=np.concatenate((temp4,BSE_Call_price[3]))
        temp5=np.array([A[0],'Put Price'])
        temp5=np.concatenate((temp5,BSE_PUT_price[0]))
        temp6=np.array([A[1],'Put Price'])
        temp6=np.concatenate((temp6,BSE_PUT_price[1]))
        temp7=np.array([A[2],'Put Price'])
        temp7=np.concatenate((temp7,BSE_PUT_price[2]))
        temp8=np.array([A[3],'Put Price'])
        temp8=np.concatenate((temp8,BSE_PUT_price[3]))
        data=np.array((temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8))
        print(tabulate(data,headers=head,tablefmt='fancy_grid'))
    # else :
    #     return NSE_volatality,NSE_Call_price,NSE_PUT_price,BSE_volatality,BSE_Call_price,BSE_PUT_price  

solve(1213,1236,0)