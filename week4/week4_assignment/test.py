import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, norm, normaltest

def return_calculate(prices: pd.DataFrame, method="ARITHMETIC", dateColumn="Date") -> pd.DataFrame:
    vars = prices.columns.values.tolist() #list of the column names
    nVars = len(vars)
    vars.remove(dateColumn) #remove the column of "date"
    if nVars == len(vars):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = nVars - 1
    p = np.array(prices.drop(columns=[dateColumn]))
    n = p.shape[0] #the number of rows
    m = p.shape[1] #the number of column
    p2 = np.empty((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]
    if method.upper() == "ARITHMETIC":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates = prices[dateColumn][1:]
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:,i]
    return out

portfolio = pd.read_csv('portfolio.csv')
prices = pd.read_csv('DailyPrices.csv')

def expo_weighted_cov(ret_data,w_lambda):
    weight = np.zeros(len(ret_data))
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)
    ret_means = ret_data - ret_data.mean()
    expo_w_cov = ret_means.T.values @ np.diag(weight) @ ret_means.values
    return expo_w_cov

def process_portfolio_data(portfolio, prices, p_type):
    if p_type == "total":
        co_assets = portfolio.drop('Portfolio', axis = 1)
        co_assets = co_assets.groupby(["Stock"], as_index=False)["Holding"].sum()
    else:
        co_assets = portfolio[portfolio['Portfolio'] == p_type]
    dailyprices = pd.concat([prices["Date"], prices[co_assets["Stock"]]], axis=1)
    holdings = co_assets['Holding']
    portfolio_price = np.dot(prices[co_assets["Stock"]].tail(1), co_assets['Holding'])
    return portfolio_price, dailyprices, holdings

print("Portfolio current prices: A, B, C, TOTAL")
A_data_p, A_day_p, A_holding = process_portfolio_data(portfolio, prices, 'A')
B_data_p, B_day_p, B_holding = process_portfolio_data(portfolio, prices, 'B')
C_data_p, C_day_p, C_holding = process_portfolio_data(portfolio, prices, 'C')
T_data_p, T_day_p, T_holding = process_portfolio_data(portfolio, prices, 'total')
print(A_data_p)
print(B_data_p)
print(C_data_p)
print(T_data_p)

#Calculate VaR using Delta Normal
def cal_delta_var(portfolio, prices, p_type, alpha=0.05, w_lambda=0.94, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    dailyprices = dailyprices.drop('Date', axis=1)
    dR_dr = (dailyprices.tail(1).T.values * holding.values.reshape(-1, 1)) / portfolio_price
    cov_mtx = expo_weighted_cov(returns, w_lambda)
    R_std = np.sqrt(np.transpose(dR_dr) @ cov_mtx @ dR_dr)
    var = (-1) * portfolio_price * norm.ppf(alpha) * R_std
    return var[0][0]


#Calculate VaR using historic simulation
def cal_his_var(portfolio, prices, p_type, alpha=0.05, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    np.random.seed(0)
    sim_ret = returns.sample(N, replace=True)
    dailyprices = dailyprices.drop('Date', axis=1)
    sim_change = np.dot(sim_ret * dailyprices.tail(1).values.reshape(dailyprices.shape[1]),holding)

    var = np.percentile(sim_change, alpha*100) * (-1)
    return var, sim_change


#print delta
delta_var_A = cal_delta_var(portfolio, prices, "A")
print("Delta Normal VaR of Portfolio A = {}".format(delta_var_A.round(4)))
delta_var_B = cal_delta_var(portfolio, prices, "B")
print("Delta Normal VaR of Portfolio B = {}".format(delta_var_B.round(4)))
delta_var_C = cal_delta_var(portfolio, prices, "C")
print("Delta Normal VaR of Portfolio C = {}".format(delta_var_C.round(4)))
delta_var_T = cal_delta_var(portfolio, prices, "total")
print("Delta Normal VaR of Portfolio TOTAL = {}".format(delta_var_T.round(4)))


#Plot Historic
his_var_A, his_dis_A = cal_his_var(portfolio, prices, "A")
print("VaR of Portfolio A using Historical Simulation = {}".format(his_var_A.round(4)))
his_var_B, his_dis_B = cal_his_var(portfolio, prices, "B")
print("VaR of Portfolio B using Historical Simulation = {}".format(his_var_B.round(4)))
his_var_C, his_dis_C = cal_his_var(portfolio, prices, "C")
print("VaR of Portfolio C using Historical Simulation = {}".format(his_var_C.round(4)))
his_var_T, his_dis_T = cal_his_var(portfolio, prices, "total")
print("VaR of Portfolio Total using Historical Simulation = {}".format(his_var_T.round(4)))