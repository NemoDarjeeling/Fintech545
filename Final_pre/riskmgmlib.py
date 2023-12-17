import pandas as pd
import numpy as np
import scipy
from statsmodels.tsa.arima.model import ARIMA
import copy
from scipy.stats import norm, t
from scipy.optimize import fsolve
from scipy.optimize import minimize
from typing import List


# weight_gen() used for generating exponentially weighted weights
def weight_gen(n, lambd=0.94):
    weight = np.zeros(n)
    for i in range(n):
        weight[i] = (1-lambd) * (lambd) ** i
    normalized_weight = weight / np.sum(weight)
    return normalized_weight

# ewcov_gen() used for calculate the exponentially weighted covariance matrix, using the normalized_weight from weight_gen()
def ewcov_gen(data, weight):
    data = data - data.mean(axis=0)
    weight = np.diag(weight)
    data_left = weight@data
    data_right = np.dot(data.T, data_left)
    return data_right


# cholesky decomposition, a component of later steps
def chol_psd(a):
    n = a.shape[1]
    # Initialize the root matrix with 0 values
    root = np.zeros((n, n))
    # loop over columns
    for j in range(n):
        s = 0
        # if we are not on the first column, calculate the dot product of the preceeding row values.
        if j > 0:
            s = root[j, 0:j].T@root[j, 0:j]
        temp = a[j, j] - s
        # here temp is the critical value, when temp>=-1e-3, there is no nan but still invalid answer, but it is close
        if temp <= 0 and temp >= -1e-3:
            temp = 0
        root[j, j] = np.sqrt(temp)
        # Check for the 0 eigan value.  Just set the column to 0 if we have one
        if root[j, j] == 0:
            for i in range(j, n):
                root[j, i] = 0
        else:
            ir = 1/root[j, j]
            for i in range(j+1, n):
                s = root[i, 0:j].T@root[j, 0:j]
                root[i, j] = (a[i, j] - s) * ir
    return root


# near_psd() used for normal way of fixing non-psd matrix
def near_psd(a, epsilon=0.0):
    is_cov = False
    for i in np.diag(a):
        if abs(i-1) > 1e-8:
            is_cov = True
        else:
            is_cov = False
            break
    if is_cov:
        invSD = np.diag(1/np.sqrt(np.diag(a)))
        a = invSD@a@invSD
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    T = 1/(np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    if is_cov:
        invSD = np.diag(1/np.diag(invSD))
        out = invSD @ out @ invSD
    return out


# Higham_method() [including Forbenius_Norm(), projection_u(), projection_s()] used for Higham's method of fix non-psd matrix
def Frobenius_Norm(a):
    return np.sqrt(np.sum(np.square(a)))

def projection_u(a):
    np.fill_diagonal(a, 1.0)
    return a

# A note here, epsilon is the smallest eigenvalue, 0 does not work well here, will still generate very small negativa values, so I set it to 1e-7
def projection_s(a, epsilon=1e-7):
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    return vecs@np.diag(vals)@vecs.T

def Higham_method(a, tol=1e-8):
    s = 0
    gamma = np.inf
    y = a
    # iteration
    while True:
        r = y - s
        x = projection_s(r)
        s = x - r
        y = projection_u(x)
        gamma_next = Frobenius_Norm(y-a)
        if abs(gamma - gamma_next) < tol:
            break
        gamma = gamma_next
    return y


# is_psd() used to check whether a matrix is PSD or not
def is_psd(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)


# sim_mvn_from_cov() used for multivariate simulation - directly from a covariance matrix, similar to the one below, just for normal dist
def sim_mvn_from_cov(cov, num_of_simulation=25000):
    return chol_psd(cov) @ np.random.normal(size=(cov.shape[0], num_of_simulation))

# variance matrix

# var() used for calculate the variance if were given a covariance matrix
def var(cov):
    return np.diag(cov)


# corr() used for calculate the correlation matrix if were given a covariance matrix
def corr(cov):
    return np.diag(1/np.sqrt(var(cov))) @ cov @ np.diag(1/np.sqrt(var(cov))).T


# cov() used for calculate the covariance matrix if were given a correlation matrix and a list of the variance
def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T


# PCA_with_percent() used for multivariate simulation - through PCA with an optional parameter for % variance explained
def PCA_with_percent(cov, percent=0.95, num_of_simulation=25000):
    eigenvalue, eigenvector = np.linalg.eigh(cov)
    total = np.sum(eigenvalue)
    for i in range(cov.shape[0]):
        i = len(eigenvalue)-i-1
        if eigenvalue[i] < 0:
            eigenvalue = eigenvalue[i+1:]
            eigenvector = eigenvector[:, i+1:]
            break
        if sum(eigenvalue[i:])/total > percent:
            eigenvalue = eigenvalue[i:]
            eigenvector = eigenvector[:, i:]
            break
    simulate = np.random.normal(size=(len(eigenvalue), num_of_simulation))
    return eigenvector @ np.diag(np.sqrt(eigenvalue)) @ simulate


# direct_simulation() used for multivariate simulation - directly from a covariance matrix
def direct_simulation(cov, n_samples=25000):
    B = chol_psd(cov)
    r = scipy.random.randn(len(B[0]), n_samples)
    return B @ r


# calculate_var used to given data and alpha, return the VaR
def calculate_var(data, mean=0, alpha=0.05): # mean is the current expected return, so you should include actual mean if you assume 0 mean, or just subtract each data with mean
    return mean-np.quantile(data, alpha)


# normal_var() used to calculate VaR when returns are fitted using normal distribution and then simulated
def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    sigma = np.std(data)
    simulation_norm = np.random.normal(mean, sigma, nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm


# ewcov_normal_var() used to calculate VaR when returns are fitted using normal distribution with ew var and then simulated
def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000, lambd=0.94):
    ew_cov = ewcov_gen(data, weight_gen(len(data), lambd))
    ew_variance = ew_cov
    sigma = np.sqrt(ew_variance)
    simulation_ew = np.random.normal(mean, sigma, nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew


# t_var() used to calculate VaR when returns are fitted using T-distribution by MLE and then simulated
def t_var(data, mean=0, alpha=0.05, nsamples=10000):
    params = scipy.stats.t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = scipy.stats.t(df, loc, scale).rvs(nsamples)
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t


# ar1_var() used to calculate VaR when returns are fitted using ar1 and then simulated
def ar1_var(returns, alpha=0.05, num_sample=1000):
    result = ARIMA(returns, order=(1, 0, 0)).fit()
    t_a = result.params[0]  # constant term
    t_phi = result.params[1]  # coefficient of the lagged term
    resid_std = np.std(result.resid) # the residual of the fit
    last_return = returns[len(returns)] # obtain the last return in returns
    Rt = t_a + t_phi * last_return + np.random.normal(loc=0, scale=resid_std, size=num_sample) # alpha + phi * Rt-1 (since it is AR(1)) + residual, which we use exact residual of the fit, to some extent an element of "simulation of historical distribution"
    var = calculate_var(Rt)
    return var

def historic_var(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)


# 5. ES calculation
def calculate_es(data, mean=0, alpha=0.05):
    return -np.mean(data[data < -calculate_var(data, mean, alpha)])


# def return_calculate(price, method='discrete'):
#     returns = []
#     for i in range(len(price)-1):
#         returns.append(price[i+1]/price[i])
#     returns = np.array(returns)
#     if method == 'discrete':
#         return returns - 1
#     if method == 'log':
#         return np.log(returns)

# derivative

# gbsm() is used to calculate the theoretical price of a European call or put option based on various inputs using the Black-Scholes-Merton formula.
def gbsm(option_type, S, X, r, b, sigma, T):
    d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*np.exp((b-r)*T)*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)
    else:
        return X*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp((b-r)*T)*norm.cdf(-d1)

# implied_vol() is used to calculate the implied volatility, or sigma, given other parameters of the option, basically reversed engineering of gbsm()
def implied_vol(option_type, S, X, T, r, b, market_price, x0=0.5):
    def equation(sigma):
        return gbsm(option_type, S, X, r, b, sigma, T) - market_price
    # Back solve the Black-Scholes formula to get the implied volatility
    return fsolve(equation, x0=x0, xtol=0.0001)[0]

# bt_no_div is used to calculate the option price, using non-gbsm, and with no dividend, could be applied to American options
def bt_no_div(call, underlying, strike, ttm, rf, b, ivol, N):
    # call: the type of option,
    # underlying: the current price of the underlying asset,
    # strike: the strike price of the option,
    # ttm: time to maturity,
    # rf: risk-free interest rate,
    # b: cost-of-carry rate,
    # ivol: implied volatility,
    # N: the number of time steps
    dt = ttm/N
    u = np.exp(ivol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = np.exp(-rf*dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return (n+1)*(n+2) // 2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
    nNodes = nNodeFunc(N)

    optionValues = [0.0] * nNodes

    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            idx = idxFunc(i,j)
            price = underlying*u**i*d**(j-i)
            optionValues[idx] = max(0,z*(price-strike))

            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)]))

    return optionValues[0]

# bt_with_div() is used to calculate option price using non-gbsm, with dividends
def bt_with_div(call, underlying, strike, ttm, rf, b, divAmts, divTimes, ivol, N):
    # call: The type of option, where True indicates a call option and False indicates a put option.
    # underlying: The current price of the underlying asset (such as a stock) for which the option is written.
    # strike: The strike price of the option, which is the fixed price at which the holder can buy (for a call) or sell (for a put) the underlying asset.
    # ttm: Time to maturity of the option, expressed in years. It represents how much time is left until the option expires.
    # rf: The risk-free interest rate, typically representing the theoretical return of an investment with no risk of financial loss.
    # b: The cost-of-carry rate, which in the context of this function, is assumed to be equal to the risk-free interest rate (rf). This simplification is due to the discrete treatment of dividends.
    # divAmts: A list of dividend amounts, with each element in the list representing the amount of a single dividend payment.
    # divTimes: A list of times at which dividends are paid, expressed as the number of time steps (from the total N steps) until each dividend payment.
    # ivol: Implied volatility of the underlying asset, reflecting the market's expectation of the asset's future volatility over the life of the option.
    # N: The number of time steps used in the binomial tree model, affecting the granularity and accuracy of the option price calculation.

    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_no_div(call, underlying, strike, ttm, rf, b, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def nNodeFunc(n: int) -> int:
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i: int, j: int) -> int:
        return nNodeFunc(j - 1) + i

    nDiv = len(divTimes)
    nNodes = nNodeFunc(divTimes[0])

    optionValues = [0] * nNodes

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))

            if j < divTimes[0]:
                #times before the dividend working backward induction
                optionValues[idx] = max(0, z * (price - strike))
                optionValues[idx] = max(optionValues[idx], df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd * optionValues[idxFunc(i, j + 1)]))
            else:
                #time of the dividend
                valNoExercise = bt_with_div(call, price - divAmts[0], strike, ttm - divTimes[0] * dt, rf, b, divAmts[1:], [t - divTimes[0] for t in divTimes[1:]], ivol, N - divTimes[0])
                valExercise = max(0, z * (price - strike))
                optionValues[idx] = max(valNoExercise, valExercise)

    return optionValues[0]

def find_iv(call, underlying, strike, ttm, rf, b, divAmts, divTimes, N, price, guess=0.5):
    def f(ivol):
        return bt_with_div(call, underlying, strike, ttm, rf, b, divAmts, divTimes, ivol, N) - price
    return fsolve(f, guess)[0]

# below are functions and codes to calculate the Greeks
def d1(S, K, b, sigma, T):
    return (np.log(S/K) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
def d2(S, K, b, sigma, T):
    return d1(S, K, b, sigma, T) - sigma*np.sqrt(T)
# delta_call = np.exp((b-r)*T)*norm.cdf(rml.d1(S, X, b, sigma, T))
# print('Delta of the call option is: ', round(delta_call,3))
# delta_put = np.exp((b-r)*T)*(norm.cdf(rml.d1(S, X, b, sigma, T)) - 1)
# print('Delta of the put option is: ', round(delta_put,3))
# gamma = np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))/(S*sigma*np.sqrt(T))
# print('Gamma of the call option is: ', round(gamma,3))
# print('Gamma of the put option is: ', round(gamma,3))
# vega = S*np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))*np.sqrt(T)
# print('Vega of the call option is: ', round(vega,3))
# print('Vega of the put option is: ', round(vega,3))
# theta_call = -S*np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))*sigma/(2*np.sqrt(T)) - (b-r)*S*np.exp((b-r)*T)*norm.cdf(rml.d1(S, X, b, sigma, T)) - r*X*np.exp(-r*T)*norm.cdf(rml.d2(S, X, b, sigma, T))
# print('Theta of the call option is: ', round(theta_call,3))
# theta_put = -S*np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))*sigma/(2*np.sqrt(T)) + (b-r)*S*np.exp((b-r)*T)*norm.cdf(-rml.d1(S, X, b, sigma, T)) + r*X*np.exp(-r*T)*norm.cdf(-rml.d2(S, X, b, sigma, T))
# print('Theta of the put option is: ', round(theta_put,3))
# # because the textbook has an assumption that b = rf but it does not hold here, we need to calculate the rho seperately
# rho_call = -T*S*np.exp(b*T - r*T)*norm.cdf(rml.d1(S, X, b, sigma, T)) + X*T*np.exp(-r*T)*norm.cdf(rml.d2(S, X, b, sigma, T))
# print('Rho of the call option is: ', round(rho_call,3))
# rho_put = -X*T*np.exp(-r*T)*norm.cdf(-rml.d2(S, X, b, sigma, T))+T*S*np.exp(b*T - r*T)*norm.cdf(-rml.d1(S, X, b, sigma, T))
# print('Rho of the put option is: ', round(rho_put,3))
# carry_rho_call = S*T*np.exp((b-r)*T)*norm.cdf(rml.d1(S, X, b, sigma, T))
# print('Carry Rho of the call option is: ', round(carry_rho_call,3))
# carry_rho_put = -S*T*np.exp((b-r)*T)*norm.cdf(-rml.d1(S, X, b, sigma, T))
# print('Carry Rho of the put option is: ', round(carry_rho_put,3))


# portfolio_return() is used to calculate the portfolio return, for the whole portfolio, not individual asset
def portfolio_return(weights, expected_returns):
    return np.sum(weights * expected_returns)

# portfolio_volatility() is used to calculate the portfolio volatility, not individual asset
def portfolio_volatility(weights, correlation_matrix, volatilities):
    covariance_matrix = np.diag(
        volatilities)@correlation_matrix@np.diag(volatilities)
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility

# sharpe_ratio is used to calculate the sharpe ratio for a portfolio
def sharpe_ratio(weights, expected_returns, correlation_matrix, volatilities):
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(
        weights, correlation_matrix, volatilities)
    sharpe_ratio = (p_return-0.04) / p_volatility
    return sharpe_ratio

# optimize_sharpe_ratio() is used to calculate the weights for each asset in the portfolio to achieve optimized sharpe ratio
def optimize_sharpe_ratio(expected_returns, volatilities, correlation_matrix):
    # expected_returns is an array or list of expected returns for each asset in the portfolio.
    # volatilities is an array or list of the volatilities (standard deviations) of each asset.
    # correlation_matrix is a matrix representing the correlation coefficients between the assets in the portfolio.
    n_assets = len(expected_returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    # The line below is restricted for positive value
    bounds = [(0, 1) for i in range(n_assets)]
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(lambda x: -sharpe_ratio(x, expected_returns, correlation_matrix, volatilities),
                      initial_weights,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result.x, -result.fun
# result.x is the list of the optimal weights for each asset in the portfolio that maximize the Sharpe ratio.
# -result.fun is the maximum Sharpe ratio achievable with these optimal weights.


# risk_parity_weight() is used to calculate the weight for each asset in the portfolio to achieve risk parity
def risk_parity_weight(corr_matrix, vol):
    # corr_matrix is a correlation matrix representing the correlation coefficients between the assets in the portfolio. remember to use corr() to convert covariance matrix to correlation matrix
    # vol is a list representing the volatilities (standard deviations) of the individual assets in the portfolio
    covar = np.outer(vol, vol) * corr_matrix
    n = covar.shape[0]

    def pvol(w):
        return np.sqrt(w @ covar @ w)

    def pCSD(w):
        pVol = pvol(w)
        csd = w * (covar @ w) / pVol
        return csd

    def sseCSD(w):
        csd = pCSD(w)
        mCSD = np.sum(csd) / n
        dCsd = csd - mCSD
        se = dCsd * dCsd
        # Add a large multiplier for better convergence
        return 1.0e5 * np.sum(se)

    # Define optimization problem
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for i in range(n)]
    w0 = np.ones(n) / n

    # Solve optimization problem
    result = minimize(sseCSD, w0, method='SLSQP',
                      bounds=bounds, constraints=cons)
    wrp = np.round(result.x, decimals=4)
    return wrp
# wrp is a list of the weights for asset to achieve risk parity


# ex_post_contribution() is used to calculate both the return contribution and risk contribution from each asset to the whole portfolio
def ex_post_contribution(w, stocks, upReturns):
    # w is a list of the assets' initial weights
    # stocks is the list of the assets' names
    # upReturns is a df of all the assets' returns

    # Calculate portfolio return and updated weights for each day
    n = upReturns.shape[0]
    pReturn = np.empty(n)
    weights = np.empty((n, len(w)))
    lastW = np.array(w)
    matReturns = upReturns[stocks].values
    for i in range(n):
        # Save current weights in matrix
        weights[i, :] = lastW

        # Update weights by return
        lastW = lastW * (1.0 + matReturns[i, :])

        # Portfolio return is the sum of the updated weights
        pR = np.sum(lastW)
        # Normalize the weights back so sum = 1
        lastW = lastW / pR
        # Store the return
        pReturn[i] = pR - 1

    # Set the portfolio return in the Update Return DataFrame
    upReturns['Portfolio'] = pReturn

    # Calculate the total return
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    # Calculate the Carino K
    k = np.log(totalRet + 1) / totalRet

    # Carino k_t is the ratio scaled by 1/K
    carinoK = np.log(1.0 + pReturn) / pReturn / k
    # Calculate the return attribution
    attrib = pd.DataFrame(
        matReturns * weights * carinoK[:, None], columns=stocks, index=upReturns.index)

    # Set up a DataFrame for output.
    Attribution = pd.DataFrame(index=["TotalReturn", "Return Attribution"])
    # Loop over the stocks
    for s in stocks + ['Portfolio']:
        # Total Stock return over the period
        tr = np.exp(np.sum(np.log(upReturns[s] + 1))) - 1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        atr = attrib[s].sum() if s != 'Portfolio' else tr
        # Set the values
        Attribution[s] = [tr, atr]
    Y = matReturns * weights
    # Set up X with the Portfolio Return
    X = np.column_stack((np.ones(n), pReturn))
    # Calculate the Beta and discard the intercept
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    B = B[1, :]
    # Component SD is Beta times the standard Deviation of the portfolio
    cSD = B * np.std(pReturn)

    # Check that the sum of component SD is equal to the portfolio SD
    np.isclose(np.sum(cSD), np.std(pReturn))

    # Add the Vol attribution to the output
    Attribution = Attribution.append(pd.DataFrame({
        'Value': 'Vol Attribution',
        **{stocks[i]: cSD[i] for i in range(len(stocks))},
        'Portfolio': np.std(pReturn)
    }, index=[0]), ignore_index=True)
    Attribution.loc[0, 'Value'] = "Total Return"
    Attribution.loc[1, 'Value'] = "Return Attribution"
    return Attribution, weights
# Attribution: A DataFrame that includes: Total return of each asset over the period. Return attribution of each asset, showing how much each contributed to the portfolio's total return. Volatility attribution, indicating each asset's contribution to the portfolio's overall risk.
# weights is a matrix (numpy array) of the updated weights of the assets in the portfolio for each time period.

# cal_t_pVals() is used to simulate the total value of multiple assets
def cal_t_pVals(port, returns_port, price):
    # port is a df specifying the holdings of each individual asset
    # returns_port is a df with historical returns of each individual individual asset along time span
    # price is an array specifying the latest price of each individual asset
    return_cdf=[]
    par=[]
    for col in returns_port.columns:
        df, loc, scale = t.fit(returns_port[col].values)
        par.append([df,loc,scale])
        return_cdf.append(t.cdf(returns_port[col].values, df=df, loc=loc, scale=scale).tolist())
    return_cdf=pd.DataFrame(return_cdf).T
    spearman_cor=return_cdf.corr(method='spearman')
    sample=pd.DataFrame(PCA_with_percent(spearman_cor)).T
    sample_cdf=[]
    for col in sample.columns:
        sample_cdf.append(norm.cdf(sample[col].values, loc=0, scale=1).tolist())
    simu_return=[]
    for i in range(len(sample_cdf)):
        simu_return.append(t.ppf(sample_cdf[i], df=par[i][0], loc=par[i][1], scale=par[i][2]))
    simu_return=np.array(simu_return)

    sim_price=(1 + simu_return.T)*price
    pVals = sim_price.dot(port['Holding'])
    pVals.sort()
    return pVals
# pVals is simulated portfolio values

# return_calculate() is used to calculate returns
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    # prices is a table of prices for different assets on each column
    # method is to choose whether we want arithmatic "DISCRETE", or continuous "LOG"
    # dateColumn is to assume there is a date column
    vars = prices.columns.values.tolist() # extract the index of columns (column names), and convert it into a list
    nVars = len(vars) # number of columns
    vars.remove(dateColumn) # remove date column to get pure data
    if nVars == len(vars): # if the number of columns does not change after we remove the date column, this means we don't have the date column from the start, and such time series analysis will be meaningless, thus we raise an error on it
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = nVars - 1 # update the number of columns by reflecting the removal of date column
    p = np.array(prices.drop(columns=[dateColumn])) # drop the date column and convert to np date frame
    n = p.shape[0] # num of rows
    m = p.shape[1] # num of column
    p2 = np.empty((n-1, m)) # creates an empty NumPy array p2 with shape (n-1, m)
    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0 # if it is discrete compounding, then r = pt / pt-1 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2) # if it is continuous compounding, then r = ln(pt / pt-1)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")") # there is no method other than discrete or compounding, so input error
    dates = prices[dateColumn][1:] # get the date, as the first row corresponds to the first day, and first day has no return, so we start from the second row
    out = pd.DataFrame({dateColumn: dates}) # initialize an empty "out" df, with its dateColumn set to be dates extracted
    for i in range(nVars): # add all rows calculated values corresponding to the stock name in that column in vars, then input this matrix into df "out"
        out[vars[i]] = p2[:,i]
    return out # "out" is the df having stock name, date and return

