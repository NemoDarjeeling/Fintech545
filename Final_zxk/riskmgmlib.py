import pandas as pd
import numpy as np
import scipy
import copy
from scipy.stats import norm, t
from scipy.optimize import fsolve
from scipy.optimize import minimize
# 1. Covariance estimation techniques


# Generating expoentially weighted weights
def weight_gen(n, lambd=0.94):
    weight = np.zeros(n)
    for i in range(n):
        weight[i] = (1-lambd) * (lambd) ** i
    normalized_weight = weight / np.sum(weight)
    return normalized_weight


def ewcov_gen(data, weight):
    data = data - data.mean(axis=0)
    weight = np.diag(weight)
    data_left = weight@data
    data_right = np.dot(data.T, data_left)
    return data_right
# 2. Non PSD fixes for correlation matrices

# chol_psd function, return the lower half matrix


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

# fixing psd matrix


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

# Implement Higham’s 2002 nearest psd correlation function


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

# if a matrix is psd


def is_psd(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)
# 3. Simulation Methods


def sim_mvn_from_cov(cov, num_of_simulation=25000):
    return chol_psd(cov) @ np.random.normal(size=(cov.shape[0], num_of_simulation))

# variance matrix


def var(cov):
    return np.diag(cov)
# Correlation matrix


def corr(cov):
    return np.diag(1/np.sqrt(var(cov))) @ cov @ np.diag(1/np.sqrt(var(cov))).T
# Covariance matrix


def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T


# using PCA with an optional parameter for % variance explained.
# return the simulation result
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

# direct simulation


def direct_simulation(cov, n_samples=25000):
    B = chol_psd(cov)
    r = scipy.random.randn(len(B[0]), n_samples)
    return B @ r

# 4. VaR calculation methods

# Given data and alpha, return the VaR


def calculate_var(data, mean=0, alpha=0.05):
    return mean-np.quantile(data, alpha)


def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    sigma = np.std(data)
    simulation_norm = np.random.normal(mean, sigma, nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm


def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000, lambd=0.94):
    ew_cov = ewcov_gen(data, weight_gen(len(data), lambd))
    ew_variance = ew_cov
    sigma = np.sqrt(ew_variance)
    simulation_ew = np.random.normal(mean, sigma, nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew


def t_var(data, mean=0, alpha=0.05, nsamples=10000):
    params = scipy.stats.t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = scipy.stats.t(df, loc, scale).rvs(nsamples)
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t


def historic_var(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)


# 5. ES calculation
def calculate_es(data, mean=0, alpha=0.05):
    return -np.mean(data[data < -calculate_var(data, mean, alpha)])

# 6. Other functions

# Given a price series, return the returns


def return_calculate(price, method='discrete'):
    returns = []
    for i in range(len(price)-1):
        returns.append(price[i+1]/price[i])
    returns = np.array(returns)
    if method == 'discrete':
        return returns - 1
    if method == 'log':
        return np.log(returns)

# derivative


def gbsm(option_type, S, X, r, b, sigma, T):
    d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*np.exp((b-r)*T)*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)
    else:
        return X*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp((b-r)*T)*norm.cdf(-d1)


def implied_vol(option_type, S, X, T, r, b, market_price, x0=0.5):
    def equation(sigma):
        return gbsm(option_type, S, X, r, b, sigma, T) - market_price
    # Back solve the Black-Scholes formula to get the implied volatility
    return fsolve(equation, x0=x0, xtol=0.0001)[0]


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

# sharpe ratio


def portfolio_return(weights, expected_returns):
    return np.sum(weights * expected_returns)


def portfolio_volatility(weights, correlation_matrix, volatilities):
    covariance_matrix = np.diag(
        volatilities)@correlation_matrix@np.diag(volatilities)
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility


def sharpe_ratio(weights, expected_returns, correlation_matrix, volatilities):
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(
        weights, correlation_matrix, volatilities)
    sharpe_ratio = (p_return-0.04) / p_volatility
    return sharpe_ratio


def optimize_sharpe_ratio(expected_returns, volatilities, correlation_matrix):
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

# risk-parity

# Define portfolio functions


def risk_parity_weight(corr_matrix, vol):
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


# EXPOST ATTRIBUTION
def ex_post_contribution(w, stocks, upReturns):
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
