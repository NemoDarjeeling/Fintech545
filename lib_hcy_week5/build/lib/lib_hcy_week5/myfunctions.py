import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm

# covariance estimation
# calculate the exponentially weighted covariance matrix
# be cautious that this returns a comprehensive covariance matrix, with headers, so use expo_weighted_cov_valueOnly() to get pure values
def expo_weighted_cov(ret_data, w_lambda=0.94):
    weight = np.zeros(ret_data.shape[0]) # create a vector with num of rows corresponding to the periods of time, so we can attach weight to each time point
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i # weight for each time point will be decided by lambda set
    weight = weight/sum(weight) # assure that sum_weight == 1
    adj_ret_means = ret_data - ret_data.mean()
    expo_w_cov = adj_ret_means.T.values @ np.diag(weight) @ adj_ret_means.values # cov(x, x) = sum(wt * (xt - miux) * (xt - miux))
    return expo_w_cov

# covariance matrix obtained through exponential weight variance vector and exponential weight correlation matrix
def cal_ew_var_ew_cor(ret_data, w_lambda=0.97):
    ew_var_ew_cor = expo_weighted_cov(ret_data, w_lambda)
    return ew_var_ew_cor

# covariance matrix obtained through pearson variance vector and pearson correlation matrix
def cal_ps_var_ps_cor(ret_data):
    ps_var_ps_cor = np.cov(ret_data.T)
    return ps_var_ps_cor

# covariance matrix obtained through exponential weight variance vector and pearson correlation matrix
def cal_ew_var_ps_cor(ret_data, w_lambda=0.97):
    ew_cov = expo_weighted_cov(ret_data, w_lambda) # get exponential weight covariance matrix
    ew_var = np.diag(ew_cov) # extract exponential weight variance vector
    D_sqrt = np.diag(np.sqrt(ew_var)) # get D^1/2
    ps_cor = ret_data.corr()
    ew_var_ps_cor = np.dot(D_sqrt, np.dot(ps_cor, D_sqrt)) # D @ cor @ D'
    return ew_var_ps_cor

# covariance matrix obtained through pearson weight variance vector and exponential weight correlation matrix
def cal_ps_var_ew_cor(ret_data, w_lambda=0.97):
    ps_cov = ret_data.cov()
    ps_var = np.diagonal(ps_cov) # extract pearson variance vector
    D_sqrt = np.diag(np.sqrt(ps_var)) # get D^1/2
    ew_cov = expo_weighted_cov(ret_data, w_lambda)
    invSD = np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov))))
    ew_cor = invSD.dot(ew_cov).dot(invSD)
    ps_var_ew_cor = np.dot(D_sqrt, np.dot(ew_cor, D_sqrt)) # D @ cor @ D'
    return ps_var_ew_cor


# non-PSD fixes
# Rebonato and Jackel deal with non-PSD matrix, covariance -> correlation matrix conversion included
def near_psd(mtx, epsilon=0.0):
    n = mtx.shape[0]

    invSD = None
    out = mtx.copy()

    # calculate the correlation matrix if we got a covariance
    if (np.diag(out) == 1.0).sum() != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD.dot(out).dot(invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.square(vecs).dot(vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T.dot(vecs).dot(l)
    out = np.matmul(B, np.transpose(B))
    # we don't directly do S*lambda*S' as in this way we would have diagonal elements != 1

    # Add back to the correlation matrix to get covariance matrix
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD.dot(out).dot(invSD)

    return out

# Calculate Frobenius Norm
def fnorm(mtxa, mtxb):
    s = mtxa - mtxb
    norm = 0
    for i in range(len(s)):
        for j in range(len(s[0])):
            norm +=s[i][j]**2
    return norm

# Higham deal with non-PSD matrix
def Pu(mtx): # the projection to make the input matrix's diagonal elements to all be one
    new_mtx = mtx.copy()
    for i in range(len(mtx)):
        for j in range(len(mtx[0])):
            if i == j:
                new_mtx[i][j] = 1
    return new_mtx

def Ps(mtx, w): # the projection to make the input non-PSD matrix to PSD matrix
    mtx = np.sqrt(w) @ mtx @ np.sqrt(w)
    vals, vecs = np.linalg.eigh(mtx)
    vals = np.array([max(i,0) for i in vals])
    new_mtx = np.sqrt(w)@ vecs @ np.diag(vals) @ vecs.T @ np.sqrt(w)
    return new_mtx

def higham_psd(mtx, w, max_iteration = 1000): # w as weight is added to allow variation in matrix weight so you can attach different importance to different value, max_iteration is to prevent possible infinite loop
    r0 = np.inf
    Y = mtx
    S = np.zeros_like(Y)

    # calculate the correlation matrix if we got a covariance, it is a norm, we don't do PSD adjustion on covariance matrix
    invSD = None
    if np.count_nonzero(np.diag(Y) == 1.0) != mtx.shape[0]:
        invSD = np.diag(1.0 / np.sqrt(np.diag(Y)))
        Y = invSD.dot(Y).dot(invSD)
    C = Y.copy()

    # just do exactly as the pdf shows, reasoning details can be found in the manchester prof's paper
    for i in range(max_iteration):
        R = Y - S
        X = Ps(R, w)
        S = X - R
        Y = Pu(X)
        r = fnorm(Y, C) # use Frobenius Norm to calculate difference
        minval = np.linalg.eigvals(Y).min()
        tol = 1e-8 # tol as tolerance to see whether the difference between matrices are converging, set to be a very small positive value
        if abs(r - r0) < tol and minval > -1e-8: # if it is converging, we will consider the difference between input non PSD matrix and output PSD matrix is small enough and break from the loop
        # minval is used to ensure non negative values, delete this would cause higham ends before matrix turning PSD
            break
        else:
            r0 = r

    # Add back to the correlation matrix to get covariance matrix
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Y = invSD.dot(Y).dot(invSD)
    return Y

# confirm matrix is PSD or not
def is_psd(mtx):
    eigenvalues = np.linalg.eigh(mtx)[0] # this returns both eigenvalues and eigenvectors, while ensure real numbers, better than np.linalg.eigvals()
    return np.all(eigenvalues >= -1e-8) # if all elements in the matrix are positive, return true, which indicates input is PSD matrix; if not so, return false which indicates input is not PSD matrix

# Multivariate normal distribution
# Cholesky Factorization for PSD matrix
def chol_psd(cov_matrix):
    # cov_mtx = cov_matrix.values # we do this to obtain the pure value of covariance metrics and not bug for stock names
    cov_mtx = cov_matrix
    n = cov_mtx.shape[0] # initialize the root matrix with 0 values
    root = np.zeros_like(cov_mtx)
    for j in range(n): # loop over columns
        s = 0.0
        if j > 0: # if we are not on the first column, calculate the dot product of the preceding row values
            s = np.dot(root[j, :j], root[j, :j])
        temp = cov_mtx[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0: # diagonal elements
            for i in range(j + 1, n):
                root[i, j] = 0.0
        else: # update off diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (cov_mtx[i, j] - s) * ir
    return root


# multivariate simulation - directly from a covariance matrix
def simu_from_cov(cov_mtx, num_samples, mean = None):
    if mean is None: # allows variation of mean vector, if not valid input for this, the default value is zero
        mean = np.zeros(cov_mtx.shape[0])

    chol_decomp = chol_psd(cov_mtx)
    std_norm_samples = np.random.randn(num_samples, cov_mtx.shape[0])

    samples = mean + np.dot(std_norm_samples, chol_decomp.T)
    return samples


# multivariate simulation - using PCA of which % variance explained as an input
def pca_percent_explain(sort_eigenvalues, percent_explain):
    if percent_explain ==1:
        return len(sort_eigenvalues)
    n_eigenvalues = 0
    cum_var = np.cumsum(sort_eigenvalues) / np.sum(sort_eigenvalues) # returns an array with the variance explained by elements from 1st to 1st / 2nd / 3rd / 4th...
    for i in range(len(cum_var)):
        if cum_var[i] >= percent_explain:
            n_eigenvalues = i + 1
            break
    return n_eigenvalues

def simu_from_pca(cov_mtx, num_samples, percent_explain):
    eigenvalues, eigenvectors = np.linalg.eig(cov_mtx)
    p_idx = eigenvalues > 0 # only want those eigenvalues are non 0, as if it is zero, we remove everything related to that, including the eigenvalue in matrix lambda and its corresponding column in the
    # p_idx will return a numpy array of boolean values, indicating which eigenvalue is positive, and which eigenvalue is not
    pos_eigenvalues = eigenvalues[p_idx]
    pos_eigenvectors = eigenvectors[:, p_idx]
    # sort in descending order
    s_idx = np.argsort(pos_eigenvalues)[::-1] # return an array indicating the sequence for original input array
    sort_eigenvalues = eigenvalues[s_idx] # rearrange the eigenvalue array
    sort_eigenvectors = eigenvectors[:, s_idx] # rearrange the column of eigenvector corresponding to the change in eigenvalue array
    n_eigenvalues = pca_percent_explain(sort_eigenvalues, percent_explain)
    # print(n_eigenvalues)
    explain_eigenvalues = sort_eigenvalues[:n_eigenvalues] # only need the required eigenvalues enough to explain % set
    explain_eigenvectors = sort_eigenvectors[:,:n_eigenvalues] # only need the required eigenvectors enough to explain % set
    # np.random.seed(42)
    std_normals = np.random.normal(size=(n_eigenvalues, num_samples))

    # actual simulation
    sqrt_eigenvalues = np.sqrt(explain_eigenvalues) # calculate the square root of significant eigenvalues
    diag_sqrt_eigenvalues = np.diag(sqrt_eigenvalues) # create a diagonal matrix from the square root of eigenvalues
    transformation_matrix = np.dot(explain_eigenvectors, diag_sqrt_eigenvalues) # compute the transformation matrix 'B'
    multivariate_normal_samples = np.dot(transformation_matrix, std_normals) # generate the multivariate normal samples
    samples = np.transpose(multivariate_normal_samples) # transpose the result for the final output
    return samples

# implement return_calculate() as Prof. Pazzula gave us
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
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

# calculate VaR using a normal distribution
def calculate_VaR(data, alpha=0.05):
  return -np.quantile(data, alpha)

def norm_VaR(returns, alpha=0.05, num_sample=1000):
        mean = returns.mean()
        std = returns.std()
        Rt = np.random.normal(mean, std, num_sample)
        var = calculate_VaR(Rt, alpha)
        return var, Rt

# the function to calculate Var based on normal dist with ew var
def norm_ew_VaR(returns, alpha=0.05, num_sample=1000, w_lambda=0.94):
        mean = returns.mean()
        std = np.sqrt(expo_weighted_cov(returns, w_lambda))
        Rt = np.random.normal(mean, std, num_sample)
        var = calculate_VaR(Rt, alpha)
        return var, Rt

# calculate VaR using a MLE fitted T distribution
def MLE_T_VaR(returns, alpha=0.05, num_sample=1000):
    result = t.fit(returns, method="MLE") # fit the returns into MLE
    df = result[0] # used to get the required parameters for T distribution simulation
    loc = result[1]
    scale = result[2]
    Rt = t(df, loc, scale).rvs(num_sample) # generate num_sample random variates from the t-distribution of (df, loc, scale)
    var = calculate_VaR(Rt, alpha)
    return var, Rt

# calculate VaR using AR(1)
def ar1_VaR(returns, alpha=0.05, num_sample=1000):
    result = ARIMA(returns, order=(1, 0, 0)).fit()
    t_a = result.params[0]  # constant term
    t_phi = result.params[1]  # coefficient of the lagged term
    resid_std = np.std(result.resid) # the residual of the fit
    last_return = returns[len(returns)] # obtain the last return in returns
    Rt = t_a + t_phi * last_return + np.random.normal(loc=0, scale=resid_std, size=num_sample) # alpha + phi * Rt-1 (since it is AR(1)) + residual, which we use exact residual of the fit, to some extent an element of "simulation of historical distribution"
    var = calculate_VaR(Rt, alpha)
    return var, Rt

# calculating VaR using historical distribution
def his_VaR(returns, alpha=0.05):
    Rt = returns.values # no further simulation, just obtain all data to get historical distribution
    var = calculate_VaR(Rt, alpha)
    return var, Rt

# Deal with portfolio
# parsing data
def parsing_port(aPortfolio, prices):
    daily_price = pd.concat([prices["Date"], prices[aPortfolio["Stock"]]], axis=1) # concat "date" and all columns that is in aPortfolio, concat by columns
    holdings = aPortfolio["Holding"]
    port_value = np.dot(prices[aPortfolio["Stock"]].tail(1), aPortfolio['Holding']) # row (the last one price of each corresponding stock in aPortfolio) * column (the holdings of each corresponding stock in Portfolio)
    return daily_price, holdings, port_value

def expo_weighted_cov_valueOnly(ret_data,w_lambda): # we need to rewrite this as previous function have heads in the matrix, it doesn't matter when we just want std, but matters
    weight = np.zeros(ret_data.shape[0])
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)
    ret_means = ret_data - ret_data.mean()
    expo_w_cov = ret_means.T.values @ np.diag(weight) @ ret_means.values
    return expo_w_cov

# calculate VaR using Delta Normal
def cal_delta_VaR(aPortfolio, prices, alpha=0.05, w_lambda=0.94):
    daily_price, holdings, port_value = parsing_port(aPortfolio, prices)
    returns = return_calculate(daily_price).drop("Date", axis = 1)
    latest_prices = daily_price.drop("Date", axis = 1).tail(1).values # tail() as opposite to head(), get last n rows
    dR_dr = latest_prices.T * holdings.values.reshape(-1,1) / port_value # transpose and re-organize into two columns, reshape(-1, 1) means reshape to be 1 column, -1 means num of rows will auto be set to required
    cov_mtx = expo_weighted_cov_valueOnly(returns, w_lambda)
    R_std = np.sqrt(np.transpose(dR_dr) @ cov_mtx @ dR_dr)
    var = (-1) * port_value * norm.ppf(alpha) * R_std
    return var[0][0] # var is [[]]

# calculate VaR using historic simulation
def cal_his_VaR(aPortfolio, prices, alpha=0.05, num_sample = 1000):
    daily_price, holdings, port_value = parsing_port(aPortfolio, prices)
    returns = return_calculate(daily_price).drop("Date", axis = 1)
    simu = returns.sample(num_sample, replace=True)
    latest_prices = daily_price.drop("Date", axis = 1).tail(1).values.reshape(-1,1)

    pchange = simu * latest_prices.T
    holdings = holdings.values.reshape(-1, 1)
    simu_change = pchange @ holdings
    var = calculate_VaR(simu_change, alpha)
    return var, simu_change

# calculate ES
def cal_ES(var, sim_data):
  return -np.mean(sim_data[sim_data <= -var])