spearman coefficient is more robust for outlier circumstances, the largest number in an array of 6 numbers being 10000 or 10000000 would always have a rank of 6, of which if you use pearson, outlier would be hazardous.

beta 0 is the mean of error term

error term would be vector during practice

we need MLE for non-normal distribution

n-k is degree of freedom, d in k is the additional parameters, such as standard deviation in normal distribution, scale and xx in T-distribution

error of t depend only on error of t-1, not depend on t-2; while y depend on y-1 and y-1 depend on y-2 and ultimately y depend on all y before that

scalar 标量 vector 向量 matrix 矩阵

goodness-of-fit tests can be used to evaluate how well a model fits data when using Maximum Likelihood Estimation (MLE), even under different distributional assumptions. However, the specific test to use might vary based on the assumed distribution.

## what is professor's code regarding MLE doing?
The Julia code performs Maximum Likelihood Estimation (MLE) in two parts:  
Part 1: MLE for a Normal Distribution  
1. **Data Generation**: 100 samples are drawn from a normal distribution with mean \( \mu = 1.0 \) and standard deviation \( \sigma = 5.0 \).
   ```julia
   samples = 100
   d = Normal(1.0, 5.0)
   x = rand(d, samples)
   ```
2. **Log-likelihood Function**: A custom log-likelihood function (`myll`) for a normal distribution is defined. This function takes the parameters \( \mu \) and \( \sigma \) and computes the log-likelihood for the given data \( x \).
   ```julia
   function myll(m, s)
       ...
       return ll
   end
   ```
   
**why would we need the log-likelihood?**
The log-likelihood function is a critical concept in statistical modeling, specifically in Maximum Likelihood Estimation (MLE). It describes how well your model explains the observed data. In other words, given a set of parameters and observed data, the likelihood gives you the probability of observing that data under those parameters.
The purpose of defining these log-likelihood functions in the code is to optimize them. By optimizing, we mean finding the parameter values that maximize the log-likelihood. These parameter values are then considered to be the 'best fit' estimates for the data at hand, according to the MLE criterion.
例如，给定一堆数据，他们有多少可能性是来自于N(0,1)? 多少可能性来自于N(1,5)? 多少可能性来自于T(a,b,n)?

3. **MLE Optimization**: The optimization problem is set up to maximize the log-likelihood using the Ipopt optimizer. Two variables \( \mu \) and \( \sigma \) are optimized.
   ```julia
   mle = Model(Ipopt.Optimizer)
   ...
   optimize!(mle)
   ```
4. **Results**: The estimated values for \( \mu \) and \( \sigma \) are printed.

Part 2: MLE for Regression
1. **Data Generation**: Generates 5000 samples for a linear regression model with 5 coefficients.
   ```julia
   n = 5000
   Beta = [i for i in 1:5]
   x = hcat(fill(1.0, n), randn(n, 4))
   y = x * Beta + randn(n)
   ```
2. **Log-likelihood Function for Regression**: A new custom log-likelihood function (`myll`) is defined. This function works with the parameters \( \beta \) (the regression coefficients) and \( \sigma \).
   ```julia 
   function myll(s, b...)
       ...
       return ll
   end
   ```
   
**How is it used in optimization?**
By register(mle,:ll,6,myll;autodiff=true)

3. **MLE Optimization**: Like before, the optimization problem is set up to maximize the log-likelihood. This time, the optimization is done over \( \beta \) and \( \sigma \).
   ```julia
   mle = Model(Ipopt.Optimizer)
   ...
   optimize!(mle)
   ```

4. **Results**: The estimated values for \( \beta \) are printed. It also calculates and prints the Ordinary Least Squares (OLS) estimates for comparison.
Overall, the code showcases how to perform MLE for estimating parameters of a normal distribution and a regression model. It uses the Ipopt optimization library for the heavy lifting.


The Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF) are tools used in time series analysis to understand the correlation structure of a data series and to identify the appropriate models to fit it.
### Autocorrelation Function (ACF):
The ACF describes the correlation between a series and its lags. In other words, it helps to understand how a given observation is related to its previous observations. For example, if the original time series is denoted by \( X_t \), the ACF gives correlations between \( X_t \) and \( X_{t-k} \) for different values of \( k \).
#### Interpretation:
- If the ACF shows a slow decay, then a Moving Average (MA) model may be appropriate.
- If the ACF cuts off after a certain point \( p \), it suggests an Autoregressive (AR) model of order \( p \).
### Partial Autocorrelation Function (PACF):
The PACF, on the other hand, describes the correlation between a series and its lags that is not explained by previous lags. For example, the partial autocorrelation at lag 2 would give the correlation between \( X_t \) and \( X_{t-2} \) that is not explained by \( X_{t-1} \).
#### Interpretation:
- If the PACF shows a sharp cut-off and/or tapers towards zero after a certain lag \( p \), this suggests that an AR model of order \( p \) should be used.
- If the PACF decays more slowly, a MA model might be more appropriate.
### How to Use ACF and PACF:
1. **Identify AR Order**: Look at the PACF plot; the point where the PACF values become insignificant (usually at the confidence interval boundary) suggests the order of the AR model.
2. **Identify MA Order**: Look at the ACF plot; the point where the ACF values fall within the confidence interval suggests the order of the MA model.
3. **Mixed Models**: Sometimes both the ACF and PACF show a mixed pattern, suggesting that an ARIMA model (a mix of AR and MA) may be appropriate.
These interpretations offer general guidelines and starting points; model identification should also include steps like fitting the model, checking its assumptions, and possibly refining the model based on those checks.