# Assignment question set
### Week 2
#### Q1: Remember from last week we discussed that skewness and kurtosis functions in statistical packages are often biased. Is your function biased? Prove or disprove your hypothesis.
#### Q2: Fit the data in problem2.csv using OLS and calculate the error vector. Look at its distribution. How well does it fit the assumption of normally distributed errors? Fit the data using MLE given the assumption of normality. Then fit the MLE using the assumption of a T distribution of the errors. Which is the best fit? What are the fitted parameters of each and how do they compare? What does this tell us about the breaking of the normality assumption in regards to expected values in this case?
#### Q3: Simulate AR(1) through AR(3) and MA(1) through MA(3) processes. Compare their ACF and PACF graphs. How do the graphs help us to identify the type and order of each process?

Q1:  
Skewness: Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive, zero, negative, or undefined. (Field: Statistics, Econometrics, Regression Analysis, Probability Distributions.)  
Kurtosis: Kurtosis is a statistical measure that describes the shape of a probability distribution, specifically in terms of the "tailedness" or peakedness. Distributions with high kurtosis exhibit tail data exceeding the tails of the normal distribution (e.g., five or more standard deviations from the mean). (Field: Statistics, Econometrics, Regression Analysis, Probability Distributions.)  
Biased: In the context of statistics, a biased estimate is an over- or under-estimation of a population parameter due to systematic error. It is contrasted with random error, which may overestimate or underestimate but on average is correct. (Field: Statistics, Econometrics, Estimation Theory.)  
Hypothesis: In statistics, a hypothesis is a claim or statement about a property of a population. Hypothesis testing is a procedure that uses statistical evidence to test if the claim is true or false. (Field: Statistics, Hypothesis Testing.)  


Q2:  
OLS (Ordinary Least Squares): OLS is a type of linear least squares method for estimating the unknown parameters in a linear regression model. OLS minimizes the sum of the squares of the differences between the observed dependent variable and those predicted by the linear function. (Field: Statistics, Econometrics, Regression Analysis.)  
Error Vector: In regression analysis, the error vector is the vector of differences between the observed and predicted values. (Field: Statistics, Econometrics, Regression Analysis.)  
MLE (Maximum Likelihood Estimation): MLE is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observed data. (Field: Statistics, Econometrics, Estimation Theory.)  
T Distribution: The t-distribution is a type of probability distribution that is symmetric and bell-shaped, similar to the normal distribution but with thicker tails. It is used in hypothesis testing and in constructing confidence intervals. (Field: Statistics, Probability Theory.)  
Normality Assumption: The assumption that the errors, or residuals, of a regression model are normally distributed. This assumption is important for statistical inference. (Field: Statistics, Regression Analysis.)  


Q3:  
AR (Autoregressive) Process: An AR process is a type of time series where the current value depends linearly on its own previous values. The number in parentheses indicates the order, or number of previous terms, used. (Field: Statistics, Time Series Analysis.)  
MA (Moving Average) Process: A MA process is a type of time series where the current value depends linearly on the previous error terms. The number in parentheses indicates the order, or number of previous error terms, used. (Field: Statistics, Time Series Analysis.)  
ACF (Autocorrelation Function): The ACF is a function that represents autocorrelation of a time series as a function of the time lag. It is used to identify patterns in the data like trends or seasonality. (Field: Statistics, Time Series Analysis.)  
PACF (Partial Autocorrelation Function): The PACF is a summary of the relationship between an observation in a time series with observations at prior time steps with the relationships of intervening observations removed. (Field: Statistics, Time Series Analysis.)  


### Week 3
#### Q1: Use the stock returns in DailyReturn.csv for this problem. DailyReturn.csv contains returns for 100 large US stocks and as well as the ETF, SPY which tracks the S&P500. Create a routine for calculating an exponentially weighted covariance matrix. If you have a package that calculates it for you, verify that it calculates the values you expect. This means you still have to implement it. Vary λ ∈ (0, 1). Use PCA and plot the cumulative variance explained by each eigenvalue for each λ chosen. What does this tell us about values of λ and the effect it has on the covariance matrix?
#### Q2: Copy the chol_psd(), and near_psd() functions from the course repository – implement in your programming language of choice. These are core functions you will need throughout the remainder of the class. Implement Higham’s 2002 nearest psd correlation function. Generate a non-psd correlation matrix that is 500x500. Use near_psd() and Higham’s method to fix the matrix. Confirm the matrix is now PSD. Compare the results of both using the Frobenius Norm. Compare the run time between the two. How does the run time of each function compare as N increases? Based on the above, discuss the pros and cons of each method and when you would use each. There is no wrong answer here, I want you to think through this and tell me what you think.
#### Q3: Using DailyReturn.csv. Implement a multivariate normal simulation that allows for simulation directly from a covariance matrix or using PCA with an optional parameter for % variance explained. If you have a library that can do these, you still need to implement it yourself for this homework and prove that it functions as expected. Generate a correlation matrix and variance vector 2 ways: 1. Standard Pearson correlation/variance (you do not need to reimplement the cor() and var() functions). 2. Exponentially weighted λ = 0. 97 Combine these to form 4 different covariance matrices. (Pearson correlation + var()), Pearson correlation + EW variance, etc.) Simulate 25,000 draws from each covariance matrix using: 1. Direct Simulation 2. PCA with 100% explained. 3. PCA with 75% explained. 4. PCA with 50% explained. Calculate the covariance of the simulated values. Compare the simulated covariance to its input matrix using the Frobenius Norm (L2 norm, sum of the square of the difference between the matrices). Compare the run times for each simulation. What can we say about the trade-offs between time to run and accuracy

Q1:  
Stock Returns: In finance, a stock return is the gain or loss made by an investor from changes in the stock price over a period of time, including any dividends paid. (Field: Finance, Investment Analysis.)  
ETF (Exchange-Traded Fund): An ETF is an investment fund traded on stock exchanges, much like stocks. An ETF holds assets such as stocks, bonds, or commodities. (Field: Finance, Investment Analysis.)  
SPY: SPY is the ticker symbol for the SPDR S&P 500 ETF Trust, which is designed to track the S&P 500 stock market index. (Field: Finance, Investment Analysis.)  
Exponentially Weighted Covariance Matrix: A type of covariance matrix where recent observations are given more weight than older observations, with weights declining exponentially. This is often used in finance for risk estimation. (Field: Finance, Econometrics, Time Series Analysis.)  
PCA (Principal Component Analysis): PCA is a statistical procedure that uses orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. (Field: Statistics, Multivariate Analysis.)  
Eigenvalue: Eigenvalues are a special set of scalars associated with a linear system of equations (i.e., a matrix equation) that are sometimes also known as characteristic roots. In the context of PCA, eigenvalues often indicate the amount of variance in the data accounted for by each principal component. (Field: Mathematics, Linear Algebra.)  


Q2:  
chol_psd(), near_psd() functions: These are functions used in matrix computations. The chol_psd() function calculates the Cholesky decomposition of a positive semi-definite matrix. The near_psd() function adjusts a given square matrix to make it positive semi-definite. (Field: Mathematics, Linear Algebra.)  
Higham's 2002 nearest psd correlation function: Higham's method is a numerical algorithm to find the nearest positive semi-definite matrix to a given matrix. (Field: Mathematics, Numerical Analysis.)  
Frobenius Norm: The Frobenius norm is a matrix norm of a given matrix which is analogous to the Euclidean norm for vectors. It is used as a measure of matrix size and to compare matrices. (Field: Mathematics, Linear Algebra.)  


Q3:  
Multivariate Normal Simulation: This refers to the simulation of multivariate normal random variables, which are sets of variables that are jointly normally distributed. This is a common task in Monte Carlo simulations. (Field: Statistics, Simulation Methods.)  
PCA (Principal Component Analysis): As defined earlier, PCA is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to simplify data for exploration and visualization, or to make data less noisy before using it to train machine learning models. (Field: Statistics, Multivariate Analysis.)  
Pearson Correlation: Pearson correlation is a measure of the linear correlation between two variables X and Y. It has a value between +1 and −1, where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation. (Field: Statistics, Correlation Analysis.)  
Variance: Variance is a measurement of the spread between numbers in a data set. More specifically, variance measures how far each number in the set is from the mean (average) and thus from every other number in the set. Variance is often denoted by the symbol σ^2. (Field: Statistics, Descriptive Statistics.)  
Exponentially Weighted Variance: This is a measure of variability that assigns more weight to recent changes in a variable's level, with the weights declining exponentially for older observations. (Field: Statistics, Time Series Analysis.)  

### Week 4
#### Q1: Calculate and compare the expected value and standard deviation of price at time 𝑡(𝑃𝑡) , given each of the 3 types of price returns, assuming 𝑟𝑡 ~ 𝑁(0, 𝜎2) . Simulate each return equation using 𝑟 and show the mean and standard deviation match your expectations.
#### Q2: Implement a function similar to the “return_calculate()” in this week’s code. Allow the user to specify the method of return calculation. Use DailyPrices.csv. Calculate the arithmetic returns for all prices. Remove the mean from the series so that the mean(META)=0 Calculate VaR 1. Using a normal distribution. 2. Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94) 3. Using a MLE fitted T distribution. 4. Using a fitted AR(1) model. 5. Using a Historic Simulation. Compare the 5 values.
#### Q3: Using Portfolio.csv and DailyPrices.csv. Assume the expected return on all stocks is 0. This file contains the stock holdings of 3 portfolios. You own each of these portfolios. Using an exponentially weighted covariance with lambda = 0.94, calculate the VaR of each portfolio as well as your total VaR (VaR of the total holdings). Express VaR as a $. Discuss your methods and your results. Choose a different model for returns and calculate VaR again. Why did you choose that model? How did the model change affect the results?

Q1:  
Expected Value: In probability theory, the expected value of a random variable is a key aspect of its probability distribution and essentially a type of average. It's the sum of all possible values each multiplied by the probability of its occurrence. (Field: Statistics, Probability Theory.)  
Standard Deviation: The standard deviation is a measure of the amount of variation or dispersion in a set of values. A low standard deviation indicates that the values tend to be close to the mean of the set, while a high standard deviation indicates that the values are spread out over a wider range. (Field: Statistics, Descriptive Statistics.)  
Price at Time 𝑡(𝑃𝑡): This is referring to the price of a particular asset at a specific time period, 𝑡. (Field: Finance, Financial Analysis.)  
Price Returns: The return on an investment can be measured as the change in price over a period of time, often expressed as a percentage of the initial investment. There are several types of return, such as simple return, log return, etc. (Field: Finance, Investment Analysis.)  
𝑟𝑡 ~ 𝑁(0, 𝜎2): This is a notation for a normally distributed random variable 𝑟𝑡, with a mean of 0 and a variance of 𝜎2. (Field: Statistics, Probability Theory.)  


Q2:  
return_calculate() Function: This function would typically calculate the returns on a financial asset based on the price data. The method of return calculation (simple, log, etc.) might be specified. (Field: Finance, Investment Analysis.)  
Arithmetic Returns: These are a type of return calculated as the difference between the price at the end and the price at the start of the period, divided by the price at the start of the period. (Field: Finance, Investment Analysis.)  
VaR (Value at Risk): VaR is a statistical technique used to measure and quantify the level of financial risk within a firm or investment portfolio over a specific time frame. (Field: Finance, Risk Management.)  
MLE fitted T Distribution: A t-distribution fitted to data using maximum likelihood estimation (MLE). (Field: Statistics, Probability Theory.)  
AR(1) Model: This refers to an autoregressive model of order 1, a type of time series model in which a value from a time series is regressed on the previous value from that series. (Field: Statistics, Time Series Analysis.)  
Historic Simulation: A method used for calculating VaR where the empirical distribution of past returns is used to simulate possible future returns. (Field: Finance, Risk Management.)  


Q3:  
Portfolio: In finance, a portfolio is a collection of financial investments like stocks, bonds, commodities, cash, and cash equivalents, including closed-end funds and exchange-traded funds (ETFs). (Field: Finance, Portfolio Management.)  
Exponentially Weighted Covariance: A covariance matrix calculated with more weight given to recent observations in a time series, with the weights declining exponentially for older observations. (Field: Statistics, Time Series Analysis.)  
VaR (Value at Risk): As defined earlier, VaR is a measure of the risk of loss for investments. It estimates how much a set of investments might lose, given normal market conditions, in a set time period such as a day. (Field: Finance, Risk Management.)  

### Week 5 
#### Q1: Use the data in problem1.csv. Fit a Normal Distribution and a Generalized T distribution to this data. Calculate the VaR and ES for both fitted distributions. Overlay the graphs the distribution PDFs, VaR, and ES values. What do you notice? Explain the differences.
#### Q2: In your main repository, create a Library for risk management. Create modules, classes, packages, etc as you see fit. Include all the functionality we have discussed so far in class. Make sure it includes 1. Covariance estimation techniques. 2. Non PSD fixes for correlation matrices 3. Simulation Methods 4. VaR calculation methods (all discussed) 5. ES calculation Create a test suite and show that each function performs as expected.
#### Q3: Use your repository from #2. Using Portfolio.csv and DailyPrices.csv. Assume the expected return on all stocks is 0. This file contains the stock holdings of 3 portfolios. You own each of these portfolios. Fit a Generalized T model to each stock and calculate the VaR and ES of each portfolio as well as your total VaR and ES. Compare the results from this to your VaR form Problem 3 from Week 4.

Q1:  
Normal Distribution: The Normal Distribution, also known as Gaussian distribution, is a probability distribution that is symmetric about the mean. It shows data near the mean are more frequent in occurrence than data far from the mean. (Field: Statistics, Probability Theory.)  
Generalized T Distribution: This is a type of probability distribution that resembles the normal distribution but has heavier tails. It's often used when data exhibit kurtosis (i.e., extreme values are more likely than under a normal distribution). (Field: Statistics, Probability Theory.)  
VaR (Value at Risk): As mentioned before, VaR is a measure of the risk of loss for investments. It estimates how much a set of investments might lose, given normal market conditions, in a set time period such as a day. (Field: Finance, Risk Management.)  
ES (Expected Shortfall): Also known as conditional value at risk (CVaR), it is a risk measure that quantifies the expected value of the tail risk. (Field: Finance, Risk Management.)  
Distribution PDFs: This refers to probability density functions (PDFs) of statistical distributions. PDF defines a probability distribution for a continuous random variable. (Field: Statistics, Probability Theory.)  


Q2:  
Library for Risk Management: Libraries are a collection of pre-compiled routines that a program can use. The routines, sometimes called modules, allow a programmer to avoid writing common functions from scratch. In this case, it is a library focused on risk management. (Field: Computer Science, Software Engineering.)  
Covariance Estimation Techniques: These are methods to estimate the covariance matrix of a given dataset. Covariance is a measure of how changes in one variable are associated with changes in a second variable. (Field: Statistics, Multivariate Analysis.)  
Non PSD Fixes for Correlation Matrices: Positive semi-definite (PSD) matrices are a necessary requirement for many statistical methodologies. In certain scenarios, computed correlation matrices are not PSD due to computational error or model misspecification. Several methods exist to correct these "non-PSD" matrices. (Field: Mathematics, Linear Algebra; Statistics.)  
Simulation Methods: Simulation is the imitation of the operation of a real-world process or system over time. In finance, it is often used to estimate the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. (Field: Statistics, Finance.)  
VaR calculation methods: Different methods for calculating Value at Risk, a measure of investment risk. (Field: Finance, Risk Management.)  
ES Calculation: Methods for calculating Expected Shortfall, a measure of tail risk. (Field: Finance, Risk Management.)  
Test Suite: A collection of test cases that are intended to be used to test a software program to show that it has some specified set of behaviors. (Field: Computer Science, Software Testing.)  


Q3:  
Portfolio: As mentioned before, in finance, a portfolio is a collection of financial investments like stocks, bonds, commodities, cash, and cash equivalents, including closed-end funds and exchange-traded funds (ETFs). (Field: Finance, Portfolio Management.)  
Generalized T model: A statistical model based on the Generalized T distribution. (Field: Statistics, Probability Theory.)  
VaR (Value at Risk): As mentioned before, VaR is a measure of the risk of loss for investments. (Field: Finance, Risk Management.)  
ES (Expected Shortfall): Also known as conditional value at risk (CVaR), it is a risk measure that quantifies the expected value of the tail risk. (Field: Finance, Risk Management.)  

### Week 6 
#### Q1: Assume you a call and a put option with the following ● Current Stock Price $165 ● Current Date 03/03/2023 ● Options Expiration Date 03/17/2023 ● Risk Free Rate of 4.25% ● Continuously Compounding Coupon of 0.53% Calculate the time to maturity using calendar days (not trading days). For a range of implied volatilities between 10% and 80%, plot the value of the call and the put. Discuss these graphs. How does the supply and demand affect the implied volatility?
#### Q2: Use the options found in AAPL_Options.csv ● Current AAPL price is 151.03 ● Current Date, Risk Free Rate and Dividend Rate are the same as problem #1. Calculate the implied volatility for each option. Plot the implied volatility vs the strike price for Puts and Calls. Discuss the shape of these graphs. What market dynamics could make these graphs? There are bonus points available on this question based on your discussion. Take some time to research if needed.
#### Q3: Use the portfolios found in problem3.csv ● Current AAPL price is 151.03 ● Current Date, Risk Free Rate and Dividend Rate are the same as problem #1. For each of the portfolios, graph the portfolio value over a range of underlying values. Plot the portfolio values and discuss the shapes. Bonus points available for tying these graphs to other topics discussed in the lecture. Using DailyPrices.csv. Calculate the log returns of AAPL. Demean the series so there is 0 mean. Fit an AR(1) model to AAPL returns. Simulate AAPL returns 10 days ahead and apply those returns to the current AAPL price (above). Calculate Mean, VaR and ES. Discuss.

Q1:  
Call Option: It's a financial contract giving the owner the right, but not the obligation, to buy a specified amount of an underlying security at a predetermined price within a specified timeframe. (Field: Finance, Options Trading)  
Put Option: It's a financial contract giving the owner the right, but not the obligation, to sell a specified amount of an underlying security at a predetermined price within a specified timeframe. (Field: Finance, Options Trading)  
Stock Price: The price of a single share of a company's stock. (Field: Finance, Stock Market)  
Option Expiration Date: The last day that an options contract can be exercised. After this, the option is worthless. (Field: Finance, Options Trading)  
Risk-Free Rate: The theoretical rate of return of an investment with no risk of financial loss. (Field: Finance, Investment Analysis)  
Continuously Compounding Coupon: A method of interest accumulation in which interest is added to the principal amount and the next interest calculation is based on this increased amount. (Field: Finance, Compound Interest)  
Implied Volatility: The market's forecast of a likely movement in a security's price. It's often used to price options contracts. (Field: Finance, Options Trading, Market Analysis)  
Supply and Demand: Fundamental economic concept of a balance caused by the quantity of a good the producer wishes to produce and the quantity consumers wish to buy. In this context, it refers to the buying and selling pressure on options, which affects the implied volatility. (Field: Economics)  


Q2:  
Strike Price: The set price at which an options contract can be bought or sold when it is exercised. (Field: Finance, Options Trading)  
AAPL_Options.csv: This is a csv file containing data, presumably about AAPL's options. The exact content isn't clear without further information, but it's likely to include details like the option type, strike price, expiration date, etc. (Field: Data Analysis, Finance, Options Trading)  


Q3:  
Portfolio: A collection of financial investments like stocks, bonds, commodities, cash, and cash equivalents, as well as their fund counterparts. (Field: Finance, Portfolio Management)  
problem3.csv: A csv file containing data, presumably about different portfolios. The exact content isn't clear without further information. (Field: Data Analysis, Finance, Portfolio Management)  
Underlying Values: In this context, it likely refers to the prices of the securities in the portfolio. (Field: Finance, Investment Analysis)  
DailyPrices.csv: A csv file containing daily price data, presumably for AAPL stock. (Field: Data Analysis, Finance, Stock Market)  
Log Returns: A measure of the rate of continuous compounding return over a period for a security. (Field: Finance, Investment Analysis, Statistics)  
Demean: This typically refers to subtracting the mean of a dataset from each data point, resulting in a new dataset with a mean of 0. (Field: Statistics, Data Analysis)  
AR(1) Model: AutoRegressive model of order 1. A type of time series model where current values are based on a linear combination of past values. (Field: Statistics, Econometrics, Time Series Analysis)  
VaR (Value at Risk): A measure of the risk of investments. It estimates how much a set of investments might lose, given normal market conditions, in a set time period such as a day. (Field: Finance, Risk Management)  
ES (Expected Shortfall): Also known as conditional value at risk (CVaR), it is a risk measure that quantifies the expected value of the tail loss. (Field: Finance, Risk Management)  

### Week 7 
#### Q1: Current Stock Price $165; Strike Price $165; Current Date 03/13/2022; Options Expiration Date 04/15/2022; Risk Free Rate of 4.25%; Continuously Compounding Coupon of 0.53%. Implement the closed form Greeks for GBSM. Implement a finite difference derivative calculation. Compare the values between the two methods for both a call and a put. Implement the binomial tree valuation for American options with and without discrete dividends. Assume the stock above: Pays dividend on 4/11/2022 of $0.88 Calculate the value of the call and the put. Calculate the Greeks of each. What is the sensitivity of the put and call to a change in the dividend amount?
#### Q2: Using the options portfolios from Problem3 last week (named problem2.csv in this week’s repo) and assuming: American Options; Current Date 03/03/2023; Current AAPL price is 165; Risk Free Rate of 4.25%; Dividend Payment of $1.00 on 3/15/2023 Using DailyPrices.csv. Fit a Normal distribution to AAPL returns – assume 0 mean return. Simulate AAPL returns 10 days ahead and apply those returns to the current AAPL price (above). Calculate Mean, VaR and ES. Calculate VaR and ES using Delta-Normal. Present all VaR and ES values a $ loss, not percentages. Compare these results to last week’s results.
#### Q3: Use the Fama French 3 factor return time series (F_F_Research_Data_Factors_daily.CSV) as well as the Carhart Momentum time series (F-F_Momentum_Factor_daily.CSV) to fit a 4 factor model to the following stocks. Fama stores values as percentages, you will need to divide by 100 (or multiply the stock returns by 100) to get like units. Based on the past 10 years of factor returns, find the expected annual return of each stock. Construct an annual covariance matrix for the 10 stocks. Assume the risk-free rate is 0.0425. Find the super-efficient portfolio

Q1:  
Stock Price: The current price at which a particular stock is being sold in the market. (Field: Finance, Stock Market.)  
Strike Price: The predetermined price at which the holder of an option can buy or sell the underlying security or commodity. (Field: Finance, Options Trading.)  
Risk-Free Rate: The theoretical rate of return of an investment with zero risk. It represents the interest an investor would expect from an absolutely risk-free investment over a specified period of time. (Field: Finance, Investment Theory.)  
Continuously Compounding Coupon: This refers to the process of earning interest on the initial deposit, also known as the principal, and the accumulated interest of previous periods of a deposit or loan. (Field: Finance, Investment Theory.)  
GBSM (Generalized Black-Scholes Model): A mathematical model used to calculate the theoretical price of financial derivatives. (Field: Finance, Derivative Pricing.)  
Finite Difference Derivative Calculation: A numerical method for approximating the derivative of a given function. (Field: Mathematics, Numerical Analysis.)  
Call and Put: A call option is a contract that gives the owner the right, but not the obligation, to buy a stock at a specified price within a specific time period. A put option gives the owner the right to sell a stock. (Field: Finance, Options Trading.)  
Binomial Tree Valuation: A method for valuing derivatives, where the underlying asset can take on one of only two possible, discrete values in the next time period for several periods. (Field: Finance, Derivative Pricing.)  
American Options: These are options contracts that can be exercised at any time up to the expiration date. (Field: Finance, Options Trading.)  
Dividend: A portion of a company's earnings that is paid to shareholders, or people that own that company's stock, on a periodic basis. (Field: Finance, Corporate Finance.)  
Greeks: In finance, Greeks are quantities representing the sensitivity of the price of derivatives such as options to a change in underlying parameters. (Field: Finance, Derivative Pricing.)  


Q2:  
American Options: As mentioned before, these are options contracts that can be exercised at any time up to the expiration date. (Field: Finance, Options Trading.)  
AAPL: AAPL is the stock ticker symbol for Apple Inc. on NASDAQ. (Field: Finance, Stock Market.)  
Risk Free Rate: As mentioned before, the theoretical rate of return of an investment with zero risk. (Field: Finance, Investment Theory.)  
Normal Distribution: As mentioned before, a probability distribution that is symmetric about the mean. (Field: Statistics, Probability Theory.)  
Simulation: In finance, simulation is often used to estimate the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. (Field: Statistics, Finance.)  
Mean, VaR and ES: As discussed before, these are measures of central tendency, risk of loss, and expected shortfall in the tail of a distribution respectively. (Field: Statistics, Finance, Risk Management.)  
Delta-Normal Method: This is a method used to calculate VaR in risk management. It assumes that all portfolio risk is attributable to market risk factors, which follow a normal distribution. (Field: Finance, Risk Management.)  


Q3:  
Fama French 3 factor return time series: Fama and French three-factor model is a stock pricing model that suggests that the expected return of a portfolio is explained by its exposure to market risk, its sensitivity to the value vs. growth factor, and its sensitivity to the size factor. (Field: Finance, Portfolio Theory.)  
Carhart Momentum time series: The Carhart four-factor model is an extension of the Fama–French three-factor model including a momentum factor for asset pricing of stocks. (Field: Finance, Portfolio Theory.)  
Expected Annual Return: The return that an investor expects to earn from an investment over a certain period of time in the future. (Field: Finance, Investment Theory.)  
Covariance Matrix: A covariance matrix is a square matrix giving the covariance between each pair of elements of a given random vector. (Field: Statistics, Multivariate Analysis.)  
Risk-Free Rate: As mentioned before, the theoretical rate of return of an investment with zero risk. (Field: Finance, Investment Theory.)  
Super-Efficient Portfolio: A portfolio that provides the highest expected return for a given level of risk or the lowest risk for a given expected return. (Field: Finance, Portfolio Theory.)  

### Knowledge domain list
Week 2:  
Skewness: Major - Statistics, Minor - Probability Distribution
Kurtosis: Major - Statistics, Minor - Probability Distribution
Biased: Major - Statistics, Minor - Estimation Theory
Hypothesis: Major - Statistics, Minor - Hypothesis Testing
OLS (Ordinary Least Squares): Major - Econometrics, Minor - Regression Analysis
Error Vector: Major - Econometrics, Minor - Regression Analysis
MLE (Maximum Likelihood Estimation): Major - Statistics, Minor - Estimation Theory
T Distribution: Major - Statistics, Minor - Probability Theory
Normality Assumption: Major - Econometrics, Minor - Regression Analysis
AR (Autoregressive) Process: Major - Econometrics, Minor - Time Series Analysis
MA (Moving Average) Process: Major - Econometrics, Minor - Time Series Analysis
ACF (Autocorrelation Function): Major - Econometrics, Minor - Time Series Analysis
PACF (Partial Autocorrelation Function): Major - Econometrics, Minor - Time Series Analysis

Week 3:  
Stock Returns: Major - Finance, Minor - Investment Analysis
ETF (Exchange-Traded Fund): Major - Finance, Minor - Investment Analysis
SPY: Major - Finance, Minor - Investment Analysis
Exponentially Weighted Covariance Matrix: Major - Econometrics, Minor - Time Series Analysis
PCA (Principal Component Analysis): Major - Statistics, Minor - Multivariate Analysis
Eigenvalue: Major - Mathematics, Minor - Linear Algebra
chol_psd(), near_psd() functions: Major - Mathematics, Minor - Linear Algebra
Higham's 2002 nearest psd correlation function: Major - Mathematics, Minor - Numerical Analysis
Frobenius Norm: Major - Mathematics, Minor - Linear Algebra
Multivariate Normal Simulation: Major - Statistics, Minor - Simulation Methods
PCA (Principal Component Analysis): Major - Statistics, Minor - Multivariate Analysis
Pearson Correlation: Major - Statistics, Minor - Correlation Analysis
Variance: Major - Statistics, Minor - Descriptive Statistics
Exponentially Weighted Variance: Major - Statistics, Minor - Time Series Analysis

Week 4  
Expected Value - Major Field: Statistics, Minor Field: Probability Theory
Standard Deviation - Major Field: Statistics, Minor Field: Descriptive Statistics
Price at Time 𝑡(𝑃𝑡) - Major Field: Finance, Minor Field: Financial Analysis
Price Returns - Major Field: Finance, Minor Field: Investment Analysis
𝑟𝑡 ~ 𝑁(0, 𝜎2) - Major Field: Statistics, Minor Field: Probability Theory
return_calculate() Function - Major Field: Finance, Minor Field: Investment Analysis
Arithmetic Returns - Major Field: Finance, Minor Field: Investment Analysis
VaR (Value at Risk) - Major Field: Finance, Minor Field: Risk Management
MLE fitted T Distribution - Major Field: Statistics, Minor Field: Probability Theory
AR(1) Model - Major Field: Statistics, Minor Field: Time Series Analysis
Historic Simulation - Major Field: Finance, Minor Field: Risk Management
Portfolio - Major Field: Finance, Minor Field: Portfolio Management
Exponentially Weighted Covariance - Major Field: Statistics, Minor Field: Time Series Analysis
VaR (Value at Risk) - Major Field: Finance, Minor Field: Risk Management

Week 5  
Normal Distribution - Major Field: Statistics, Minor Field: Probability Theory
Generalized T Distribution - Major Field: Statistics, Minor Field: Probability Theory
VaR (Value at Risk) - Major Field: Finance, Minor Field: Risk Management
ES (Expected Shortfall) - Major Field: Finance, Minor Field: Risk Management
Distribution PDFs - Major Field: Statistics, Minor Field: Probability Theory
Library for Risk Management - Major Field: Computer Science, Minor Field: Software Engineering
Covariance Estimation Techniques - Major Field: Statistics, Minor Field: Multivariate Analysis
Non PSD Fixes for Correlation Matrices - Major Field: Mathematics, Minor Field: Linear Algebra; Statistics
Simulation Methods - Major Field: Statistics, Minor Field: Finance
VaR calculation methods - Major Field: Finance, Minor Field: Risk Management
ES Calculation - Major Field: Finance, Minor Field: Risk Management
Test Suite - Major Field: Computer Science, Minor Field: Software Testing
Portfolio - Major Field: Finance, Minor Field: Portfolio Management
Generalized T model - Major Field: Statistics, Minor Field: Probability Theory
VaR (Value at Risk) - Major Field: Finance, Minor Field: Risk Management
ES (Expected Shortfall) - Major Field: Finance, Minor Field: Risk Management

Week 6  
Call Option - Major Field: Finance, Minor Field: Options Trading
Put Option - Major Field: Finance, Minor Field: Options Trading
Stock Price - Major Field: Finance, Minor Field: Stock Market
Option Expiration Date - Major Field: Finance, Minor Field: Options Trading
Risk-Free Rate - Major Field: Finance, Minor Field: Investment Analysis
Continuously Compounding Coupon - Major Field: Finance, Minor Field: Compound Interest
Implied Volatility - Major Field: Finance, Minor Field: Options Trading
Supply and Demand - Major Field: Economics
Strike Price - Major Field: Finance, Minor Field: Options Trading
AAPL_Options.csv - Major Field: Data Analysis, Minor Field: Options Trading
Portfolio - Major Field: Finance, Minor Field: Portfolio Management
problem3.csv - Major Field: Data Analysis, Minor Field: Portfolio Management
Underlying Values - Major Field: Finance, Minor Field: Investment Analysis
DailyPrices.csv - Major Field: Data Analysis, Minor Field: Stock Market
Log Returns - Major Field: Finance, Minor Field: Investment Analysis
Demean - Major Field: Statistics, Minor Field: Data Analysis
AR(1) Model - Major Field: Statistics, Minor Field: Time Series Analysis
VaR (Value at Risk) - Major Field: Finance, Minor Field: Risk Management
ES (Expected Shortfall) - Major Field: Finance, Minor Field: Risk Management

Week 7  
Stock Price - Major Field: Finance, Minor Field: Stock Market
Strike Price - Major Field: Finance, Minor Field: Options Trading
Risk-Free Rate - Major Field: Finance, Minor Field: Investment Theory
Continuously Compounding Coupon - Major Field: Finance, Minor Field: Investment Theory
GBSM (Generalized Black-Scholes Model) - Major Field: Finance, Minor Field: Derivative Pricing
Finite Difference Derivative Calculation - Major Field: Mathematics, Minor Field: Numerical Analysis
Call and Put - Major Field: Finance, Minor Field: Options Trading
Binomial Tree Valuation - Major Field: Finance, Minor Field: Derivative Pricing
American Options - Major Field: Finance, Minor Field: Options Trading
Dividend - Major Field: Finance, Minor Field: Corporate Finance
Greeks - Major Field: Finance, Minor Field: Derivative Pricing
American Options - Major Field: Finance, Minor Field: Options Trading
AAPL - Major Field: Finance, Minor Field: Stock Market
Risk Free Rate - Major Field: Finance, Minor Field: Investment Theory
Normal Distribution - Major Field: Statistics, Minor Field: Probability Theory
Simulation - Major Field: Statistics, Minor Field: Finance
Mean, VaR and ES - Major Field: Statistics, Minor Field: Risk Management
Delta-Normal Method - Major Field: Finance, Minor Field: Risk Management
Fama French 3 factor return time series - Major Field: Finance, Minor Field: Portfolio Theory
Carhart Momentum time series - Major Field: Finance, Minor Field: Portfolio Theory
Expected Annual Return - Major Field: Finance, Minor Field: Investment Theory
Covariance Matrix - Major Field: Statistics, Minor Field: Multivariate Analysis
Risk-Free Rate - Major Field: Finance, Minor Field: Investment Theory
Super-Efficient Portfolio - Major Field: Finance, Minor Field: Portfolio Theory

### Knowledge frequency list
#### Finance: 38/97 = 0.392(no)  
Risk Management: 9/38 = 0.237  
Investment Analysis: 8/38 = 0.211  
Options Trading: 7/38 = 0.184  
Investment Theory: 4/38 = 0.105  
Portfolio Management: 3/38 = 0.079  
Stock Market: 3/38 = 0.079  
Derivative Pricing: 3/38 = 0.079  
Financial Analysis: 1/38 = 0.026  
Corporate Finance: 1/38 = 0.026  
#### Statistics: 24/97 = 0.247(yes)  
Probability Theory: 8/24 = 0.333  
Risk Management: 3/24 = 0.125  
Probability Distribution: 2/24 = 0.083  
Estimation Theory: 2/24 = 0.083  
Multivariate Analysis: 2/24 = 0.083  
Descriptive Statistics: 2/24 = 0.083  
Hypothesis Testing: 1/24 = 0.042  
Correlation Analysis: 1/24 = 0.042  
Simulation Methods: 1/24 = 0.042  
Time Series Analysis: 1/24 = 0.042  
Data Analysis: 1/24 = 0.042  
#### Econometrics: 9/97 = 0.093(yes)  
Time Series Analysis: 6/9 = 0.667  
Regression Analysis: 3/9 = 0.333  
#### Mathematics: 6/97 = 0.062(no)  
Linear Algebra: 3/6 = 0.500  
Numerical Analysis: 2/6 = 0.333  
Statistics: 1/6 = 0.167  
#### Data Analysis: 3/97 = 0.031(half)  
Options Trading: 1/3 = 0.333  
Portfolio Management: 1/3 = 0.333  
Stock Market: 1/3 = 0.333 
#### Computer Science: 2/97 = 0.021(yes)  
Software Engineering: 1/2 = 0.500  
Software Testing: 1/2 = 0.500  
#### Economics: 1/97 = 0.010(yes)  
Supply and Demand: 1/1 = 1  

The willingness I want to take this course is: 0.247 + 0.093 + 0.021 + 0.010 + 0.031 = 0.402

#### Week 2 Frequency Ratio (Major Fields) Ranked High to Low:  
Statistics: 6/13 = 46.2%  
Econometrics: 7/13 = 53.8%  
(willingness = 1)
#### Week 3 Frequency Ratio (Major Fields) Ranked High to Low:  
Statistics: 6/14 = 42.9%  
Finance: 3/14 = 21.4%  
Mathematics: 4/14 = 28.6%  
Econometrics: 1/14 = 7.1%  
(willingness = 0.429 + 0.071 = 0.5)
#### Week 4 Frequency Ratio (Major Fields) Ranked High to Low:  
Statistics: 5/14 = 35.7%  
Finance: 9/14 = 64.3%  
(willingness = 0.357)
#### Week 5 Frequency Ratio (Major Fields) Ranked High to Low:  
Statistics: 4/16 = 25%  
Finance: 9/16 = 56.25%  
Mathematics: 1/16 = 6.25%  
Computer Science: 2/16 = 12.5%  
(willingness = 0.25 + 0.125 = 0.357)
#### Week 6 Frequency Ratio (Major Fields) Ranked High to Low:
Statistics: 2/19 = 10.5%  
Finance: 11/19 = 57.9%  
Data Analysis: 3/19 = 15.8%  
Economics: 1/19 = 5.3%  
Computer Science: 1/19 = 5.3%  
Mathematics: 1/19 = 5.3%  
(willingness = 0.105 + 0.079 + 0.053 + 0.053 = 0.29)
#### Week 7 Frequency Ratio (Major Fields) Ranked High to Low:  
Finance: 17/24 = 70.8%
Statistics: 4/24 = 16.7%
Mathematics: 1/24 = 4.2%
(willingness = 0.167)

Course Fintech545 covers a wide range of topics in finance and statistics, with a heavy emphasis on derivative pricing, options trading, risk management, and understanding of probability theory and distributions. The course likely involves a lot of numerical computation and simulation methods, given the repeated mentions of these topics.  
Upon completion of this course, you should be able to:  
Understand and apply key concepts in finance and statistics to financial technologies and risk management.  
Calculate option prices and Greeks using different models.  
Understand and calculate VaR and ES under various assumptions and models.  
Simulate price returns and generate covariance matrices.  
Fit data to different distributions and use those for simulations and risk assessments.  
Understand and apply different portfolio theories to find expected returns and construct optimal portfolios.  
