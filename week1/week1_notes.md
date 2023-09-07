### week 1 lecture note clarification
When we repeat the sampling process for 100 times, we fail to reject the hypothesis, we assume the kurtosis is unbiased, as given the sampling result we get, it is likely that this sample mean not deviates from the real mean kurtosis value;
When we repeat the sampling process for 1000 times, we reject the hypothesis, we assume the kurtosis is biased, as given the sampling result we get, it is not likely that this sample mean not deviates from the real mean kurtosis value; 

# Test the kurtosis function for bias in small sample sizes
d = Normal(0,1)
samples = 1000 # samples = 100
kurts = Vector{Float64}(undef,samples)
Threads.@threads for i in 1:samples
    kurts[i] = kurtosis(rand(d,100,000))
end
t = mean(kurts)/sqrt(var(kurts)/samples)
p = 2*(1 - cdf(TDist(samples-1),abs(t)))
println("p-value - $p")

**Changing 100 to 1000 concerns the change in statistical power, not n, changing 100,000 for the more is changing n**

When samples = 100, you fail to reject the null hypothesis that the kurtosis function is unbiased. This could be because with a smaller sample size, you have less statistical power to detect bias, even if it exists.
When samples = 1000, you reject the null hypothesis, suggesting that the kurtosis function is biased. This is likely due to the increased statistical power with a larger sample size, making it more likely to detect even subtle biases.
