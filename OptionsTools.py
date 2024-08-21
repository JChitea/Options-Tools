import math as mt
from scipy.stats import norm

def get_price_C(St,K,d1,d2, r, T, t):
    return (St*norm.cdf(d1))-(mt.exp(-1*r*(T-t))*K*norm.cdf(d2))

def get_price_P(St,K,d1,d2, r, T, t):
    return (K*mt.exp(-1*r*(T-t))*norm.cdf(-1*d2))-(norm.cdf(-1*d1)*St)

def get_d1(rfr, St, K, vari, T, t):
    return (mt.log(St/K)+((rfr+((vari**2)/2))*(T-t)))/(mt.sqrt(T-t))

def get_d2(d1, vari, T, t):
    return d1-(vari*mt.sqrt(T-t))

def delta_call(d1):
    return norm.cdf(d1)

def delta_put(d1):
    return d1-1

def norm_pdf(x):
    return (1/mt.sqrt(2*mt.pi))*mt.exp(-1*((x**2)/2))

def gamma(d1, St, T, t, vari):
    return (norm_pdf(d1))/(vari*St*mt.sqrt(T-t))

def vega(d1, St, T, t):
    return St*mt.sqrt(T-t)*norm_pdf(d1)

d1 = get_d1(0.06,97.70,100,0.85,0.19,0)
d2 = get_d2(d1,0.85,0.19,0)
print(get_price_C(97.70,100,d1,d2,0.06,0.19,0))