import scipy
import math

# Analytical approach to estimating number of days remaining.
# See John's excellent write-up for a detailed explanation.

def d_variance(d, mu, sigma):
    return d*sigma**2 + d**2*mu**2

def kappa_d(d, mu, sigma, gamma, kappa):
    return d**2*sigma**4*kappa / d_variance(d, mu, sigma)**2 \
           + 4*d**2*gamma*mu*sigma**3 / d_variance(d, mu, sigma)**2 \
           + 6*d**3*mu**2*sigma**3 / d_variance(d, mu, sigma)**2 \
           + d**4*mu**4 / d_variance(d, mu, sigma)**2

# pixels remaining, days, mean, std deviation, skew, kurtosis
def calc(R, d, mu, sigma, gamma=0, kappa=3):
    d_sigma = math.sqrt(d_variance(d, mu, sigma))
    return R / d_sigma - R/(24*d_sigma)*(kappa_d(d, mu, sigma, gamma, kappa) - 3) \
           * ((R/d_sigma)**2 - 3)

# Find target value for percentage chance of running out of toner
def target_for_alpha(alpha=0.05):
    norm=scipy.stats.norm(0,1)
    return norm.ppf(1-alpha)
    
def find_d(alpha, R, mu, sigma, gamma=0, kappa=3, max_days=1000):
    target = target_for_alpha(alpha)
    for d in range(1,max_days):
        if calc(R, d, mu, sigma, gamma, kappa) < target:
            break
    return d
