import scipy
import math

# Analytical approach to estimating number of days remaining.
# See John's excellent write-up for a detailed explanation.

def variance_bar(mu, sigma):
    return sigma**2 + mu**2

def kappa_bar(mu, sigma, gamma, kappa):
    return sigma**4*kappa / variance_bar(mu, sigma)**2 \
           + 4*gamma*mu*sigma**3 / variance_bar(mu, sigma)**2 \
           + 6*3*mu**2*sigma**2 / variance_bar(mu, sigma)**2 \
           + 4*mu**4 / variance_bar(mu, sigma)**2

def f(x, mu, sigma, gamma, kappa):
    sigma_bar = math.sqrt(variance_bar(mu, sigma))
    return x / sigma_bar \
           - (x / (24*sigma_bar)) \
           * ( \
              kappa_bar(mu, sigma, gamma, kappa) - 3 \
              * (x**2/variance_bar(mu, sigma) - 3) \
             )

def phi_d(d, x, mu, sigma, gamma, kappa):
    norm=scipy.stats.norm(0,1)
    return 2 * norm.cdf(f(x, mu, sigma, gamma, kappa)) / d**2 \
           - 1 / d**2 \
           + (1 - 1/d**2) * norm.cdf((x - mu) * math.sqrt(d) / sigma)
    
def find_d(alpha, R, mu, sigma, gamma=0, kappa=3, max_days=1000):
    target = 1 - alpha 
    for d in range(1,max_days):
        if phi_d(d, R/d, mu, sigma, gamma, kappa) > target:
            break
    return d
