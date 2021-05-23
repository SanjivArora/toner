import math
from functools import lru_cache, wraps

import numpy as np

from .common import timed

# Adapted from https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
def npCacheMap(*args, **kwargs):
    '''LRU cache implementation for functions whose FIRST parameter is a 2D numpy array'''

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(list(map(np.array, hashable_array)))
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator

# Bin midpoints
def binmeans(bins):
    return np.mean([bins[:-1], bins[1:]], axis=0)

# Find midpoint value for bin at given percentile for a numpy histogram
def histPercentile(h, pc=95):
    cs = np.cumsum(h[0])
    bin_idx = np.searchsorted(cs, np.percentile(cs, pc))
    return binmeans(h[1])[bin_idx]

# Calculate sum and weight for all combinations of numpy-style histogram bins. If remove_negligible_threshold is true, remove smallest and largest sums until we find a weight greater than negligible_threshold
def sumHistRvs(h1, h2, bins=100, remove_negligible_outliers=True, negligible_threshold=0.00001):
    xs = binmeans(h1[1]) + binmeans(h2[1]).reshape(-1,1)
    xs = xs.reshape(-1)
    ws = h1[0] * h2[0].reshape(-1,1)
    ws = ws.reshape(-1)

    if remove_negligible_outliers:
        xs, ws = zip(*sorted(zip(xs, ws)))
        #print(list(zip(xs,ws)))
        n = len(ws)
        min_i = 0
        for w in ws:
            if w > negligible_threshold/bins/n:
                break
            min_i+=1
        max_i = len(ws)
        for w in reversed(ws):
            if w > negligible_threshold/bins/n:
                break
            max_i-=1
        xs = xs[min_i:max_i]
        ws = ws[min_i:max_i]
    return np.histogram(xs, weights=ws, bins=bins, density=True)

# Return histogram of sums of <days>-length sequences from random variable with distribution represented by numpy histogram h
@npCacheMap(maxsize=1024)
#@timed
def hSampleSum(h, days, bins=32):
    if days==1:
        return h
    if days<1:
        raise Exception()
    a = math.floor(days/2)
    b = days-a
    return sumHistRvs(hSampleSum(h, a), hSampleSum(h, b), bins=bins)
