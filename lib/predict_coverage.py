import random
import io
import datetime
import itertools
import traceback
import math
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
import boto3
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#import names, cloud, developer, process
#import names as names
from .developer import *
from .process import *
from .names import colors_norm
from .common import timed
from .predict_closed import find_d


def filterNaN(xs):
    return([x for x in xs if not np.isnan(x)])

def filterNeg(xs):
    return([x for x in xs if x >= 0])

def filterInf(xs):
    return([x for x in xs if x not in [np.inf, -np.inf]])

# For distributions not defined for 0
def enforceMinVal(xs, val=1e-1):
    return([x if x>val else val for x in xs])

def dropEmpty(xss):
    return [xs for xs in xss if len(xs) > 0]

def filterData(xs):
    return filterInf(filterNeg(filterNaN(xs)))

def calcDist(machine_hist, model_hist, fleet_hist, model_n=20, samples=5000, min_model_hist=1000):
    # Calculate a proportion of samples to come from the model vs the machine
    # sample from model and machine histories according to the ratio of model_n : len(model_hist)
    # With a model_n value of 40, a machine with 10 days of history will draw 80% of samples from model history and 20% from machine history
    # If there are fewer than min_model_days machine-days of history available for the model, draw the balance from the fleet history
    model_frac = model_n / (model_n + len(machine_hist))
    
    # Split model_frac into fleet_frac and model_frac components if there are not enough days of model history
    fleet_frac = 0
    if len(model_hist) < min_model_hist:
        split = (min_model_hist - len(model_hist)) / min_model_hist
        fleet_frac = model_frac * split
        model_frac = model_frac * (1 - split)

    model_s = int(np.ceil(model_frac * samples))
    fleet_s = int(np.ceil(fleet_frac * samples))
    machine_s = max(0, samples-model_s-fleet_s)
    #print(model_s, machine_s, fleet_s)
    
    # Sample with replacement
    machine_samples = random.choices(machine_hist, k=machine_s)
    model_samples = random.choices(model_hist, k=model_s)
    fleet_samples = random.choices(fleet_hist, k=fleet_s)
    
    return model_samples + machine_samples

def predictCoverage(dist, days, n=1000):
    if days==0:
        return [0]
    d = [np.sum(random.choices(dist, k=days)) for i in range(n)]
    return np.array(d)

# Use a rolling sum for fast approximate calculation
def predictCoverageFast(dist, days, n=100000):
    if days==0:
        return [0]
    r = np.random.choice(dist, n+days-1)
    s = pd.Series(r)
    d = s.rolling(days).sum()[(days-1):]
    return np.array(d)

def latestProjection(df, color='K'):
    field = f'Coverage.Per.Toner.{color}'
    by_serial = df[['Serial', 'RetrievedDate', field]].sort_values('RetrievedDate', ascending=False).groupby('Serial')
    return by_serial.apply(lambda x: x[field].bfill().iloc[0])

def imputedCoveragePerToner(fleet_df, model_df, color='K'):
    def inner(xs, ys):
        valid = xs.dropna()
        if len(valid):
            imputed = np.percentile(ys.dropna(), 20)
        else:
            imputed = np.nan
        xs = xs.fillna(imputed)
        return xs
    # If we don't have a measured coverage per toner, use the 20th percentile of the latest values for the model
    xs = latestProjection(model_df, color)
    xs = inner(xs, xs)
    # If this failed, use the 20th percentile of the latest values for the fleet
    ys = latestProjection(fleet_df, color)
    xs = inner(xs, ys)
    return xs
    
def makeCoveragePerTonerMapInner(x):
    model, model_df = x
    return (model, {c: imputedCoveragePerToner(prediction_df, model_df, c) for c in colors_norm})

@timed
def makeCoveragePerTonerMap():
    # Build this per model as we need to impute missing values with data from peer machines
    # {model: {color: imputed_cov_per_toner}}
    by_model = prediction_df.groupby('Model')
    with multiprocessing.Pool() as pool:
        res_parts = pool.map(makeCoveragePerTonerMapInner, by_model, chunksize=1)
    return dict(res_parts)

# {color: {serial: data}}
@timed
def makeFilteredDataMap(df):
    by_serial = df.sort_values('RetrievedDate', ascending=False).groupby('Serial')
    res = {}
    for c in colors_norm:
        ser_to_filtered = {n: filterData(d[f'Coverage.{c}.rate'].tolist()) for n, d in by_serial}
        res[c]=ser_to_filtered
    return res

def makeLatestToners(df):
    return {c: latestToner(df, c) for c in colors_norm}

def makeModelHistMap(df, filtered_data_map):
    return {
        m: {c: np.array(list(itertools.chain(*getFilteredData(m, c, filtered_data_map)))) for c in colors_norm}
        for m in df.Model.unique()
    }

# Generate whole-fleet history {color: coverage}
@timed
def makeFleetHistory(filtered_data_map):
    # {color: {serial: hist}} -> {color: hist}
    return {c: [y for x in [hist for hist in filtered_data_map[c].values()] for y in x] for c in colors_norm}
        
def getModel(ser):
    return ser[0:3]

# Get list of datasets for model and color
def getFilteredData(m, c, filtered_data_map):
    return [x for s, x in filtered_data_map[c].items() if getModel(s)==m]

def latestToner(df, color='K'):
    field = f'Toner.{color}'
    by_serial = df.sort_values('RetrievedDate', ascending=False).groupby('Serial')
    return by_serial.apply(lambda x: x[field].bfill().iloc[0])

# Find earliest day a monte carlo simulation indicates nth percentile cumulative coverage will result in toner out
def estimateDaysToZeroBruteForce(ser, cov_per_toner_map, filtered_data_map, model_hist_map, fleet_hist_map, latest_toners, color='K', cov_percentile=95):
    machine_hist = filtered_data_map[color][ser]
    per_toner = cov_per_toner_map[getModel(ser)][color][ser]
    latest_toner = latest_toners[color][ser]
    assert not pd.isna(latest_toner)
    cov_dist = calcDist(machine_hist, model_hist_map[getModel(ser)][color], fleet_hist_map[color])
    cov_remaining_est = latest_toner * per_toner
    for day in range(0, 101):
        cov_predicted = predictCoverageFast(cov_dist, day)
        cov = np.percentile(cov_predicted, cov_percentile)
        #print(f"{ser}:{day}:{cov_remaining_est}:{cov}")
        if cov >= cov_remaining_est:
            break
    return day

# Binary search for O(maxdays) -> O(log(maxdays)) speedup
def estimateDaysToZero(ser, cov_per_toner_map, filtered_data_map, model_hist_map, fleet_hist_map, latest_toners, color='K', cov_percentile=95, min_val=1, max_val=1000):
    machine_hist = filtered_data_map[color][ser]
    per_toner = cov_per_toner_map[getModel(ser)][color][ser]
    latest_toner = latest_toners[color][ser]
    assert not pd.isna(latest_toner)
    cov_dist = calcDist(machine_hist, model_hist_map[getModel(ser)][color], fleet_hist_map[color])
    cov_remaining_est = latest_toner * per_toner
    if cov_remaining_est == 0:
        return 0
    def inner(min_v, max_v):
        if max_v - min_v <= 1:
            return min_v
        test = (min_v+max_v) / 2
        test = int(test)
        cov_predicted = predictCoverageFast(cov_dist, test)
        cov = np.percentile(cov_predicted, cov_percentile)
        #print(f"{ser}:{test}:{int(cov_remaining_est)}:{int(cov)}")
        if cov >= cov_remaining_est:
            return inner(min_v=min_val, max_v=test)
        else:
            return inner(min_v=test, max_v=max_v)
    return inner(min_val, max_val)

# Closed form approximation to estimate days to zero
def estimateDaysToZeroClosedForm(ser, cov_per_toner_map, filtered_data_map, model_hist_map, fleet_hist_map, latest_toners, color='K', cov_percentile=95, min_val=1, max_val=1000):
    machine_hist = filtered_data_map[color][ser]
    per_toner = cov_per_toner_map[getModel(ser)][color][ser]
    latest_toner = latest_toners[color][ser]
    assert not pd.isna(latest_toner)
    cov_dist = calcDist(machine_hist, model_hist_map[getModel(ser)][color], fleet_hist_map[color])
    cov_remaining_est = latest_toner * per_toner
    if cov_remaining_est == 0:
        return 0
    mu=np.mean(cov_dist)
    sigma=np.std(cov_dist)
    gamma=scipy.stats.skew(cov_dist)
    kappa=scipy.stats.kurtosis(cov_dist)
    return find_d(1-(cov_percentile/100), cov_remaining_est, mu, sigma, gamma, kappa, max_days=max_val)


def makePredictionsInner(x, c, percentile=95, max_days=1000, f=estimateDaysToZeroClosedForm):
    (s, d) = x
    res=pd.DataFrame()
    print(f"Predicting {s} {c}")
    res.loc[s, 'Serial'] = s
    res.loc[s, 'Toner.Color'] = c
    data_date = d.RetrievedDate.max()
    res.loc[s, 'LatestData'] = data_date
    lag_days = (dt.datetime.today().date() - data_date).days
    res.loc[s, 'DataAge'] = lag_days
    toner = d[f'Toner.{c}'][0]
    assert not pd.isna(toner)
    res.loc[s, f'Toner.Percent'] = toner
    def predict(percentile=percentile, max_val=max_days*2):
        return f(
            s,
            prediction_cov_per_toner_map,
            prediction_filtered_data_map,
            prediction_model_hist_map,
            prediction_fleet_hist_map,
            prediction_latest_toners,
            color=c,
            cov_percentile=percentile,
            max_val=max_val,
        )
    # To produce an easily readible maximum value for the expected number of days to empty, predict well beyond max_days before subtracting lag then clip to max_days
    res.loc[s, f'Days.To.Zero.From.Today.Earliest.Expected'] = min(predict() - lag_days, max_days)
    res.loc[s, f'Days.To.Zero.From.Today.Expected'] = min(predict(percentile=50) - lag_days, max_days)
    return res

@timed
def makePredictionsForSerial(x, method='binary', percentile=95):
    # Closed form approximation
    if method=='closed':
        f=estimateDaysToZeroClosedForm
    # Binary search with monte carlo simulation
    elif method=='binary':
        f=estimateDaysToZero
    # Brute force search with monte carlo simulation
    elif method=='brute':
        f=estimateDaysToZeroBruteForce
    else:
        raise(Exception(f'Unknown method: {method}'))
    parts = []
    for c in colors_norm:
        try:
            part = makePredictionsInner(x, c, percentile, f=f)
            parts.append(part)
        except:
            print(f"Exception processing {c} for {x[0]}")
            traceback.print_exc()
    if not parts:
        return False
    else:
        return pd.concat(parts)

# Use global variables for shared data as workaround for limitations of multiprocessing module
# Note that this means that makePredictions() is not thread-safe.
prediction_df = None
prediction_cov_per_toner_map = None
prediction_filtered_data_map = None
prediction_latest_toners = None
prediction_model_hist_map = None
prediction_fleet_hist_map = None

@timed
def makePredictions(df, sers=None, method='binary', percentile=95):
    global prediction_df
    global prediction_cov_per_toner_map
    global prediction_filtered_data_map
    global prediction_latest_toners
    global prediction_model_hist_map
    global prediction_fleet_hist_map
    prediction_df = df
    prediction_cov_per_toner_map = makeCoveragePerTonerMap()
    prediction_filtered_data_map = makeFilteredDataMap(df)
    prediction_latest_toners = makeLatestToners(df)
    prediction_model_hist_map = makeModelHistMap(df, prediction_filtered_data_map)
    prediction_fleet_hist_map = makeFleetHistory(prediction_filtered_data_map)
    if sers is None:
        sers = df.Serial.unique()
    to_predict = df[df.Serial.isin(sers)]
    by_serial = to_predict.sort_values('RetrievedDate', ascending=False).groupby('Serial')
    with multiprocessing.Pool() as pool:
        args = [(x, method, percentile) for x in by_serial]
        res_parts = pool.starmap(makePredictionsForSerial, args, chunksize=1)
    successes = [x for x in res_parts if x is not False]
    print(f"{len(successes)} serials successfully processed, {len(res_parts)-len(successes)} failures")
    return pd.concat(successes)

def plotToner(s, color=None):
    fig = go.Figure()
    df = res[res.Serial==s].sort_values('RetrievedDate')
    if color is None:
        colors = colors_norm
    else:
        colors = [color]
    for c in colors:
        fig.add_trace(go.Scatter(
            x=df.RetrievedDate,
            y=df[f'Toner.{c}'],
            name=f'Toner.{c}',
        ))
        f=f'Toner.End.Status.{c}'
        toner_status = df[f].isin(['N']).astype('int') * 100
        fig.add_trace(go.Scatter(
            x=df.RetrievedDate,
            y=toner_status,
            name=f'Toner Near End - {c}',
        ))
        toner_status = df[f].isin(['E']).astype('int') * 100
        fig.add_trace(go.Scatter(
            x=df.RetrievedDate,
            y=toner_status,
            name=f'Toner At End - {c}',
        ))
    fig.update_layout(title=s)
    fig.show()

# Find days at zero toner
# df is a dataframe for a specific serial, sorted by retrieved date in descending order
def daysAtZero(df, color):
    f = f'Toner.{color}'
    s = df[f]
    if s[0] != 0:
        return 0
    positive = s[s!=0]
    if positive.size>0:
        idx = positive.index[0]
    else:
        idx = s.index[-1]
    delta = df.iloc[0].RetrievedDate - df.loc[idx].RetrievedDate
    return delta.days
  
def daysAtEnd(df, color, vals=['E']):
    f = f'Toner.End.Status.{color}'
    s = df[f]
    if s[0] not in vals:
        return 0
    positive = s[~s.isin(vals)]
    if positive.size>0:
        idx = positive.index[0]
    else:
        idx = s.index[-1]
    delta = df.iloc[0].RetrievedDate - df.loc[idx].RetrievedDate
    return delta.days

