import random
import io
import datetime
import itertools
import traceback
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import boto3
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#import names, cloud, developer, process
#import names as names
from .developer import *
from .process import *
from .names import colors_norm


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

def calcDist(machine_hist, model_hist, model_n=20, samples=5000):
    # Calculate a proportion of samples to come from the model vs the machine
    # sample from model and machine histories according to the ratio of model_n : len(model_hist)
    # With a model_n value of 40, a machine with 10 days of history will draw 80% of samples from model history and 20% from machine history
    model_frac = model_n / (model_n + len(machine_hist))
    
    model_s = int(np.ceil(model_frac * samples))
    machine_s = samples-model_s
    #print(model_s, machine_s)
    
    # Sample with replacement
    model_samples = random.choices(model_hist, k=model_s)
    machine_samples = random.choices(machine_hist, k=machine_s)
    
    return model_samples + machine_samples

def predictCoverage(dist, days, n=1000):
    if days==0:
        return [0]
    d = [np.sum(random.choices(dist, k=days)) for i in range(n)]
    return np.array(d)

def latestProjection(df, color='K'):
    field = f'Coverage.Per.Toner.{color}'
    by_serial = df.sort_values('RetrievedDate', ascending=False).groupby('Serial')
    return by_serial.apply(lambda x: x[field].bfill().iloc[0])

def imputedCoveragePerToner(df, color='K'):
    field = f'Coverage.Per.Toner.{color}'
    # If we don't have a measured coverage per toner, use the 30th percentile of the latest values for the group
    xs = latestProjection(df, color)
    valid = xs.dropna()
    if len(valid):
        imputed = np.percentile(xs.dropna(), 20)
    else:
        imputed = np.nan
    xs = xs.fillna(imputed)
    return xs
    
def makeCoveragePerTonerMap(df):
    # Build this per model as we need to impute missing values with data from peer machines
    # {model: {color: imputed_cov_per_toner}}
    cov_per_toner_map = {
        m: {c: imputedCoveragePerToner(m_df, c) for c in colors_norm}
        for m, m_df in df.groupby('Model')
    }
    return cov_per_toner_map

# {color: {serial: data}}
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

def getModel(ser):
    return ser[0:3]

# Get list of datasets for model and color
def getFilteredData(m, c, filtered_data_map):
    return [x for s, x in filtered_data_map[c].items() if getModel(s)==m]

def latestToner(df, color='K'):
    field = f'Toner.{color}'
    by_serial = df.sort_values('RetrievedDate', ascending=False).groupby('Serial')
    return by_serial.apply(lambda x: x[field].bfill().iloc[0])

def estimateDaysToZeroBruteForce(ser, cov_per_toner_map, filtered_data_map, model_hist_map, latest_toners, color='K', cov_percentile=95):
    machine_hist = filtered_data_map[color][ser]
    per_toner = cov_per_toner_map[getModel(ser)][color][ser]
    latest_toner = latest_toners[color][ser]
    cov_dist = calcDist(machine_hist, model_hist_map[getModel(ser)][color])
    cov_remaining_est = latest_toner * per_toner
    for day in range(0, 101):
        cov_predicted = predictCoverage(cov_dist, day)
        cov = np.percentile(cov_predicted, cov_percentile)
        #print(f"{ser}:{day}:{cov_remaining_est}:{cov}")
        if cov >= cov_remaining_est:
            break
    return day

# Binary search for O(maxdays) -> O(log(maxdays)) speedup
def estimateDaysToZero(ser, cov_per_toner_map, filtered_data_map, model_hist_map, latest_toners, color='K', cov_percentile=95):
    machine_hist = filtered_data_map[color][ser]
    per_toner = cov_per_toner_map[getModel(ser)][color][ser]
    latest_toner = latest_toners[color][ser]
    cov_dist = calcDist(machine_hist, model_hist_map[getModel(ser)][color])
    cov_remaining_est = latest_toner * per_toner
    if cov_remaining_est == 0:
        return 0
    def inner(min_val=1, max_val=1000):
        if max_val - min_val <= 1:
            return min_val
        test = (min_val+max_val) / 2
        test = int(test)
        cov_predicted = predictCoverage(cov_dist, test)
        cov = np.percentile(cov_predicted, cov_percentile)
        #print(f"{ser}:{test}:{int(cov_remaining_est)}:{int(cov)}")
        if cov >= cov_remaining_est:
            return inner(min_val=min_val, max_val=test)
        else:
            return inner(min_val=test, max_val=max_val)
    return inner()

def makePredictionsInner(x, c, percentile=95):
    (s, d) = x
    res=pd.DataFrame()
    print(f"Predicting {s} {c}")
    res.loc[s, 'Serial'] = s
    res.loc[s, 'Toner.Color'] = c
    data_date = d.RetrievedDate.max()
    res.loc[s, 'LatestData'] = data_date
    lag_days = (dt.datetime.today().date() - data_date).days
    res.loc[s, 'DataAge'] = lag_days
    res.loc[s, f'Toner.Percent'] = d[f'Toner.{c}'][0]
    def predict(percentile=percentile):
       return estimateDaysToZero(s, prediction_cov_per_toner_map, prediction_filtered_data_map, prediction_model_hist_map, prediction_latest_toners, color=c)
    res.loc[s, f'Days.To.Zero.From.Today.Earliest.Expected'] = predict() - lag_days
    res.loc[s, f'Days.To.Zero.From.Today.Expected'] = predict(percentile=50) - lag_days
    return res

def makePredictionsForSerial(x, percentile=95):
    try:
        parts = map(lambda c: makePredictionsInner(x, c, percentile), colors_norm)
        return pd.concat(parts)
    except:
        print(f"Exception processing {x[0]}")
        traceback.print_exc()
        return False

# Use global variables for shared data as workaround for limitations of multiprocessing module
# Note that this means that makePredictions() is not thread-safe.
prediction_cov_per_toner_map = None
prediction_filtered_data_map = None
prediction_latest_toners = None
prediction_model_hist_map = None
def makePredictions(df):
    global prediction_cov_per_toner_map
    global prediction_filtered_data_map
    global prediction_latest_toners
    global prediction_model_hist_map
    prediction_cov_per_toner_map = makeCoveragePerTonerMap(df)
    prediction_filtered_data_map = makeFilteredDataMap(df)
    prediction_latest_toners = makeLatestToners(df)
    prediction_model_hist_map = makeModelHistMap(df, prediction_filtered_data_map)
    by_serial = df.sort_values('RetrievedDate', ascending=False).groupby('Serial')
    with multiprocessing.Pool() as pool:
        res_parts = pool.map(makePredictionsForSerial, by_serial)
    successes = [x for x in res_parts if x is not False]
    print("{len(successes)} serials successfully processed, {len(res_parts)-len(successes)} failures")
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
#         toner_status = df[f].isin(['N','E']).astype('int') * 100
#         fig.add_trace(go.Scatter(
#             x=df.RetrievedDate,
#             y=toner_status,
#             name=f'Toner At/Near End - {c}',
#         ))
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

