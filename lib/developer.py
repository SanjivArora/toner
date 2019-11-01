import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime as dt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# Find readings for serials where the latest reading is no older than two months
# If field is specified, filter out rows where the value is nan
def hasRecentReadings(df, field=None, weeks=4*2):
    if field is None:
        field = 'RetrievedDateTime'
    x = df[['Serial', 'RetrievedDate', field]].dropna()
    recent_idx = x.RetrievedDate > dt.datetime.today().date() - dt.timedelta(weeks=weeks)
    sers = x.loc[recent_idx, 'Serial'].unique()
    filtered = df.loc[df.Serial.isin(sers)]
    return filtered

# Filter for latest dev unit with recent readings
def currentDevUnits(df, color):
    f = f"Developer.Replaced.{color}"
    df = hasRecentReadings(df, f'Toner.Usage.Ratio.{color}')
    gp = df[['Serial', f]].groupby(['Serial'], group_keys=False)
    idxs = gp.apply(lambda x: x[f]==np.max(x[f]))
    if len(idxs)==0:
        return df[0:0]
    else:
        filtered = df.loc[idxs]
        return filtered

# Find serials where readings for the current dev unit have:
# -The latest specific toner usage > mult*median
# -At least two elevated readings > 2*median
def currentBadDevs(df, color, cur=None, mult=None):
    if mult is None:
        mult = 2.5
    if cur is None:
        cur = currentDevUnits(df, color)
    sers = cur.Serial.unique()
    res = []
    for ser, df1 in cur.groupby(cur.Serial):
        vals = df1[f'Toner.Usage.Ratio.{color}'].dropna()
        latest_high = (vals.head(1) > mult).sum() == 1
        two_elevated = (vals > 2).sum() >= 2
        if latest_high and two_elevated:
            res.append(ser)
    return res

def plotCurrentBadDevs(df, color, model=None, log=True, mult=None):
    if model:
        df = df.loc[df.Model==model]
    cur = currentDevUnits(df, color)
    sers = currentBadDevs(df, color, cur, mult=mult)
    traces = []
    for ser in sers:
        data = cur.loc[cur.Serial==ser,['RetrievedDate', f'Toner.Usage.Ratio.{color}']].dropna()
        ydata = data [f'Toner.Usage.Ratio.{color}']
        if log:
            ydata = np.log(ydata)
        traces.append(go.Scatter(x=data['RetrievedDate'], y=ydata, name=ser))
    if traces:
        title = f'Toner Efficiency Outliers - {model or ""} {color}'
        ylabel = 'Toner Usage vs Normal'
        if log:
            ylabel += " (log)"
        layout = go.Layout(
            title=title,
            xaxis=dict(
                title='Date'
            ),
            yaxis=dict(
                title=ylabel
            ),
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.layout.update(showlegend=True)
        iplot(fig)
        return True
    else:
        print("No current bad dev units")
        return False
        
# Find serials with recent dev unit replacements
def recentDevReplacement(df, color, weeks=26):
    f = f"Developer.Replaced.{color}"
    recent_idx = df[f] > dt.datetime.today().date() - dt.timedelta(weeks=weeks)
    return df[recent_idx].Serial.unique()
    
def plotDevLifeCycle(df, color, model=None, log=True):
    if model:
        df = df[df.Model==model]
    #sers = currentBadDevs(df, color, df)
    
    # Find serials with dev replacements
    sers = recentDevReplacement(df, color)
    df = df[df.Serial.isin(sers)]
    
    by_serial = df.groupby('Serial')
    traces = []
    for ser, df1 in by_serial:
        # Only graph machines with history for several toner replacements 
        if df1[f'Toner.{color}.replaced'].sum() < 3:
           continue
        data = df1[['RetrievedDate', f'Developer.Rotation.{color}', f'Toner.Usage.Ratio.{color}']].dropna()
        ydata = data[f'Toner.Usage.Ratio.{color}']
        if log:
            ydata = np.log(ydata)
        traces.append(go.Scatter(x=data[f'Developer.Rotation.{color}'], y=ydata, name=ser, opacity=0.3))
    if traces:
        title = f'Def Lifecycle - {model or ""} {color}'
        ylabel = 'Toner Usage vs Normal'
        if log:
            ylabel += " (log)"
        layout = go.Layout(
            title=title,
            xaxis=dict(
                title='Dev '
            ),
            yaxis=dict(
                title=ylabel
            ),
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.layout.update(showlegend=True)
        iplot(fig)



#def plotRates(summary):
#    data = summary[replacement_rates]
#    for i in range(data.shape[1]):
#        plt.xlim((0,0.05))
#        sns.distplot(data.iloc[:,i].where(lambda x: x>=0).dropna(), color=color_display[i], bins=50)
#        plt.show()

#plot_cols = ['K', 'Y', 'M', "C"]
#plot_models = res.Model.unique()
#for m in sorted(plot_models):
#    for c in plot_cols:
#        plotCurrentBadDevs(selectTonerStats(res), c, m)
#
#
#for c in plot_cols:
#    sers = currentBadDevs(res, c)
#    print(f"Outliers for {c}:")
#    for ser in sorted(sers):
#        print(ser)
#    print()
#
#
#plot_cols = ['K', 'Y', 'M', "C"]
#plot_models = res.Model.unique()
#for m in sorted(plot_models):
#    for c in plot_cols:
#        plotDevLifeCycle(selectTonerStats(res), c, m)

