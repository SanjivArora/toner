import datetime

import numpy as np
import plotly
import plotly.graph_objs as go

from .names import colors_norm


def plotAtRemoteLatency(df, max_days=7, days_history=None):
    if days_history:
        start = datetime.datetime.today() - datetime.timedelta(days=days_history)
        df = df[df.FileDate > start.date()]

    dates = []
    latencies = []
    counts = []

    for d, df1 in df.groupby('FileDate'):
        #print(d)
        ls = df1.FileDate - df1.RetrievedDate
        ls = ls.apply(lambda x: x.days)
        freqs = np.histogram(
            ls.clip(upper=max_days),
            range=(0, max_days),
            bins=max_days+1,
        )[0]
        dates += [d]*(max_days+1)
        latencies += range(max_days+1)
        counts += list(freqs)
    fig = go.Figure(data=go.Heatmap(
            x=dates,
            y=latencies,
            z=counts,
            colorscale='Viridis',
            #colorscale='Electric'
    ))
    fig.update_layout(
        title='RNZ F@lcon Data Internal Latency',
        #xaxis_nticks=36
    )
    fig.show()

def plotToner(df, s, color=None):
    fig = go.Figure()
    df = df[df.Serial==s].sort_values('RetrievedDate')
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
