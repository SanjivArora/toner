from names import *

def calcDelta(x, field):
    #x = x.fillna(0)
    x=x[field]
    x1 = x.shift(-1)
    delta = x-x1
    return(delta)

def addDeltas(df, names):
    by_machine=df.groupby('Serial', group_keys=False)
    for name in names:
        res = by_machine.apply(partial(calcDelta, field=name))
        df[name+".delta"] = res

def addRates(df, names):
    for name in names:
        coldata = df[name+'.delta'] / df['RetrievedDateTime.delta.days']
        coldata = coldata.replace([np.inf, -np.inf], np.nan)
        # Use tolist() to ignore index
        df[name+".rate"] = coldata.tolist()
        
def addReplacements(df, names):
    by_machine=df.groupby('Serial', group_keys=False)
    for name in names:
        res = by_machine.apply(lambda x: x[name+'.delta']>0)
        df[name+".replaced"] = res
        
def addSumRate(df, names):
    data = df[names].divide(df['RetrievedDateTime.delta.days'], axis='rows')
    df[[name+".rate" for name in names]] = data
    
def addTonerPages(df, color='K'):
    def inner(x):
        replaced = x[f'Toner.{color}.replaced']
        vals = x[f'Pages.Previous.Toner.{color}'].where(replaced)
        # Write previous toner pages to all rows for respective previous bottles
        vals = vals.shift(1).fillna(method='ffill')
        return vals
    by_machine=df.groupby('Serial', group_keys=False)
    res = by_machine.apply(inner)
    #df[f'Pages.Toner.{color}'] = res['res'].tolist()
    df[f'Pages.Total.Toner.{color}'] = res
    return(res)

# Project number of pages per bottle using minimum nonzero toner% reading and calculate ratio with actual number of pages
def projectPages(df, color="K", threshold=30):
    def inner(x):
        min_idx = x[f'Toner.{color}'].where(x[f'Toner.{color}'] < threshold).where(x[f'Toner.{color}'] > 0).idxmin()
        # Direct equality with np.nan to check valid min_idx fails so test for expected type
        if not type(min_idx)==str:
            projection=np.nan
            ratio = np.nan
        else:
            toner = x.loc[min_idx, f'Toner.{color}']
            pages = x.loc[min_idx, f'Pages.Toner.{color}']
            projection = pages * 100/(100-toner)
            total_pages = x.loc[min_idx, f'Pages.Total.Toner.{color}']
            ratio = total_pages / projection
        # Record at the first reading for the toner bottle
        res = pd.DataFrame({
            f'Projected.Pages.{color}': projection,
            f'Projected.Pages.Ratio.{color}': ratio,
        }, index=[x.head(1).index])
        return(res)
    by_toner = df.groupby(f'TonerIndex.{color}', group_keys=False)
    res = by_toner.apply(inner).reset_index()
    fields = [f'Projected.Pages.{color}', f'Projected.Pages.Ratio.{color}']
    for field in fields:
        df.loc[res.idx, field] = res[field].tolist()
            
def projectCoverage(df, color="K", min_range=50):
    # Estimate coverage at start and end of the bottle using minimum and maximum nonzero toner% readings
    # TODO: Also calculate ratio of prospectively estimated end of bottle coverage to restrospectively estimated start of bottle coverare for next bottle
    # The prospective estimation measures the copier's estimate of the coverage on reaching 0% remaining toner, htre retrospective estimate is for actual coverage on replacement
    #
    # Coverage.Start.<color> is the cumulative coverage at the start of the bottle
    # Projected.Coverage.<color> is the projected coverage for this toner bottle (i.e. not cumulative)
    def inner(x):
        min_idx = x[f'Toner.{color}'].where(x[f'Toner.{color}'] > 0).idxmin()
        max_idx = x[f'Toner.{color}'].where(x[f'Toner.{color}'] > 0).idxmax()
        # Direct equality with np.nan to check valid min_idx fails so test for expected type
        if (not type(min_idx)==str) or (not type(max_idx)==str):
            cov_start = np.nan
            cov_projected = np.nan
        else:
            # Min and max refer to toner level
            toner_min = x.loc[min_idx, f'Toner.{color}']
            toner_max = x.loc[max_idx, f'Toner.{color}']
            cov_min = x.loc[min_idx, f'Coverage.{color}']
            cov_max = x.loc[max_idx, f'Coverage.{color}']
            #pages_min = x.loc[min_idx, f'Pages.Toner.{color}']
            #pages_max = x.loc[max_idx, f'Pages.Toner.{color}']
            toner_delta = toner_max - toner_min
            cov_delta = cov_min - cov_max
            #pages_delta = pages_min - pages_max
            if toner_delta < min_range or cov_delta <= 0:
                cov_start = np.nan
                cov_projected = np.nan
            else:
                cov_per_toner = cov_delta / toner_delta
                cov_start = cov_max - cov_per_toner * (100-toner_max)
                cov_projected = (cov_min - cov_start) + cov_per_toner * (100-toner_min)
        # Record at the first reading for the toner bottle
        res = pd.DataFrame({
            f'Coverage.Start.{color}': cov_start,
            f'Projected.Coverage.{color}': cov_projected,
        }, index=[x.head(1).index])
        return(res)
    by_toner = df.groupby(f'TonerIndex.{color}', group_keys=False)
    res = by_toner.apply(inner).reset_index()
    fields = [f'Coverage.Start.{color}', f'Projected.Coverage.{color}']
    for field in fields:
        df.loc[res.idx, field] = res[field].tolist()
        
# Calculate median by serial rather than by toner bottle, as machines with bad efficiency go through more toner
def medianCoverage(df, color, model):
    cov = f'Projected.Coverage.{color}'
    all_valid = df.loc[df.Model==model,['Serial', cov]].dropna()
    medians_for_serial = all_valid.groupby('Serial')[cov].median()
    res = medians_for_serial.dropna().median()
    return(res)

# Specific toner usage for coverage vs median for color and model
def addTonerRatio(df, color, model):
    med = medianCoverage(df, color, model)
    cov = f'Projected.Coverage.{color}'
    f = f'Toner.Usage.Ratio.{color}'
    df.loc[df.Model==model, f] = med / df.loc[df.Model==model, cov]
