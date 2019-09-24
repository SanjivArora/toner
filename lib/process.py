import multiprocessing
import traceback

from .names import *
from .derived import *
from .cloud import *


def addFields(df, name_dict):
    for new, orig in name_dict.items():
        df[new] = df[orig]

def normalizeFields(p, show_fields=False, allow_missing=False):
    names = p.columns.tolist()
    # Ensure the alphanumerical model IDs are interpreted as strings
    p.Model = p.Model.astype('str')
    
    toner = findFields(names, '.*Toner.Bottle.%s.Remaining.Toner.(?!previous).*', 'Toner.%s', allow_missing=allow_missing)
    addFields(p, toner)
    toner_names = list(toner.keys())
    
    toner_end = findFields(names, '.*Toner.Bottle.%s.End.SP7.*', 'Toner.End.Status.%s', allow_missing=allow_missing)
    addFields(p, toner_end)

    toner_end_date = findFields(names, '.*Toner.Bottle.%s.End.Date.*', 'Toner.End.Date.%s', allow_missing=allow_missing)
    addFields(p, toner_end_date)

    pages = findFields(names, '.*Pages.Current.Toner.%s.(?!previous).*', 'Pages.Toner.%s', allow_missing=allow_missing)
    addFields(p, pages)
    
    prev_pages = findFields(names, '.*Pages.Current.Toner.%s.previous.*', 'Pages.Previous.Toner.%s', allow_missing=allow_missing)
    addFields(p, prev_pages)

    cov = findFields(names, '.*Pixel.Coverage.Accumulation.Coverage.%s.*', 'Coverage.%s', allow_missing=allow_missing)
    addFields(p, cov)
    cov_names = list(cov.keys())
    
    dev_replacement = findFields(
        names,
        ['.*Replacement.Date.Developer.%s.*',
         '.*Unit.Replacement.Date.Dev.Unit.%s.*'],
         'Developer.Replaced.%s',
         allow_missing=allow_missing,
    )
    addFields(p, dev_replacement)
    
    #dev_rotation = findFields(
    #    names,
    #    ['.*(?<!Previous.Unit).PM.Counter.Rotation.Developer.%s.*',
    #     '.*Drive.Distance.Counter.%s_Developer.*'],
    #    'Developer.Rotation.%s',
    #    take_first=True,
    #    allow_missing=allow_missing,
    #)
    #addFields(p, dev_rotation)

    if show_fields:
        print("Toner level fields:")
        print(toner)
        print("Toner end status:")
        print(toner_end)
        print("Toner end date:")
        print(toner_end_date)
        print("Current toner pages:")
        print(pages)
        print("Previous toner pages:")
        print(prev_pages)
        print("Coverage:")
        print(cov)  
        print("Dev unit replacement:")
        print(dev_replacement)
        print("Dev unit rotations:")
        print(dev_rotation)
    return toner_names, cov_names

def processFile(s3_url, show_fields=False, keep_orig=False, allow_missing=False, toner_stats=True):
    p = readFromS3(s3_url)
    orig_fields = p.columns
    
    toner_names, cov_names = normalizeFields(p, show_fields, allow_missing)
    colors_matched = [c for c in colors_norm if f'Toner.{c}' in toner_names]

    # Add delta and replacements
    times = ['RetrievedDate', 'RetrievedDateTime']
    to_add = list(toner_names + cov_names + times)
    delta_fields = [x+'.delta' for x in to_add]
    time_deltas = [x+'.delta' for x in times]
    replacement_fields = [x+'.replaced' for x in toner_names]

    print(f"Adding deltas, rates and replacement fields for {s3_url}")
    addDeltas(p, to_add)
    # Coerce values to timedeltas / NaT
    for d in time_deltas:
        p[d] = pd.to_timedelta(p[d])
    p['RetrievedDateTime.delta.days']=p['RetrievedDateTime.delta'] / datetime.timedelta(days=1)
    addRates(p, to_add)
    addReplacements(p, toner_names)
    
    if toner_stats:
        print("Adding toner bottle indices")
        p = sortReadings(p)
        for color in colors_matched:
            p = indexToner(p, color)
            
        print("Adding final page count for current toner")
        for color in colors_matched:
            addTonerPages(p, color)
            
        print("Project pages for toner bottles and calculate ratios vs. actual pages")
        for color in colors_matched:
            projectPages(p, color)
            
        print("Project coverage for toner bottles and calculate ratios vs. estimated coverage")
        for color in colors_matched:
            projectCoverage(p, color)

        models = p.Model.unique()
        print("Add toner usage ratios")
        for color in colors_matched:
            for m in models: 
                addTonerRatio(p, color, m)

    if not keep_orig:
        print("Dropping original fields")
        to_use = set(p.columns).difference(orig_fields)
        # Keep these regardless
        to_use.update([
            'Serial',
            'Model',
            'RetrievedDate',
            'RetrievedDateTime',
            'FileDate',
        ])
        p = p[to_use]
    
    return(p)

def summarizeTonerStats(p):
    print(f"Generating summary for {s3_url}")
    x=p[['Serial','RetrievedDateTime.delta.days']+replacement_fields].groupby('Serial')
    summary=x.sum()
    
    addSumRate(summary, replacement_fields)
    replacement_rates = [name+".rate" for name in replacement_fields]
    return(p)

def sortReadings(df):
    df.sort_values(['Serial', 'RetrievedDateTime'], inplace=True, ascending=False)
    return(df)

def indexToner(df, color='K'):
    prev = df.shift(-1)
    new_ser = df.Serial != prev.Serial
    new = new_ser | df[f'Toner.{color}.replaced']
    new_labels = df.index
    df[f'TonerIndex.{color}'] = new_labels.where(new)
    df[f'TonerIndex.{color}'] = df[[f'TonerIndex.{color}']].fillna(method='bfill')
    return(df)

# Take only rows with summary stats for toner bottles
def selectTonerStats(df):
    colnames = df.columns[df.columns.str.contains('^Projected.Pages.*')]
    cols = df[colnames].dropna(how='all')
    res = df.loc[cols.index]
    return res

def doProcessing(args, summary_rows_only=False):
    path, kwargs = args
    try:
        res = processFile(f"s3://{in_bucket_name}/{path}", **kwargs)
        if summary_rows_only:
            res = selectTonerStats(res)
    except:
        print(f"Exception processing {path}")
        traceback.print_exc()
        res=None
    return res

def buildDataset(to_use, kwargs={}, num_procs=int(np.ceil(multiprocessing.cpu_count() / 2)), f=doProcessing):
    args = [(path, kwargs) for path in to_use]
    with multiprocessing.Pool(num_procs) as pool:
        res_parts = pool.map(doProcessing, args, chunksize=1)
    res_parts = [x for x in res_parts if x is not None]
    print("Combining result parts")
    res = pd.concat(res_parts)
    return res
