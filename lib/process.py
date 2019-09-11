import multiprocessing
import traceback

from .names import *
from .derived import *
from .cloud import *


def addFields(df, name_dict):
    for new, orig in name_dict.items():
        df[new] = df[orig]

def processFile(s3_url, show_fields=False):
    p = readFromS3(s3_url)
    names = p.columns.tolist()
    
    toner = findFields(names, '.*Toner.Bottle.%s.Remaining.Toner.(?!previous).*', 'Toner.%s')
    addFields(p, toner)
    toner_names = list(toner.keys())
    toner_raw = list(toner.values())
    
    pages = findFields(names, '.*Pages.Current.Toner.%s.(?!previous).*', 'Pages.Toner.%s')
    addFields(p, pages)
    
    prev_pages = findFields(names, '.*Pages.Current.Toner.%s.previous.*', 'Pages.Previous.Toner.%s')
    addFields(p, prev_pages)

    cov = findFields(names, '.*Pixel.Coverage.Accumulation.Coverage.%s.*', 'Coverage.%s')
    addFields(p, cov)
    
    dev_replacement = findFields(
        names,
        ['.*Replacement.Date.Developer.%s.*',
         '.*Unit.Replacement.Date.Dev.Unit.%s.*'],
         'Developer.Replaced.%s'
    )
    addFields(p, dev_replacement)
    
    #dev_rotation = findFields(
    #    names,
    #    ['.*(?<!Previous.Unit).PM.Counter.Rotation.Developer.%s.*',
    #     '.*Drive.Distance.Counter.%s_Developer.*'],
    #    'Developer.Rotation.%s',
    #    take_first=True
    #)
    #addFields(p, dev_rotation)


    if show_fields:
        print("Toner level fields:")
        print(toner)
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
    
    times = ['RetrievedDate', 'RetrievedDateTime']

    # Add fields for 
    to_add = list(toner_names + times)
    delta_fields = [x+'.delta' for x in to_add]
    time_deltas = [x+'.delta' for x in times]
    replacement_fields = [x+'.replaced' for x in toner_names]

    print(f"Adding deltas and replacement fields for {s3_url}")
    addDeltas(p, to_add)
    # Coerce values to timedeltas / NaT
    for d in time_deltas:
        p[d] = pd.to_timedelta(p[d])
    p['RetrievedDateTime.delta.days']=p['RetrievedDateTime.delta'] / datetime.timedelta(days=1)
    addReplacements(p, toner_names)
    
    print("Adding toner bottle indices")
    p = sortReadings(p)
    for color in colors_norm:
        p = indexToner(p, color)
        
    print("Adding final page count for current toner")
    for color in colors_norm:
        addTonerPages(p, color)
        
    print("Project pages for toner bottles and calculate ratios vs. actual pages")
    for color in colors_norm:
        projectPages(p, color)
        
    print("Project coverage for toner bottles and calculate ratios vs. estimated coverage")
    for color in colors_norm:
        projectCoverage(p, color)
        
    models = p.Model.unique()
    print("Add toner usage ratios")
    for color in colors_norm:
        for m in models: 
            addTonerRatio(p, color, m)
    
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


models = ['E15', 'E16', 'E17', 'E18', 'E19']
#models = ['E18', 'E19']
#models = ['E26']
#models = ['G70', 'G71', 'G72', 'G73', 'G74', 'G75']


def doProcessing(path, summary_rows_only=True):
    try:
        res = processFile(f"s3://{in_bucket_name}/{path}")
        if summary_rows_only:
            res = selectTonerStats(res)
    except:
        print(f"Exception processing {path}")
        traceback.print_exc()
        res=None
    return res

#to_use = [x for x in cs if x[0] in models]
#to_use = cs


def buildDataset(to_use, num_procs=int(np.ceil(multiprocessing.cpu_count() / 2))):
    with multiprocessing.Pool(num_procs) as pool:
        res_parts = pool.map(doProcessing, to_use, chunksize=1)
    res_parts = [x for x in res_parts if x is not None]
    print("Combining result parts")
    res = pd.concat(res_parts)
    return res



#sers = res.loc[res['Projected.Coverage.Y'] < 10000, 'Serial'].unique()
#sers

#toner = findFields(
#    names,
#    [#'.*Current.Toner.%s.(?!previous).*',
#     '.*Toner.Bottle.%s.Remaining.Toner.(?!previous).*'],
#    'Toner.%s'
#)
#prev_pages = findFields(names, '.*Pages.Current.Toner.%s.previous.*', 'Pages.Previous.Toner.%s')
#pages = findFields(names, '.*Pages.Current.Toner.%s.(?!previous).*', 'Pages.Toner.%s')
#coverage = findFields(names, '.*Pixel.Coverage.Accumulation.Coverage.%s.*', 'Coverage.%s')
##At least one model has two apparently identical counters for this (same name, different number). Use the first one if so.
#developer_rotation = findFields(
#    names,
#    ['.*(?<!Previous.Unit).PM.Counter.Rotation.Developer.%s.*',
#     '.*Drive.Distance.Counter.%s_Developer.*'],
#    'Developer.Rotation.%s',
#    take_first=True)
#
#all_fields = [
#    toner,
#    prev_pages,
#    pages,
#    coverage,
#    developer_rotation,
#]
#
#for d in all_fields:
#    for k, v in d.items():
#        print(f"{k}: {v}")
        
#fields = reduce(lambda x,y: x.update(y) or x, all_fields, {})

