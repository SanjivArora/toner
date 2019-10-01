import multiprocessing
import multiprocessing.pool
import traceback
import time
from random import randint

from .names import *
from .derived import *
from .cloud import *
from .common import timed


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


# Use global variables for shared data as workaround for limitations of multiprocessing module
# Note that this means that processFile() is not thread-safe.
process_df = None
def applyColorSetInner(args):
    f, color = args
    return f(process_df, color)

def applyColorSet(f, colors):
    pool = multiprocessing.Pool(len(colors))        
    res_parts = pool.map(applyColorSetInner, [(f, color) for color in colors])
    res = pd.DataFrame()
    for part in res_parts:
        #print(part)
        duplicated = set(res.columns).intersection(part.columns)
        if duplicated:
            raise Exception(f"Duplicated columns: {', '.join(duplicated)}")
        res[part.columns]=part
    return res


def processFile(s3_url, show_fields=False, keep_orig=False, allow_missing=False, toner_stats=True):
    global process_df

    start_time = time.time()

    def status(text):
        now = time.time()
        delta_s = now - start_time
        print(f"{text} [{delta_s:.2f}s elapsed for {s3_url}]")

    status("Reading data from S3")
    p = readFromS3(s3_url)
    orig_fields = p.columns
    
    status("Normalizing field names")
    toner_names, cov_names = normalizeFields(p, show_fields, allow_missing)
    colors_matched = [c for c in colors_norm if f'Toner.{c}' in toner_names]

    if not colors_matched:
        raise Exception(f"No colors matched for {s3_url}")

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

    # Add delta and replacements
    times = ['RetrievedDate', 'RetrievedDateTime']
    to_add = list(toner_names + cov_names + times)
    delta_fields = [x+'.delta' for x in to_add]
    time_deltas = [x+'.delta' for x in times]
    replacement_fields = [x+'.replaced' for x in toner_names]

    status(f"Adding deltas, rates and replacement fields for {s3_url}")
    addDeltas(p, to_add)
    # Coerce values to timedeltas / NaT
    for d in time_deltas:
        p[d] = pd.to_timedelta(p[d])
    p['RetrievedDateTime.delta.days']=p['RetrievedDateTime.delta'] / datetime.timedelta(days=1)
    addRates(p, to_add)
    addReplacements(p, toner_names)
    
    p = sortReadings(p)
        
    if toner_stats:
        def apply(f):
            res = applyColorSet(f, colors_matched)
            p[res.columns] = res
            #print(p[res.columns])
            return p

        process_df = p

        status("Adding toner bottle indices")
        process_df = apply(indexToner)
            
        status("Adding final page count for current toner")
        process_df = apply(getTonerPages)
            
        status("Project pages for toner bottles and calculate ratios vs. actual pages")
        process_df = apply(projectPages)
            
        status("Project coverage for toner bottles and calculate ratios vs. estimated coverage")
        process_df = apply(projectCoverage)

        status("Add toner usage ratios")
        process_df = apply(getTonerRatio)
    
    status("Finished")
    return p

def summarizeTonerStats(p):
    print(f"Generating summary for {s3_url}")
    x=p[['Serial','RetrievedDateTime.delta.days']+replacement_fields].groupby('Serial')
    summary=x.sum()
    
    addSumRate(summary, replacement_fields)
    replacement_rates = [name+".rate" for name in replacement_fields]
    return p

def sortReadings(df):
    df.sort_values(['Serial', 'RetrievedDateTime'], inplace=True, ascending=False)
    return df

def indexToner(df, color='K'):
    prev = df.shift(-1)
    new_ser = df.Serial != prev.Serial
    new = new_ser | df[f'Toner.{color}.replaced']
    new_labels = df.index
    f = f'TonerIndex.{color}'
    res = new_labels.to_frame(name=f)
    res[f] = new_labels.where(new)
    res[f] = res[f].fillna(method='bfill')
    return res

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

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NonDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

@timed
def buildDataset(to_use, kwargs={}, num_procs=int(np.ceil(multiprocessing.cpu_count() / 2)), f=doProcessing):
    args = [(path, kwargs) for path in to_use]
    # Use non-daemonic pool as we wanted children to be able to fork
    with NonDaemonPool(num_procs) as pool:
        res_parts = pool.map(doProcessing, args, chunksize=1)
    res_parts = [x for x in res_parts if x is not None]
    print("Combining result parts")
    res = pd.concat(res_parts)
    return res
