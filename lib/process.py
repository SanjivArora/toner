import multiprocessing
import multiprocessing.pool
import traceback
import time
import gc

from . names import *
from . derived import *
from . cloud import *
from . common import timed


def addFields(df, name_dict):
    for new, orig in name_dict.items():
        df[new] = df[orig]

def normalizeFields(p, show_fields=False, allow_missing=False):
    names = p.columns.tolist()
    # Ensure the alphanumerical model IDs are interpreted as strings
    p.Model = p.Model.astype('str')
    
    toner = findFields(names, '.*Toner.Bottle.%s.Remaining.Toner.(?!previous).*','Toner.%s', allow_missing=allow_missing)
    addFields(p, toner)
    
    try:
        toner_bw =  findFields(names, '.*Toner.status.Percentage','Toner.K', allow_missing=allow_missing,colors=False) 

        addFields(p,toner_bw)
        toner.update(toner_bw)
    except:
        pass

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
         'Developer.Replacement.Date.Recorded.%s',
         allow_missing=True,
    )
    addFields(p, dev_replacement)
    # Convert 'yymmdd' representation of dev unit replacement dates to datetime
    for f in dev_replacement.keys():
        p[f] = pd.to_datetime(p[f], format='%y%m%d', errors='coerce')

    pcus = findFields(names, '.*Estimated.Usage.Rate.PCU.%s', 'PCU.Yield.%s', allow_missing=allow_missing)
    addFields(p, pcus)
    pcus_names = list(pcus.keys())

    total_pages = findFields(names, '.*Total.Total.PrtPGS.SP8.*', 'Pages', colors=False, allow_missing=allow_missing)
    addFields(p, total_pages)
    pages_names = list(total_pages.keys())

    copy_bw_pages = findFields(names, '.*Total.PrtPGS.Color.Mode.B.W.SP8.*', 'Pages.Copy.BW', colors=False, allow_missing=allow_missing)
    addFields(p, copy_bw_pages)
    pages_names += list(copy_bw_pages.keys())

    print_bw_pages = findFields(names, '.*Total.PrtPGS.Print.Color.Mode.B.W.SP8.*', 'Pages.Print.BW', colors=False, allow_missing=allow_missing)
    addFields(p, print_bw_pages)
    pages_names += list(print_bw_pages.keys())

    # Include jobs with pages as same requirements and semantically similar
    total_jobs = findFields(names, '.*Total.Total.Jobs.SP8.*', 'Jobs', colors=False, allow_missing=allow_missing)
    addFields(p, total_jobs)
    pages_names = list(total_jobs.keys())

    used_bottles = findFields(names, ['.*used.cartridge.Total.%s', 
                                      '.*Toner.Use.Count.%s'],'Toner.Bottles.Total.%s', allow_missing=allow_missing)
    addFields(p, used_bottles)
    used_bottles_names = list(used_bottles.keys())
    
    dev_rotation = findFields(
        names,
        ['.*(?<!Previous.Unit).PM.Counter.Rotation.Developer.%s.*',
         '.*Display.Distance.Dev.Unit.%s.SP7.942',
         '.*Drive.Distance.Counter.%s_Developer.*',
        ],
        'Developer.Rotation.%s',
        take_first=True,
        allow_missing=allow_missing,
    )
    addFields(p, dev_rotation)

    try:
        bw_dev_rotation= findFields(names, '.*Drive.Distance.Counter.Development.*','Developer.Rotation.K',allow_missing=allow_missing,colors=False)
        
        addFields(p,bw_dev_rotation)
        dev_rotation.update(bw_dev_rotation)
    except:
        pass

    dev_rotation_names = list(dev_rotation.keys())

    # Always allow missing for toner call threshold as this comes in mutually exclusive variants
    toner_call_threshold = findFields(names, '.*Toner.Call.Threshold.SP5.*', 'Toner.Call.Threshold', colors=False, allow_missing=True)
    addFields(p, toner_call_threshold)
    toner_call_threshold_colored = findFields(names, '.*Toner.Call.Threshold.%s.*', 'Toner.Call.Threshold.%s', colors=['K', 'CMY'], allow_missing=True)
    addFields(p, toner_call_threshold_colored)
    toner_call_timing = findFields(names, '.*Toner.Call.Timing.*', 'Toner.Call.Timing', colors=False, allow_missing=allow_missing)
    addFields(p, toner_call_timing)


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
        print("Total Pages:")
        print(total_pages)
        print("Used Bottles:")
        print(used_bottles)
        print("Dev unit rotations:")
        print(dev_rotation)
        print("PCUs:")
        print(pcus)

    return toner_names, cov_names, pages_names, used_bottles_names, dev_rotation_names, pcus_names


# Use global variables for shared data as workaround for limitations of multiprocessing module
# Note that this means that processFile() is not thread-safe.
process_df = None
def applyColorSetInner(args):
    f, color = args
    return f(process_df, color)

def applyColorSet(f, colors):
    with multiprocessing.Pool(len(colors)) as pool:
        res_parts = pool.map(applyColorSetInner, [(f, color) for color in colors], chunksize=1)
    res = pd.DataFrame()
    for part in res_parts:
        #print(part)
        duplicated = set(res.columns).intersection(part.columns)
        if duplicated:
            raise Exception(f"Duplicated columns: {', '.join(duplicated)}")
        res = pd.concat([res, part], axis=1, sort=False)
    return res

def processFile(s3_url, show_fields=False, keep_orig=False, allow_missing=False, toner_stats=True, nz_only=False, in_bucket=None, skip_processing=False, require_color_match=True):
    global process_df

    start_time = time.time()

    # This is memory intensive, so garbage collect to free up any lingering allocations
    gc.collect()

    def status(text):
        now = time.time()
        delta_s = now - start_time
        print(f"{text} [{delta_s:.2f}s elapsed for {s3_url}]")

    status("Reading data from S3")
    p = readFromS3(s3_url)

    # At time of writing the RNZ Falcon files contain data for both AU and NZ machines. This is a workaround to restrict the result to NZ machines where required.
    # TODO: implement by checking if serial is in the regional MRP file rather than using exported NZ DB content (this should be general as well as more current)
    if nz_only:
        p = p[p.Serial.isin(process_nz_sers)]

    if skip_processing:
        return p

    orig_fields = p.columns
    
    status("Normalizing field names")
    toner_names, cov_names, pages_names, used_bottles_names, dev_rotation_names, pcus_names = normalizeFields(p, show_fields, allow_missing)
    colors_matched = [c for c in colors_norm if f'Toner.{c}' in toner_names]
    print(colors_matched)
    if require_color_match and not colors_matched:
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

    # Derive total BW and color page counts
    print("Adding page count stats")
    p['Pages.BW'] = p['Pages.Print.BW'] + p['Pages.Copy.BW']
    p['Pages.Color'] = p['Pages'] - p['Pages.BW']
    pages_names += ['Pages.BW', 'Pages.Color']

    # Add delta and replacements
    times = ['RetrievedDate', 'RetrievedDateTime']
    to_add = list(toner_names + cov_names + times + pages_names + used_bottles_names + dev_rotation_names + pcus_names)
    time_deltas = [x+'.delta' for x in times]

    status(f"Adding deltas, rates and replacement fields for {s3_url}")
    addDeltas(p, to_add)
    # Coerce values to timedeltas / NaT
    for d in time_deltas:
        p[d] = pd.to_timedelta(p[d])
    p['RetrievedDateTime.delta.days']=p['RetrievedDateTime.delta'] / datetime.timedelta(days=1)
    addRates(p, to_add)
    addReplacements(p, toner_names)
    # Toner counts down passage of time, PCUs count up - so a negative delta indicates a replacement
    addReplacements(p, pcus_names, negative=True)
    
    process_df = sortReadings(p)
        
    if colors_matched and toner_stats:
        def apply(f):
            global process_df
            res = applyColorSet(f, colors_matched)
            process_df = pd.concat([process_df, res], axis=1, sort=False)

        status("Adding toner bottle indices")
        apply(indexToner)
        if pcus_names:
            status("Adding PCU indices")
            apply(indexPCU)
            
        status("Adding final page count for current toner")
        apply(getTonerPages)
        
        status("Deriving dev unit replacement dates")
        apply(deriveDevReplacements)
        apply(selectDevReplacementDate)
            
        status("Project pages for toner bottles and calculate ratios vs. actual pages")
        apply(projectPages)
            
        status("Project coverage for toner bottles and calculate ratios vs. estimated coverage")
        apply(projectCoverage)

        status("Add toner usage ratios")
        apply(getTonerRatio)
    
    res = process_df
    process_df = None
    status("Finished")
    return res 

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

def indexPCU(df, color='K'):
    prev = df.shift(-1)
    new_ser = df.Serial != prev.Serial
    new = new_ser | df[f'PCU.Yield.{color}.replaced']
    new_labels = df.index
    f = f'PCUIndex.{color}'
    res = new_labels.to_frame(name=f)
    res[f] = new_labels.where(new)
    res[f] = res[f].fillna(method='bfill')
    return res

def deriveDevReplacements(df, color='K'):
    rotation = f'Developer.Rotation.{color}'
    prev = df.shift(-1)
    new = df[rotation] < prev[rotation]
    new[-1] = True
    dates = pd.to_datetime(df.RetrievedDate)
    f = f'Developer.Replacement.Date.Derived.{color}'
    res = dates.to_frame(name=f)
    res[f] = dates.where(new)
    res[f] = res[f].fillna(method='bfill')
    return res

def selectDevReplacementDate(df, color='K'):
    priorities = [
        f'Developer.Replacement.Date.Recorded.{color}',
        f'Developer.Replacement.Date.Derived.{color}',
    ]
    for f in priorities:
        if f in df.columns:
            df[f'Developer.Replacement.Date.{color}'] = df[f]
            break
    return df[[f'Developer.Replacement.Date.{color}']]

# Take only rows with summary stats for toner bottles
def selectTonerStats(df):
    colnames = df.columns[df.columns.str.contains('^Projected.Pages.*')]
    cols = df[colnames].dropna(how='all')
    res = df.loc[cols.index]
    return res

def doProcessing(args, summary_rows_only=False):
    path, kwargs = args
    try:
        bucket_name = kwargs.get('in_bucket', in_bucket_name)
        res = processFile(f"s3://{bucket_name}/{path}", **kwargs)
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

# Global to make NZ serial list available for children - TODO: replace with MRP data
process_nz_sers = None

@timed
def buildDataset(to_use, kwargs={}, num_procs=None, f=doProcessing):
    if num_procs is None:
        num_procs = int(np.ceil(multiprocessing.cpu_count() / 4))
    global process_nz_sers
    ser_df = pd.read_csv('s3://ricoh-prediction-misc/mif/current/customer.csv')
    process_nz_sers = ser_df.SerialNo

    args = [(path, kwargs) for path in to_use]
    # Use non-daemonic pool as we want children to be able to fork
    with NonDaemonPool(num_procs) as pool:
        res_parts = pool.map(doProcessing, args, chunksize=1)
    res_parts = [x for x in res_parts if x is not None]
    print("Combining result parts")
    res = pd.concat(res_parts)
    return res
