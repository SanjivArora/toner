# Functionality for reporting on fleet history (e.g. pages) 

from multiprocessing import Pool

from importlib import reload
reload(derived)
reload(names)
reload(cloud)
reload(developer)
reload(process)
reload(visualization)
reload(predict_coverage)

from .developer import *
from .process import *
from .visualization import *
from .names import colors_norm


bucket='s3://ricoh-prediction-data/'

fs = [
    'Device Serial Number',
    'Customer Name/ID',
    'Model Name',
    'Acquisition Date (mm/dd/yyyy)', 
    'Acquisition Time',
    'Total',
    'B&W Total',
    'Color Total',
]
      
def readfile(x):
    res = pd.read_csv(bucket+x)
    res = res[fs]
    return res
    
with Pool() as pool:
    res_parts = pool.map(readfile, mrps)
    print("Done")
