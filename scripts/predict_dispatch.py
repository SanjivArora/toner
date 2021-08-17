#!/usr/bin/env python3

import boto3
import os
import sys
import traceback
import datetime as dt
import re
import time
import xml
import xml.dom
import xml.dom.minidom
import xml.etree.ElementTree as ET
import pytz


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import lib
from lib import names, cloud, developer, process
from lib.developer import *
from lib.process import *
from lib.predict_coverage import *
from lib.names import colors_norm
     

################################################################################
# Config
################################################################################

# Testing mode - append '-test' to prefixes in S3
testing = False

# Set to limit the number of processes to use to build dataset. Default is the number of physical cores.
build_dataset_procs = None

models = None
#models = ['E19']
#models = ['308']
#models = ['E21']

# Predict every machine, regardless of whether the serial is in the Auckland University list
predict_all=False
#predict_all=True

# Map accounts to start dates
accounts={
    # Initial trial
    64121:"20210709", #Cashmere
    63094:"20210709", #City Care
    9014:"20210709", #Ricoh head office
    # Expanded trial
    32343:"20210818", #MOJ
    36171:"20210818", #Corrections
    10138:"20210818", #Visy Board
}

initial_toner_min = 15

dispatch_bucket='ricoh-prediction-dispatch'
# Canonical orders: flat <toner index>.<color>
# Where <toner index> is <serial>-<date>
canonical_orders_prefix = 'orders-canonical'
# Orders for S21 import: <date>/Predictive-RNZ-ASP-<datetimestamp>.xml
orders_prefix = 'orders'
# Serials for predictive-only toner dispatch: <date>/PredictiveSerialNumbers.xml
predictive_serials_prefix = 'predictive-serials'

threshold_days=14
max_data_age=21


color_to_call_type = {
    'K': 'TBL',
    'Y': 'TYE',
    'M': 'TMA',
    'C': 'TCY',
}

if testing:
    testing_suffix = '-test'
else:
    testing_suffix = ''

################################################################################
# Functions
################################################################################

def predSerialXML(preds):
    root = ET.Element('PredictiveSerialNumbers');

    serials = []
    for i,row in preds.iterrows():
        row_dict = {k:str(v) for k,v in row.to_dict().items()}
        serials.append(row_dict['Serial'])
        
    serials = set(serials)
    for ser in serials:
        item = ET.SubElement(root, 'SerialNumber')
        item.text = ser

    return root

def orderName(ts):
    # Timestamp with milliseconds
    order_ts = ts.strftime("%Y%m%d%H%M%S%f")[:-3]
    return f"Predictive-RNZ-ASP-{order_ts}"
    
def orderXML(ser, model, customer, color, ts):
    order_name=orderName(ts)
    # Format as required for datetime in toner order XML
    xml_ts=ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    
    root = ET.Element('ARemoteMessage')

    header = ET.SubElement(root, 'ARemoteHeader')
    ET.SubElement(header, 'ARemoteID').text=order_name
    ET.SubElement(header, 'MessageCode').text="4"
    ET.SubElement(header, 'MessageOrigin').text="P"
    ET.SubElement(header, 'IssueDateTime').text=xml_ts
    ET.SubElement(header, 'MachineSN').text=ser
    # All data is from Ricoh AtRemote, so with very few exceptions Ricoh machines. We aren't likely to be able to generate orders for any non-Ricoh machines.
    ET.SubElement(header, 'VendorName').text="Ricoh"
    ET.SubElement(header, 'ModelName').text=model
    ET.SubElement(header, 'CustomerID').text=str(customer)
    ET.SubElement(header, 'MessageCode').text="4"

    content = ET.SubElement(root, 'ARemoteHeader')
    call = ET.SubElement(content, 'SupplyCall')
    ET.SubElement(call, 'CallType').text=color_to_call_type[color]
    ET.SubElement(call, 'OccurrenceDateTime').text=xml_ts

    return root

# Return utf-8 encoded XML with pretty formatting
def prettyXML(e):
    s=ET.tostring(e)

    x = xml.dom.minidom.parseString(s)  # or xml.dom.minidom.parseString(xml_string)
    xml_pretty = x.toprettyxml(encoding="utf-8")
    return xml_pretty

def pprintXML(e):
    print(prettyXML(e).decode('utf-8'))

def c_to_indices_map(c):
    return res[['Serial',f'TonerIndex.{c}']].drop_duplicates().groupby(['Serial']).apply(lambda x: x[f'TonerIndex.{c}'].sort_values(ascending=False).unique())

def process_pred(p):
    ts = dt.datetime.now()
    toner_indices = indices_maps[p['Toner.Color']][p.Serial]
    order = orderXML(p.Serial, p.Model, p.Customer, p['Toner.Color'], ts)
    
    #TODO: Handle case of toner being replaced before we order, e.g. very large burst of printing or premature replacement
    # Need to track first day of predictive ordering per machine (newly added machines, etc) - we don't
    # want to order for each historical toner after system start date when adding a new copier.
    
    if p['Days.To.Zero.From.Today.Earliest.Expected']>threshold_days or p['DataAge']>max_data_age:
        return
     
    canonical_key = f"{canonical_orders_prefix}{testing_suffix}/{toner_indices[0]}.{p['Toner.Color']}"
    if canonical_key not in canonical_hist:
        writeBytesToS3(f's3://{dispatch_bucket}/{canonical_key}', prettyXML(order))
        # One-off to reduce chance of duplicate toner dispatch: skip if toner is already below probable threshold
        # I.e. assume toner has been sent and write notional canonical order
        if datestring==accounts.get(int(p.PrimaryAcc), "Not Found") and p['Toner.Percent']<initial_toner_min:
            print(f"Skipping order for {canonical_key} as under starting toner threshold")
            return
        print(f'Writing order for {canonical_key}')
        writeBytesToS3(f's3://{dispatch_bucket}/{orders_prefix}{testing_suffix}/{datestring}/{orderName(ts)}.xml', prettyXML(order))

################################################################################
# Commands
################################################################################

cs = cloud.getCacheDetails()
model_paths = dict([x[0], x[3]] for x in cs)

if models:
    model_paths = {m:p for m, p in model_paths.items() if m in models}
res = buildDataset(model_paths.values(), num_procs=build_dataset_procs)

if predict_all:
    sers = res.Serial.unique()
else:
    ser_df = pd.read_csv('s3://ricoh-prediction-misc/mif/current/customer.csv')
    # Types from CSV MIF data are unreliable, forcing string representation works for the purposes of this script
    ser_df.PrimaryAcc = ser_df.PrimaryAcc.astype('str')
    sers = ser_df[ser_df.PrimaryAcc.isin([str(x) for x in accounts.keys()])]['SerialNo']


preds = makePredictions(res, sers)
pred_serial = predSerialXML(preds)

to_merge = ser_df[['SerialNo', 'Name', 'Customer', 'AccountNo','PrimaryAcc', 'Model']]
to_merge = to_merge.rename(columns={'SerialNo':'Serial'})
assert(len(to_merge.Serial.unique())==len(to_merge.Serial))
joined = preds.merge(to_merge, on='Serial')

indices_maps={c:c_to_indices_map(c) for c in colors_norm}

canonical_hist=cloud.S3Keys(dispatch_bucket, canonical_orders_prefix+'/')
canonical_hist=list(canonical_hist)

tz = pytz.timezone('NZ')
nz_now = dt.datetime.now(tz)
datestring = nz_now.strftime('%Y%m%d')

writeBytesToS3(f's3://{dispatch_bucket}/{predictive_serials_prefix}{testing_suffix}/{datestring}/PredictiveSerialNumbers.xml', prettyXML(pred_serial))

for i in range(len(preds)):
    try:
        p = joined.iloc[i,:]
        process_pred(p)
    except:
        traceback.print_exc()
