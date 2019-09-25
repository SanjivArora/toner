#!/usr/bin/env python3

import datetime as dt

import boto3
from botocore.exceptions import ClientError
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

# Handle pathing for import 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.predict_coverage import *
from lib import cloud


# Set to limit the number of cores to use to build dataset (default is number of logical cores)
build_datasets_procs = None

models = None
#models = ['E19']

to_emails = ['bnarsey@ricoh.co.nz', 'mvas@ricoh.co.nz', 'smatthews@ricoh.co.nz', 'redwey@ricoh.co.nz']
#to_emails = ['smatthews@ricoh.co.nz']
from_email = "Ricoh Prediction <ricoh-prediction-mail@sdmatthews.com>"
email_subject = 'Auckland University Predictions'
aws_region = "us-east-1"

# Adapted from https://stackoverflow.com/questions/45298069/ses-attachments-with-python
def sendEmail(preds, current):
    ses = boto3.client('ses', region_name=aws_region)
    msg = MIMEMultipart()
    msg['Subject'] = email_subject
    msg['From'] = from_email 
    msg['To'] = ", ".join(to_emails)

    # what a recipient sees if they don't use an email reader
    msg.preamble = 'Multipart message.\n'

    # the message body
    part = MIMEText('')
    msg.attach(part)

    # the attachments
    preds_file = f'au-preds-{date_string}.csv'
    current_file = f'au-current-{date_string}.csv'
    part = MIMEApplication(preds.to_csv(index=False))
    part.add_header('Content-Disposition', 'attachment', filename=preds_file)
    msg.attach(part)
    part = MIMEApplication(current.to_csv(index=False))
    part.add_header('Content-Disposition', 'attachment', filename=current_file)
    msg.attach(part)

    # and send the message
    result = ses.send_raw_email(
        Source=msg['From'],
        Destinations=to_emails,
        RawMessage={'Data': msg.as_string()}
    )                                                                                                       
    print(result)



cs = cloud.getCacheDetails()
model_paths = dict([x[0], x[3]] for x in cs)
if models:
    model_paths = {m:p for m, p in model_paths.items() if m in models}

build_datasets_procs = None
res = buildDataset(model_paths.values(), kwargs={'allow_missing':True}, num_procs=build_datasets_procs)

au_sers = pd.read_excel('s3://ricoh-prediction-misc/UoA.xlsx')['SerialNo']
#au_sers = res.Serial.unique()

#preds = makePredictions(res)
preds = makePredictions(res[res.Serial.isin(au_sers)])

current = preds[preds['Days.To.Zero.From.Today.Earliest.Expected'] <= 7]
current = current[current['DataAge'] < 7] 
current = current.sort_values(['Days.To.Zero.From.Today.Earliest.Expected', 'Serial'], ascending=True)

print(current)

try:
    t=dt.datetime.today()
    date_string = f"{t.year}{t.month:02d}{t.day:02d}"
    writeCSVFileToS3(f's3://ricoh-prediction-misc/au-preds/au-preds-{date_string}.csv', preds)
    writeCSVFileToS3(f's3://ricoh-prediction-misc/au-preds/au-current-{date_string}.csv', current)
except:
    print("Exception saving to S3:")
    traceback.format_exc()

try:
    sendEmail(preds, current)
except:
    print("Exception sending email:") 
    traceback.format_exc()


#by_ser = res[res.Serial.isin(au_sers)].groupby('Serial')
##by_ser = res.sort_values(['RetrievedDate'], ascending=False).groupby('Serial')
#
#for c in colors_norm:
#    days=14
#    atz = by_ser.apply(lambda x: daysAtZero(x, c))
#    print(f"{c} at zero for over {days} days:")
#    print(atz[atz>days])
#    print()
#
#
## In[35]:
#
#
##by_ser = res[res.Serial.isin(au_sers)].groupby('Serial')
#by_ser = res.sort_values(['RetrievedDate'], ascending=False).groupby('Serial')
#
#for c in colors_norm:
#    days=90
#    atz = by_ser.apply(lambda x: daysAtEnd(x, c))
#    print(f"{c} at end for over {days} days:")
#    print(atz[atz>days])
#    print()
#    for s in atz[atz>days].index:
#        plotToner(s, c)
#        pass
#
