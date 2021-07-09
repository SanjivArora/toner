import feather
import boto3
import io
import datetime
import math
import dateutil
import itertools
import datetime as dt
from functools import partial
from functools import reduce
import pandas as pd


in_bucket_name = 'ricoh-prediction-data-cache'



def readFeatherFileFromS3(s3_url):
    assert s3_url.startswith("s3://")
    bucket_name, key_name = s3_url[5:].split("/", 1)

    s3_client = boto3.client('s3')
    retr = s3_client.get_object(Bucket=bucket_name, Key=key_name)
    
    # Disable threading for reading feather file as this interacts poorly with multiprocessing
    return pd.read_feather(io.BytesIO(retr['Body'].read()), use_threads=False)

def writeFeatherFileToS3(s3_url, df):
    assert s3_url.startswith("s3://")
    bucket_name, key_name = s3_url[5:].split("/", 1)
    
    f = io.BytesIO()
    df.to_feather(f)
    f.seek(0)
    
    s3_client = boto3.client('s3')
    s3_client.upload_fileobj(f, bucket_name, key_name)
    
def writeCSVFileToS3(s3_url, df, index=False):
    assert s3_url.startswith("s3://")
    bucket_name, key_name = s3_url[5:].split("/", 1)
    
    contents = io.StringIO()
    df.to_csv(contents, index=index)
    contents.seek(0)
    f = io.BytesIO(contents.getvalue().encode())
    
    s3_client = boto3.client('s3')
    s3_client.upload_fileobj(f, bucket_name, key_name)

def writeBytesToS3(s3_url, b):
    assert s3_url.startswith("s3://")
    bucket_name, key_name = s3_url[5:].split("/", 1)
    
    f = io.BytesIO()
    f.write(b)
    f.seek(0)
    
    s3_client = boto3.client('s3')
    s3_client.upload_fileobj(f, bucket_name, key_name)
    
def readFromS3(s3_url):
    print(f"Reading feather file from {s3_url}")
    p = readFeatherFileFromS3(s3_url)
    # Add meaningful unique row indices
    idx=[str(x) + '-' + str(y) for (x, y) in zip(p['Serial'], p['FileDate'])]
    p['idx']=idx
    p=p.set_index('idx')
    p.sort_values('RetrievedDateTime', inplace=True, ascending=False)
    return p 

# https://stackoverflow.com/questions/30249069/listing-contents-of-a-bucket-with-boto3
def S3Keys(bucket_name, prefix='/', delimiter='/', start_after=''):
    prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    s3_client = boto3.client('s3')
    s3_paginator = s3_client.get_paginator('list_objects_v2')
    for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
        for content in page.get('Contents', ()):
            yield content['Key']
            
# Find common prefixes
def S3Prefixes(bucket_name, delimeter='/'):
    s3_client = boto3.client('s3')
    s3_paginator = s3_client.get_paginator('list_objects_v2')
    for result in s3_paginator.paginate(Bucket=in_bucket_name, Delimiter=delimeter):
        for prefix in result.get('CommonPrefixes'):
            yield prefix.get('Prefix')


def parseName(p):
    parts = p.split('/')
    date = dateutil.parser.parse(parts[0])
    region = parts[1]
    assert parts[2].endswith('.feather')
    model = parts[2].split('.')[0]
    return model, region, date, p

# Get (model, region, date, path) tuples for cached datasets
def getCacheDetails(region = 'RNZ', bucket=in_bucket_name):
    ks = S3Keys(bucket)
    ks = list(ks)
    candidates = list(map(parseName, ks))
    candidates = list(filter(lambda x: x[1]==region, candidates))

    candidates.sort(key=lambda x: x[0])
    by_model = itertools.groupby(candidates, lambda x: x[0])
    # Get most recent
    most_recent = [max(list(x[1])) for x in by_model]
    return most_recent

