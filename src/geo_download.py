# Author Yinsen Miao
import pandas as pd
import requests
import time
import json
import os
from tqdm import tqdm
import multiprocessing

prefix = "../data/parid"
os.makedirs(prefix, exist_ok=True)
has_parids = [file.replace(".txt", "") for file in os.listdir(prefix)]

# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def dispatch_jobs(data, job_number):
    total = len(data)
    chunk_size = int(total / job_number)
    slice = chunks(data, chunk_size)
    jobs = []

    for i, s in enumerate(slice):
        j = multiprocessing.Process(target=do_job, args=(i, s))
        jobs.append(j)
    for j in jobs:
        j.start()

def do_job(job_id, data_slice):
    for parid in data_slice:
        r = requests.get(url="http://tools.wprdc.org/property-api/v0/parcels/%s" % parid)
        with open("%s/%s.txt" % (prefix, parid), "w") as file:
            json.dump(r.json(), file)


if __name__ == "__main__":
    dat = pd.read_pickle("../data/cleandata.pkl")
    all_parids = dat["PARID"].unique().tolist()
    miss_parids = list(set(all_parids) - set(has_parids))
    print("Look for geo information for %d houses" % len(miss_parids))
    dispatch_jobs(miss_parids, 10)