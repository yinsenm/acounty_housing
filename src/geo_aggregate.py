# Author Yinsen Miao
import pandas as pd
import time
import json
import os
from tqdm import tqdm

prefix = "../data/parid"
os.makedirs(prefix, exist_ok=True)
has_parids = [file.replace(".txt", "") for file in os.listdir(prefix)]

list_parid_geo = []
for parid in tqdm(has_parids):
    try:
        with open('%s/%s.txt' % (prefix, parid)) as json_file:
            data = json.load(json_file)
        latitude, longitude = data["results"][0]["geos"]["centroid"]["coordinates"]
        latitude, longitude = float(latitude), float(longitude)
        census_tract = data["results"][0]["data"]["centroids_and_geo_info"][0]["geo_name_tract"]
        parid_geo = {
            "parid": parid,
            "latitude": latitude,
            "longitude": longitude,
            "census_tract": census_tract
        }
        list_parid_geo.append(parid_geo)
    except:
        print("%s" % parid)

# aggregate all parids
parid_geo_df = pd.DataFrame(list_parid_geo)
parid_geo_df.to_csv("../data/parid_geo.csv", index=False)
