import os, json, collections
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from encoders import parse_schema
from multiprocessing import Pool

with open("ny_schema.json",'r') as f:
    schema = json.load(f)
partiton_field = schema["filters"][0]["field"]
partiton_values = schema["filters"][0]["values"]

encode = parse_schema(schema)["encode_fn"]

partitions = collections.defaultdict(pd.DataFrame)

print("repartition")
p_files = [f for f in os.listdir() if f.endswith(".parquet")]
j_files = []
for f in tqdm(p_files):
    df = pd.read_parquet(f)
    for v in partiton_values:
        partitions[v]=pd.concat([partitions[v],df[df[partiton_field]==v]])

print("save to local json files")
for p,df in partitions.items():
    j_file = "paritioned/"+p+".json"
    df.to_json(j_file, orient='records')
    j_files.append(j_file)

print("encoding each partition")
#TODO: run in parallel
# pool = Pool(8)
# def encode_one_partition(jf):
    # with open(jf,'r') as f:
        # data = json.load(f)
    # arr = np.vstack([encode(datum) for datum in data])
    # np.save(jf.replace(".json", ".npy"), arr)
# pool.map(encode_one_partition,j_files)

for jf in tqdm(j_files):
    print(jf)
    with open(jf,'r') as f:
        data = json.load(f)
    arr = np.vstack([encode(datum) for datum in data])
    np.save(jf.replace(".json", ".npy"), arr)
