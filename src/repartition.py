import os, json, collections
import requests
import pandas as pd
from tqdm import tqdm
with open("ny_schema.json",'r') as f:
    schema = json.load(f)
partiton_field = schema["filters"][0]["field"]
partiton_values = schema["filters"][0]["values"]

partitions = collections.defaultdict(pd.DataFrame)

p_files = [f for f in os.listdir() if f.endswith(".parquet")]
j_files = []
for f in tqdm(p_files):
    df = pd.read_parquet(f)
    for v in partiton_values:
        partitions[v]=pd.concat([partitions[v],df[df[partiton_field]==v]])

for p,df in partitions.items():
    df.to_json("paritioned/"+p+".json", orient='records')

