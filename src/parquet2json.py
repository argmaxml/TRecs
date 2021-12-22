import os, json
import requests
import pandas as pd
from tqdm import tqdm

p_files = [f for f in os.listdir() if f.endswith(".parquet")]
j_files = []
for f in tqdm(p_files):
    of = f.replace(".parquet", ".json")
    df = pd.read_parquet(f)
    df.to_json(of, orient='records')
    j_files.append(os.getcwd()+os.sep+of)

print("Done transforming to json")

with open("ny_schema.json", 'r') as f:
    schema = json.load(f)
requests.post("http://127.0.0.1:5000/init_schema", json=schema)
for f in tqdm(j_files):
    print(requests.post("http://127.0.0.1:5000/index", json=f).json())
