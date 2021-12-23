import json, collections, logging
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from encoders import parse_schema
from pathlib import Path
from joblib import delayed, Parallel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
data_dir = Path(".").parent.absolute() /"data/ny"
REPARTITON_LIMIT=10000

with open(data_dir / "schema.json",'r') as f:
    schema = json.load(f)
partiton_field = schema["filters"][0]["field"]
partiton_values = schema["filters"][0]["values"]

encode = parse_schema(schema)["encode_fn"]

partitions = collections.defaultdict(pd.DataFrame)

logging.debug("Repartition")
p_files = [f for f in data_dir.glob("*.parquet")]
for v in tqdm(partiton_values):
    subindex = 0
    for f in p_files:
        df = pd.read_parquet(f)
        df=df[df[partiton_field]==v]
        n = len(df)//REPARTITON_LIMIT
        for sp in range(n+1):
            increment_df = df.iloc[:REPARTITON_LIMIT]
            if len(partitions[f"{v}_{subindex:05d}"])+len(increment_df)>REPARTITON_LIMIT:
                subindex+=1
            partitions[f"{v}_{subindex:05d}"]=pd.concat([partitions[v],increment_df])
            df=df.iloc[REPARTITON_LIMIT:]
            if len(increment_df)>0:
                subindex+=1

logging.debug("Save to local json files")
(data_dir/"partitioned").mkdir(exist_ok=True)
j_files = []
for p,df in partitions.items():
    j_file = data_dir/"partitioned"/(p+".json")
    df.to_json(j_file, orient='records')
    j_files.append(j_file)

logging.debug("Encoding each partition")
def fstack(f,lst):
    """Combines np.vstack and parallelism"""
    if len(lst)==0:
        return np.array([])
    elif len(lst)<1024:
        return np.vstack(Parallel(-1)([f(d) for d in lst]))
    return np.vstack([fstack(f,lst[:len(lst)//2]),fstack(f,lst[len(lst)//2:])])

for jf in tqdm(j_files):
    with open(jf,'r') as f:
        data = json.load(f)
    arr = fstack(delayed(encode), data)
    np.save(str(jf).replace(".json", ".npy"), arr)
    jf.unlink()
