import json, collections, logging, os
import requests
import numpy as np
import pandas as pd
from operator import itemgetter as at
from tqdm import tqdm
from encoders import parse_schema
from pathlib import Path
from joblib import delayed, Parallel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
data_dir = Path(__file__).absolute().parent.parent /"data/ny"
REPARTITON_LIMIT=10000
PARTITION_SEP='_'

with open(data_dir / "schema.json",'r') as f:
    schema = json.load(f)
partition_dir=(data_dir/"partitioned")
partition_dir.mkdir(exist_ok=True)
# copy schema
with (partition_dir/"schema.json").open('w') as f:
    json.dump(schema, f)
#
partiton_field = schema["filters"][0]["field"]
partiton_values = schema["filters"][0]["values"]

schema = parse_schema(schema)
encode = schema["encode_fn"]

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
            if len(partitions[f"{v}{PARTITION_SEP}{subindex:05d}"])+len(increment_df)>REPARTITON_LIMIT:
                subindex+=1
            partitions[f"{v}{PARTITION_SEP}{subindex:05d}"]=pd.concat([partitions[f"{v}{PARTITION_SEP}{subindex:05d}"],increment_df])
            df=df.iloc[REPARTITON_LIMIT:]
            if len(increment_df)>0:
                subindex+=1

logging.debug("Save to local json files")
# save jsons
j_files = []
for p,df in partitions.items():
    j_file = partition_dir/(p+".json")
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

start_num_idx=0
for jf in tqdm(j_files):
    part_name, part_num = jf.name.rsplit(PARTITION_SEP,1)
    with open(jf,'r') as f:
        data = json.load(f)
    arr = fstack(delayed(encode), data)
    np.save(jf.with_suffix('.npy'), arr)
    with jf.with_suffix('.meta').open('w') as f:
        index_num = schema["index_num"]({partiton_field:part_name})
        meta = {
            "size":len(arr),
            "start_num_idx":start_num_idx,
            "ids": [d['id'] for d in data],
            "partition_name": part_name,
            "partition_number":part_num,
            "index_num":index_num,
            }
        json.dump(meta,f)
    start_num_idx+=len(arr)
    #jf.unlink() #TODO: why not working ?
    os.remove(str(jf))

