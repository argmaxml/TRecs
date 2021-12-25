import json, logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import delayed, Parallel
from encoders import parse_schema
from similarity_helpers import LazyHnsw, FlatFaiss

index_labels = []
#TODO: make a cofig
Index = FlatFaiss

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

partition_dir = Path(__file__).absolute().parent.parent / "data/ny/partitioned"
data_dir = Path(__file__).absolute().parent.parent / "data"
model_dir = Path(__file__).absolute().parent.parent / "models/test"

with open(partition_dir / "schema.json",'r') as f:
    schema = parse_schema(json.load(f))

with (data_dir / "config.json").open('r') as f:
    config = json.load(f)

for p in tqdm(list(partition_dir.glob("*.npy"))):
    if "_" not in p.name:
        continue
    with p.with_suffix('.meta').open('r') as f:
        meta = json.load(f)
    index_instance = Index(schema["metric"], schema["dim"], **config["hnswlib"])
    try:
        index_instance.load_index(str(model_dir/str(meta["index_num"])))
    except:
        pass
    arr=np.load(p)
    ids=np.arange(meta["start_num_idx"],meta["start_num_idx"]+meta["size"])
    if len(arr)==0:
        continue
    index_instance.add_items(arr,ids)
    index_instance.save_index(str(model_dir/str(meta["index_num"])))
    #logging.debug(part_name)
# print("Done transforming to json")

# with open("schema.json", 'r') as f:
    # schema = json.load(f)
# requests.post("http://127.0.0.1:5000/init_schema", json=schema)
# for f in tqdm(j_files):
    # print(requests.post("http://127.0.0.1:5000/index", json=f).json())
