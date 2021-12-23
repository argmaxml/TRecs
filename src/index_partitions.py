import json, logging
import requests
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

with open(partition_dir / "schema.json",'r') as f:
    schema = parse_schema(json.load(f))

with (data_dir / "config.json").open('r') as f:
    config = json.load(f)

REPARTITON_LIMIT=10000

for p in partition_dir.glob("*.npy"):
    if "_" not in p.name:
        continue
    part_name, part_num = p.name.rsplit('_',1)
    index_num = schema["index_num"(part_num)]
    logging.debug(part_name)
# print("Done transforming to json")

# with open("schema.json", 'r') as f:
    # schema = json.load(f)
# requests.post("http://127.0.0.1:5000/init_schema", json=schema)
# for f in tqdm(j_files):
    # print(requests.post("http://127.0.0.1:5000/index", json=f).json())
