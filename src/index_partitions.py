import json, logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from joblib import delayed, Parallel
from encoders import parse_schema
from similarity_helpers import LazyHnsw, FlatFaiss

start = datetime.now()

index_labels = []
#TODO: make a cofig
Index = FlatFaiss

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

partition_dir = Path(__file__).absolute().parent.parent / "data/ny/partitioned"
data_dir = Path(__file__).absolute().parent.parent / "data"
model_dir = Path(__file__).absolute().parent.parent / "models/test"

with (data_dir / "config.json").open('r') as f:
    config = json.load(f)
logging.debug("Copy Schema")
with (partition_dir/"schema.json").open('r') as i:
    with (model_dir/"schema.json").open('w') as o:
        json.dump(json.load(i),o)
logging.debug("Start Index")

#for p in tqdm(list(partition_dir.glob("*.npy"))):
@delayed
def index_one_partition(p):
    with p.with_suffix('.meta').open('r') as f:
        meta = json.load(f)
    index_instance = Index(meta["metric"], meta["dim"], **config["hnswlib"])
    try:
        index_instance.load_index(str(model_dir/str(meta["index_num"])))
    except:
        pass
    arr=np.load(p)
    ids=np.arange(meta["start_num_idx"],meta["start_num_idx"]+meta["size"])
    if len(arr)==0:
        return []
    index_instance.add_items(arr,ids)
    index_instance.save_index(str(model_dir/str(meta["index_num"])))
    return [(int(i),str(l)) for i,l in zip(ids,meta["ids"])]

index_labels = Parallel(-1)([index_one_partition(p) for p in partition_dir.glob("*.npy")])
index_labels = sorted(sum(index_labels,[]))
index_labels = [l for i,l in index_labels]
logging.debug("Save labels")
with (model_dir/"index_labels.json").open('w') as f:
    json.dump(index_labels,f)

end = datetime.now()
logging.debug("Took {s} seconds to index".format(s=(end-start).seconds))
# with open("schema.json", 'r') as f:
    # schema = json.load(f)
# requests.post("http://127.0.0.1:5000/init_schema", json=schema)
# for f in tqdm(j_files):
    # print(requests.post("http://127.0.0.1:5000/index", json=f).json())
