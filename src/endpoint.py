import os
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import sys, json, itertools
from fastapi import FastAPI
from operator import itemgetter as at
from pathlib import Path
import gc,ctypes
from smart_open import open
libc = ctypes.CDLL("libc.so.6")
sys.path.append("../src")
from hnsw_helpers import LazyHnsw
import encoders
import boto3

data_dir = Path(__file__).absolute().parent.parent / "data"
save_dir = Path(__file__).absolute().parent.parent / "data"/'index_data'
bucket='talent-fabric-data-lake'
s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucket)
files_list = []
for object_summary in my_bucket.objects.filter(Prefix="tabsim/new_york_relevant_triplets_to_index/"):
    boto3.client('s3').download_file(bucket, object_summary.key, f"../data/index_data/part{object_summary.key.split('/')[-1]}")
    files_list.append(f"../data/index_data/part{object_summary.key.split('/')[-1]}")

ids_list = []
for f in files_list:
    if f.endswith('parquet'):
        df = pd.read_parquet(f)
        ids_list.extend(df['id'].to_list())
        str_file = df.to_json(path_or_buf = str(save_dir.resolve()) + '/' + f.split('/')[-1].split('.')[0]+'.json', orient='records')
        # with (save_dir/(f.split('/')[-1].split('.')[0]+'.json')).open('w') as jsf:
        #     json.dump(str_file, jsf)
with open('indexed_ids.txt', 'w') as f:
    for element in ids_list:
        f.write(element + "\n")



boto3.client('s3').upload_file('indexed_ids.txt', bucket,"tabsim/indexed_ids.txt")
model_dir = Path(__file__).absolute().parent.parent / "models"
app = FastAPI()
with (data_dir / "config.json").open('r') as f:
    config = json.load(f)
partitions, schema = None, None
index_labels = []


class Column(BaseModel):
    field: str
    values: List[str]
    type: Optional[str]
    weight: Optional[float]
    window: Optional[List[float]]
    url: Optional[str]


class Schema(BaseModel):
    metric: str
    filters: List[Column]
    encoders: List[Column]

    def to_dict(self):
        return {
            "metric": self.metric,
            "filters": [vars(c) for c in self.filters],
            "encoders": [vars(c) for c in self.encoders],
        }


class KnnQuery(BaseModel):
    data: Dict[str, str]
    k: int
    explain:Optional[bool]=False


def free_memory():
    gc.collect()
    libc.malloc_trim(0)


@app.get("/")
async def read_root():
    return {"status": "OK", "schema_initialized": schema is not None}


@app.get("/partitions")
async def api_partitions():
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    display = lambda t: str(t[0]) if len(t)==1 else str(t)
    max_elements  = {display(p):partitions[i].max_elements  for i,p in enumerate(schema["partitions"])}
    element_count = {display(p):partitions[i].element_count for i,p in enumerate(schema["partitions"])}
    return {"status": "OK", "max_elements": max_elements, "element_count":element_count, "n": len(schema["partitions"])}


@app.post("/encode")
async def api_encode(data: Dict[str, str]):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    vec = schema["encode_fn"](data)
    return {"status": "OK", "vec": [float(x) for x in vec]}


@app.post("/init_schema")
def init_schema(sr: Schema):
    global schema, partitions
    data_dir.mkdir(parents=True, exist_ok=True)
    schema_dict = sr.to_dict()
    with (data_dir/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    schema = encoders.parse_schema(schema_dict)
    partitions = [LazyHnsw(schema["metric"], schema["dim"], **config["hnswlib"]) for _ in schema["partitions"]]
    enc_sizes = {k:len(v) for k,v in schema["encoders"].items()}
    free_memory()
    return {"status": "OK", "partitions": len(partitions), "vector_size":schema["dim"], "feature_sizes":enc_sizes}


@app.post("/index")
async def api_index(data: Union[List[Dict[str, str]], str]=save_dir):
    if type(data)==str:
        # read data remotely
        data = []
        flag = 0
        for f in os.listdir(save_dir):
            # if flag == 3:
            #     break
            if f.endswith('json'):
                with open(save_dir/f, 'r') as jsf:
                    datum_file = json.load(jsf)
                    data.extend(datum_file)
                    # flag+=1
        with open('s3://' + bucket + '/tabsim' + "/current_indexed_data.json", 'w') as f:
            json.dump(data,f)
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    try:
        vecs = sorted([(schema["index_num"](datum), schema["encode_fn"](datum), datum["id"]) for datum in data],
                      key=at(0))
    except KeyError as e:
        return {"status": "error", "message": str(e)}
    affected_partitions = 0
    labels = set(index_labels)
    for idx, grp in itertools.groupby(vecs, at(0)):
        _, items, ids = zip(*grp)
        for id in ids:
            if id not in labels:
                labels.add(id)
                index_labels.append(id)
        #if (partitions[idx].max_elements < len(items)):
        #    partitions[idx].resize_index(len(items))
        affected_partitions += 1
        num_ids = list(map(index_labels.index, ids))
        partitions[idx].add_items(items, num_ids, -1)

    return {"status": "OK", "affected_partitions": affected_partitions}


@app.post("/query")
async def api_query(query: KnnQuery):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    try:
        idx = schema["index_num"](query.data)
    except Exception as e:
        return {"status": "error", "message": "Error in partitioning: " + str(e)}
    try:
        vec = schema["encode_fn"](query.data)
    except Exception as e:
        return {"status": "error", "message": "Error in encoding: " + str(e)}
    try:
        num_ids, distances = partitions[idx].knn_query(vec, k=query.k)
    except Exception as e:
        return {"status": "error", "message": "Error in querying: " + str(e)}
    if len(num_ids) == 0:
        labels, distances = [], []
    else:
        labels = [index_labels[n] for n in num_ids[0]]
        distances = [float(d) for d in distances[0]]
    ret = {"status": "OK", "ids": labels, "distances": distances}
    if query.explain:
        explanation = []
        X = partitions[idx].get_items(num_ids[0])
        for ret_vec in X:
            start=0
            explanation.append({})
            for col,enc in schema["encoders"].items():
                end = start + len(enc)
                ret_part = ret_vec[start:end]
                query_part =   vec[start:end]
                if schema["metric"]=='l2':
                    sim=np.sqrt(((ret_part-query_part)**2).sum())
                else:
                    sim=np.dot(ret_part,query_part)
                explanation[-1][col]=float(sim)
                start = end
        ret["explanation"]=explanation
    return ret


@app.post("/save_model")
async def api_save(model_name:str):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    (model_dir/model_name).mkdir(parents=True, exist_ok=True)
    with (model_dir/model_name/"index_labels.json").open('w') as f:
        json.dump(index_labels,f)
    with (data_dir/"schema.json").open('r') as f:
        schema_dict=json.load(f)
    with (model_dir/model_name/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    saved=0
    for i,p in enumerate(partitions):
        fname = str(model_dir/model_name/str(i))
        try:
            p.save_index(fname)
            saved+=1
        except:
            continue
    return {"status": "OK", "saved_indices": saved}

@app.post("/load_model")
async def api_load(model_name:str):
    global index_labels, partitions, schema
    with (model_dir/model_name/"schema.json").open('r') as f:
        schema_dict=json.load(f)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    schema = encoders.parse_schema(schema_dict)
    partitions = [LazyHnsw(schema["metric"], schema["dim"], **config["hnswlib"]) for _ in schema["partitions"]]
    free_memory()
    (model_dir/model_name).mkdir(parents=True, exist_ok=True)
    with (model_dir/model_name/"index_labels.json").open('r') as f:
        index_labels=json.load(f)
    loaded = 0
    for i,p in enumerate(partitions):
        fname = str(model_dir/model_name/str(i))
        try:
            p.load_index(fname)
            loaded+=1
        except:
            continue
    return {"status": "OK", "loaded_indices": loaded}

@app.post("/list_models")
async def api_list():
    return [d.name for d in model_dir.iterdir() if d.is_dir()]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, log_level="info")
