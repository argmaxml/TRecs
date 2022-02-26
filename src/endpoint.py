from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import numpy as np
import sys, json
from fastapi import FastAPI
from pathlib import Path
import gc
from smart_open import open
try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    def free_memory():
        gc.collect()
        libc.malloc_trim(0)
except:
    def free_memory():
        gc.collect()
sys.path.append("../src")
import partitioner
import pandas as pd

data_dir = Path(__file__).absolute().parent.parent / "data"
api = FastAPI()



class Column(BaseModel):
    field: str
    values: List[str]
    type: Optional[str]
    weight: Optional[float]
    window: Optional[List[float]]
    url: Optional[str]
    entity: Optional[str]
    environment: Optional[str]
    feature: Optional[str]
    length: Optional[int]


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


@api.get("/")
async def read_root():
    return {"status": "OK", "schema_initialized": partitioner.schema_initialized()}


@api.get("/partitions")
async def api_partitions():
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    display = lambda t: str(t[0]) if len(t)==1 else str(t)
    max_elements  = {display(p):partitions[i].get_max_elements()  for i,p in enumerate(partitioner.get_partitions())}
    element_count = {display(p):partitions[i].get_current_count() for i,p in enumerate(partitioner.get_partitions())}
    return {"status": "OK", "max_elements": max_elements, "element_count":element_count, "n": len(partitioner.get_partitions()),"dim":partitioner.get_embedding_dimension()}


@api.post("/fetch")
def api_fetch(lbls: List[str]):
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    if len(index_labels)==0:
        return {"status": "error", "message": "No items are indexed"}
    found = set(lbls)&set(index_labels)
    ids = [index_labels.index(l) for l in found]
    ret = collections.defaultdict(list)
    for p,pn in zip(partitions,partitioner.get_partitions()):
        try:
            ret[pn].extend([tuple(float(v) for v in vec) for vec in p.get_items(ids)])
        except:
            # not found
            continue
    ret = map(lambda k,v: (k[0],v) if len(k)==1 else (str(k), v),ret.keys(), ret.values())
    ret = dict(filter(lambda kv: bool(kv[1]),ret))
    return ret

@api.post("/encode")
async def api_encode(data: Dict[str, str]):
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    vec = schema["encode_fn"](data)
    return {"status": "OK", "vec": [float(x) for x in vec]}


@api.post("/init_schema")
def init_schema(sr: Schema):
    schema_dict = sr.to_dict()
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    partitions, enc_sizes = partitioner.init_schema(schema_dict)
    free_memory()
    return {"status": "OK", "partitions": len(partitions), "vector_size":partitioner.get_embedding_dimension(), "feature_sizes":enc_sizes, "total_items":partitioner.get_total_items()}

@api.post("/get_schema")
def get_schema():
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    else:
        with (data_dir/"schema.json").open('r') as f:
            schema_dict=json.load(f)
        return schema_dict

@api.post("/index")
async def api_index(data: Union[List[Dict[str, str]], str]):
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    if type(data)==str:
        # read data remotely
        with open(data, 'r') as f:
            data = json.load(f)
    errors, affected_partitions = partitioner.index(data)
    if any(errors):
        return {"status": "error", "items": errors}
    return {"status": "OK", "affected_partitions": affected_partitions}


@api.post("/query")
async def api_query(query: KnnQuery):
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    if partitioner.get_total_items()==0:
        return {"status": "error", "message": "No items are indexed"}
    try:
        labels,distances, explanation =partitioner.query(query.data, query.k, query.explain)
        if any(explanation):
            return {"status": "OK", "ids": labels, "distances": distances, "explanation":explanation}
        return {"status": "OK", "ids": labels, "distances": distances}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@api.post("/save_model")
async def api_save(model_name:str):
    if not partitioner.schema_initialized():
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

@api.post("/load_model")
async def api_load(model_name:str):
    global index_labels, partitions, schema
    with (model_dir/model_name/"schema.json").open('r') as f:
        schema_dict=json.load(f)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    schema = encoders.parse_schema(schema_dict)
    partitions = [Index(schema["metric"], schema["dim"], **sim_params) for _ in partitioner.get_partitions()]
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

@api.post("/list_models")
async def api_list():
    ret = [d.name for d in model_dir.iterdir() if d.is_dir()]
    ret.sort()
    return ret

if __name__ == "__main__":
    import uvicorn
    from argparse import ArgumentParser
    argparse = ArgumentParser()
    argparse.add_argument('--host', default='0.0.0.0', type=str, help='host')
    argparse.add_argument('--port', default=5000, type=int, help='port')
    args = argparse.parse_args()
    uvicorn.run("__main__:api", host=args.host, port=args.port, log_level="info")
