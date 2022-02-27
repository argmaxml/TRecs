from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import sys, json
from fastapi import FastAPI
from pathlib import Path
import gc
try:
    from smart_open import open
except:
    pass
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
from partitioner import Partitioner
import pandas as pd

data_dir = Path(__file__).absolute().parent.parent / "data"
with (data_dir / "config.json").open('r') as f:
    config = json.load(f)
partitioner = Partitioner(config)
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
    ret = partitioner.get_partition_stats()
    ret["status"] = "OK"
    return ret


@api.post("/fetch")
def api_fetch(lbls: List[str]):
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    if partitioner.get_total_items()==0:
        return {"status": "error", "message": "No items are indexed"}
    return partitioner.fetch(lbls)

@api.post("/encode")
async def api_encode(data: Dict[str, str]):
    if not partitioner.schema_initialized():
        return {"status": "error", "message": "Schema not initialized"}
    vec = partitioner.encode(data)
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
    saved = partitioner.save_model(model_name)
    return {"status": "OK", "saved_indices": saved}

@api.post("/load_model")
async def api_load(model_name:str):
    loaded = partitioner.load_model(model_name)
    free_memory()
    return {"status": "OK", "loaded_indices": loaded}

@api.post("/list_models")
async def api_list():
    return partitioner.list_models()

if __name__ == "__main__":
    import uvicorn
    from argparse import ArgumentParser
    argparse = ArgumentParser()
    argparse.add_argument('--host', default='0.0.0.0', type=str, help='host')
    argparse.add_argument('--port', default=5000, type=int, help='port')
    args = argparse.parse_args()
    uvicorn.run("__main__:api", host=args.host, port=args.port, log_level="info")
