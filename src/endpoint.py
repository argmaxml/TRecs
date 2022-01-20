from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import numpy as np
import sys, json, itertools, collections
from fastapi import FastAPI
from operator import itemgetter as at
from pathlib import Path
import gc,ctypes
from smart_open import open
libc = ctypes.CDLL("libc.so.6")
sys.path.append("../src")
from similarity_helpers import parse_server_name
import encoders

data_dir = Path(__file__).absolute().parent.parent / "data"
model_dir = Path(__file__).absolute().parent.parent / "models"
api = FastAPI()
with (data_dir / "config.json").open('r') as f:
    config = json.load(f)
sim_params=config[config["similarity_engine"]]
Index = parse_server_name(config["similarity_engine"])
partitions, schema = None, None
index_labels = []


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


def free_memory():
    gc.collect()
    libc.malloc_trim(0)


@api.get("/")
async def read_root():
    return {"status": "OK", "schema_initialized": schema is not None}


@api.get("/partitions")
async def api_partitions():
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    display = lambda t: str(t[0]) if len(t)==1 else str(t)
    max_elements  = {display(p):partitions[i].get_max_elements()  for i,p in enumerate(schema["partitions"])}
    element_count = {display(p):partitions[i].get_current_count() for i,p in enumerate(schema["partitions"])}
    return {"status": "OK", "max_elements": max_elements, "element_count":element_count, "n": len(schema["partitions"]),"dim":schema["dim"]}


@api.post("/fetch")
def api_fetch(lbls: List[str]):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    if len(index_labels)==0:
        return {"status": "error", "message": "No items are indexed"}
    found = set(lbls)&set(index_labels)
    ids = [index_labels.index(l) for l in found]
    ret = collections.defaultdict(list)
    for p,pn in zip(partitions,schema["partitions"]):
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
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    vec = schema["encode_fn"](data)
    return {"status": "OK", "vec": [float(x) for x in vec]}


@api.post("/init_schema")
def init_schema(sr: Schema):
    global schema, partitions
    data_dir.mkdir(parents=True, exist_ok=True)
    schema_dict = sr.to_dict()
    with (data_dir/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    schema = encoders.parse_schema(schema_dict)
    partitions = [Index(schema["metric"], schema["dim"], **sim_params) for _ in schema["partitions"]]
    enc_sizes = {k:len(v) for k,v in schema["encoders"].items()}
    free_memory()
    return {"status": "OK", "partitions": len(partitions), "vector_size":schema["dim"], "feature_sizes":enc_sizes, "total_items":len(index_labels)}

@api.post("/get_schema")
def get_schema():
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    else:
        return schema

@api.post("/get_schema_weights")
def get_schema():
    if schema is None:
        return {"status": "error", "message": "Schema and weights not initialized"}
    else:
        return {key: schema['encoders'][key].column_weight for key in schema['encoders']}

@api.post("/index")
async def api_index(data: Union[List[Dict[str, str]], str]):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    if type(data)==str:
        # read data remotely
        with open(data, 'r') as f:
            data = json.load(f)
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
        affected_partitions += 1
        num_ids = list(map(index_labels.index, ids))
        partitions[idx].add_items(items, num_ids)
    return {"status": "OK", "affected_partitions": affected_partitions}


@api.post("/query")
async def api_query(query: KnnQuery):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    if len(index_labels)==0:
        return {"status": "error", "message": "No items are indexed"}
    try:
        idx = schema["index_num"](query.data)
    except Exception as e:
        return {"status": "error", "message": "Error in partitioning: " + str(e)}
    try:
        vec = schema["encode_fn"](query.data)
    except Exception as e:
        return {"status": "error", "message": "Error in encoding: " + str(e)}
    try:
        vec = vec.reshape(1,-1).astype('float32') # for faiss
        distances, num_ids = partitions[idx].search(vec, k=query.k)
    except Exception as e:
        return {"status": "error", "message": "Error in querying: " + str(e)}
    if len(num_ids) == 0:
        labels, distances = [], []
    else:
        labels = [index_labels[n] for n in num_ids[0]]
        distances = [float(d) for d in distances[0]]
    ret = {"status": "OK", "ids": labels, "distances": distances}
    if query.explain:
        vec = vec.reshape(-1)
        explanation = []
        X = partitions[idx].get_items(num_ids[0])
        first_sim = None
        for ret_vec in X:
            start=0
            explanation.append({})
            for col,enc in schema["encoders"].items():
                if enc.column_weight==0:
                    explanation[-1][col] = float(enc.column_weight)
                    continue
                end = start + len(enc)
                ret_part = ret_vec[start:end]
                query_part =   vec[start:end]
                if schema["metric"]=='l2':
                    # The returned distance from the similarity server is not squared
                    dst=((ret_part-query_part)**2).sum()
                else:
                    sim=np.dot(ret_part,query_part)
                    # Correct dot product to be ascending
                    if first_sim is None:
                        first_sim = sim
                        dst = 0
                    else:
                        dst = 1-sim/first_sim
                explanation[-1][col]=float(dst)
                start = end
        ret["explanation"]=explanation
    return ret


@api.post("/save_model")
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

@api.post("/load_model")
async def api_load(model_name:str):
    global index_labels, partitions, schema
    with (model_dir/model_name/"schema.json").open('r') as f:
        schema_dict=json.load(f)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir/"schema.json").open('w') as f:
        json.dump(schema_dict,f)
    schema = encoders.parse_schema(schema_dict)
    partitions = [Index(schema["metric"], schema["dim"], **sim_params) for _ in schema["partitions"]]
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

    uvicorn.run("__main__:api", host="0.0.0.0", port=5000, log_level="info")
