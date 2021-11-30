from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import numpy as np
import sys, json, itertools
from fastapi import FastAPI
from operator import itemgetter as at
from pathlib import Path

sys.path.append("../src")
from hnsw_helpers import LazyHnsw
import encoders

data_dir = Path(__file__).absolute().parent.parent / "data"
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


@app.get("/")
async def read_root():
    return {"status": "OK", "schema_initialized": schema is not None}


@app.get("/partitions")
async def api_partitions():
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    return {"status": "OK", "partitons": schema["partitions"], "n": len(schema["partitions"])}


@app.post("/encode")
async def api_encode(data: Dict[str, str]):
    if schema is None:
        return {"status": "error", "message": "Schema not initialized"}
    vec = schema["encode_fn"](data)
    return {"status": "OK", "vec": [float(x) for x in vec]}


@app.post("/init_schema")
def init_schema(sr: Schema):
    global schema, partitions
    schema = encoders.parse_schema(sr.to_dict())
    partitions = [LazyHnsw(schema["metric"], schema["dim"], **config["hnswlib"]) for _ in schema["partitions"]]
    # for index in partitions:
    # index.init_index(**config["hnswlib"])
    return {"status": "OK"}


@app.post("/index")
async def api_index(data: List[Dict[str, str]]):
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
        if (partitions[idx].max_elements < len(items)):
            partitions[idx].resize_index(len(items))
        affected_partitions += 1
        num_ids = list(map(index_labels.index, ids))
        partitions[idx].add_items(items, num_ids)
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
    return {"status": "OK", "ids": labels, "distances": distances}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, log_level="info")
