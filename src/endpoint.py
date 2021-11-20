from typing import Optional, List, Dict, Any 
from pydantic import BaseModel
import numpy as np
import sys, json, itertools
from fastapi import FastAPI
from operator import itemgetter as at
from pathlib import Path
import hnswlib
sys.path.append("../src")
import encoders
data_dir = Path(__file__).absolute().parent.parent / "data"
model_dir = Path(__file__).absolute().parent.parent / "models"
app = FastAPI()
with (data_dir/"config.json").open('r') as f:
    config=json.load(f)
# TODO: Read from API
with (data_dir/"schema.json").open('r') as f:
    schema=encoders.parse_schema(f)
partitions = [hnswlib.Index(schema["metric"], schema["dim"]) for _ in schema["partitions"]]
for index in partitions:
    index.init_index(**config["hnswlib"])


class Customer(BaseModel):
	name: str
	columns:List[str]

@app.get("/")
async def read_root():
    return {"status": "OK"}

@app.get("/partitions")
async def api_partitions():
    return {"partitons": schema["partitions"], "n":len(schema["partitions"])}

@app.post("/encode")
async def api_encode(data: Dict[str,str]):
    vec = schema["encode_fn"](data)
    return [float(x) for x in vec]

@app.post("/index")
async def api_index(data: List[Dict[str,str]]):
    try:
        vecs = sorted([(schema["index_num"](datum),schema["encode_fn"](datum), datum["id"]) for datum in data])
    except KeyError as e:
        return {"status": "error", "message": str(e)}
    affected_partitions = 0
    for idx, grp in itertools.groupby(vecs, at(0)):
        _,items,ids = zip(*grp)
        partitions[idx].add_items(items, ids)
        affected_partitions+=1
    return {"affected_partitions": affected_partitions}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, log_level="info")

