from typing import Optional, List, Dict, Any 
from pydantic import BaseModel
import numpy as np
import sys
from fastapi import FastAPI
from operator import itemgetter as at
from pathlib import Path
import hnswlib
sys.path.append("../src")
import encoders
data_dir = Path(__file__).absolute().parent.parent / "data"
model_dir = Path(__file__).absolute().parent.parent / "models"
app = FastAPI()
# TODO: Read from API
with (data_dir/"schema.json").open('r') as f:
    schema=encoders.parse_schema(f)

class Customer(BaseModel):
	name: str
	columns:List[str]

@app.get("/")
async def read_root():
    return {"status": "OK"}

@app.post("/encode")
async def api_encode(data: Dict[str,str]):
    vec = schema["encode_fn"](data)
    return [float(x) for x in vec]

@app.post("/index")
async def api_index(data: List[Dict[str,str]]):
    vec = schema["encode_fn"](data)
    return [float(x) for x in vec]

if __name__=="__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, log_level="info")

