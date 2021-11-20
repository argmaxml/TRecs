from typing import Optional, List 
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI
from operator import itemgetter as at
from pathlib import Path
import hnswlib
# sys.path.append("../src")
data_dir = Path(__file__).absolute().parent.parent / "data"
model_dir = Path(__file__).absolute().parent.parent / "models"
app = FastAPI()

class Customer(BaseModel):
	name: str
	columns:List[str]

@app.get("/")
async def read_root():
    return {"status": "OK"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, log_level="info")

