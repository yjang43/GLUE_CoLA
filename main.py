from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

db = []

class Item(BaseModel):
    passage: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def read_root(item: Item):
    passage = item.passage
    prob = 0.1  # TODO: replace this value with make_pred
    return {"prob": prob}
