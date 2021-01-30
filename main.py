from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from model import Model

app = FastAPI()

model = Model()

class Item(BaseModel):
    passage: str    # english passage


@app.get("/set-model")  # allow setting to different model for result comparison
def set_model(model_str: str):
    """ Allow setting to different model for result comparison

    [Parameters]
    model_str: file name of model in str
               should be one of "bert-base-cased.pk", "distilbert-base-cased.pk", "roberta-base.pk"
    """
    is_success = model.init_model(model_str)
    return is_success


@app.post("/generate") 
def generate_prob(item: Item):
    """ Generate probability of acceptability in JSON when requested with JSON

    [Parameters]
    item: item with passage as an attribute
    """
    passage = item.passage
    prob = model.make_pred(passage)
    return {"prob": prob}
