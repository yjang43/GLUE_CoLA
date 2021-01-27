import time
import pickle
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


FILE_DIR = os.path.dirname(os.path.realpath(__file__))

model_fn = "roberta-base.pk"
fp = os.path.join(FILE_DIR, "models/", model_fn)
print(fp)
with open(fp, 'rb') as f:
    obj = pickle.load(f)
    model, tokenizer = obj['model'], obj['tokenizer']


def make_pred(passage):
    inf_time = time.time()
    inp = tokenizer(passage, return_tensors='pt')
    out = model(**inp).logits
    out = torch.softmax(out, dim=1)
    inf_time = round(time.time() - inf_time, 3)
    prob = round(out[0][1].item(), 3)
    print(f"prob: {prob}\ttime: {inf_time}")
    return {"prob": prob}


if __name__ == "__main__":
    sequence = ["I am a boy."]
    make_pred(sequence)