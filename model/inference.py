import time
import pickle
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Model:
    model_fns = ["bert-base-cased.pk", "distilbert-base-cased.pk", "roberta-base.pk"]   # chosen pre-trained models
 
    def __init__(self):
        """ Model constructor
        """

        default_model = "bert-base-cased.pk"   # best performance model
        self.init_model(default_model)

    def init_model(self, model_fn):
        """ Initialize model with tokenizer

        [Parameters]
        model_fn: file name of model in str
        """

        if model_fn not in self.model_fns:  # cannot init unless models chosen
            return False
        fp = os.path.join(FILE_DIR, "models/", model_fn)
        with open(fp, 'rb') as f:
            obj = pickle.load(f)
            self.model, self.tokenizer = obj['model'], obj['tokenizer']
        return True

    def make_pred(self, passage):
        """ Make prediction of acceptability of given passage

        [Parameters]
        passage: english passage
        """
        # inf_time = time.time()
        inp = self.tokenizer(passage, return_tensors='pt')
        out = self.model(**inp).logits
        out = torch.softmax(out, dim=1)
        # inf_time = round(time.time() - inf_time, 3)
        prob = round(out[0][1].item(), 3)
        return prob
