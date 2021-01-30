# GLUE_CoLA
Benchmark BERT models with GLUE's CoLA task and implement in API form

## Models
I chose models with cased tokenization because capitalization is a part of acceptability(correct grammar) task.
BERT is used for a baseline.
DistilBERT is used for a faster training with comparalbe result.
RoBERTa is used to acquire the best pretrained model.

* bert-base-cased
* distilbert-base-cased
* roberta-base

## Install
``` bash
git clone https://github.com/yjang43/GLUE_CoLA.git
pip install -r requirements.txt
```

## Reproducing fine-tuned model

Open [https://colab.research.google.com/](https://colab.research.google.com/) and upload __model/colab_finetune.ipynb__ with __model/fun_glue.py__. 
Then running each cell will fine-tune models.

## Reproducing serialized model
Due to the large size of fine-tuned result, I decided to extract only models/tokenizer for memory efficiency (still large size though).
Reproduce with the following commandline.

``` bash
python model/serialization.py
```

## Run

``` bash
uvicorn main:app
```
Server open at the following link:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Use what FastAPI provides: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
