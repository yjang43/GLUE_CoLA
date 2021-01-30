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
cd GLUE_CoLA/
pip install -r requirements.txt
```

## Reproducing fine-tuned model

Create a directory where fine-tuned model will be downloaded.
``` bash
mkdir model/models_colab
```
Open [https://colab.research.google.com/](https://colab.research.google.com/) and upload __model/colab_finetune.ipynb__ with __model/fun_glue.py__. 
Then running each cell will fine-tune models.
Because the model will be saved on your Google Drive under the following file names, 'bert-base-cased.zip', 'distilbert-base-cased.zip', and 'roberta-base.zip', make sure to provide a correct verification code.
Download the zip files from your Google Drive, and unzip them to directory you craeted, model/models_colab/

## Reproducing serialized model
Due to the large size of fine-tuned result, I decided to extract only models/tokenizer for memory efficiency (still large size though).
Reproduce with the following command line.

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
