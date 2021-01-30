## Models
* bert-base-cased
* distilbert-base-cased
* roberta-base

I chose models with cased tokenization because capitalization is a part of acceptability(correct grammar) task.
BERT is used for a baseline.
DistilBERT is used for a faster training with comparalbe result.
RoBERTa is used to acquire the best pretrained model.

## Reproducing fine-tuned model

Open [https://colab.research.google.com/](https://colab.research.google.com/) and upload __colab_finetune.ipynb__ with __fun_glue.py__. 
Then running each cell will fine-tune models.

## Reproducing serialized model
Due to the large size of fine-tuned result, I decided to extract only models/tokenizer for memory efficiency (still large size though).
Reproduce with the following commandline.

``` bash
python model/serialization.py
```