import pickle
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer


FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def serialize(fn, fp):
    """ Model trained from Google Colab is too big, so extract model and tokenizer.

    [Parameters]
    fn: file name in str
    fp: file path in str
    """
    model = AutoModelForSequenceClassification.from_pretrained(fp)
    tokenizer = AutoTokenizer.from_pretrained(fp)
    obj = {'model': model, 'tokenizer': tokenizer}
    os.mkdir(os.path.join(FILE_DIR, 'models'))
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    print("********************************************************************************")
    print("Serialize models/tokenizers downloaded from Google Colab")
    print("********************************************************************************")
    serialize(os.path.join(FILE_DIR, 'models/bert-base-cased.pk'), os.path.join(FILE_DIR, 'models_colab/bert-base-cased/'))
    serialize(os.path.join(FILE_DIR, 'models/distilbert-base-cased.pk'), os.path.join(FILE_DIR, 'models_colab/distilbert-base-cased/'))
    serialize(os.path.join(FILE_DIR, 'models/roberta-base.pk'), os.path.join(FILE_DIR, 'models_colab/roberta-base/'))