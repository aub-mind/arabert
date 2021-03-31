# AraBERT v1 & v2 : Pre-training BERT for Arabic Language Understanding
<img src="https://github.com/aub-mind/arabert/blob/master/arabert_logo.png" width="100" align="left"/>

**AraBERT** is an Arabic pretrained lanaguage model based on [Google's BERT architechture](https://github.com/google-research/bert). AraBERT uses the same BERT-Base config. More details are available in the [AraBERT Paper](https://arxiv.org/abs/2003.00104v2) and in the [AraBERT Meetup](https://github.com/WissamAntoun/pydata_khobar_meetup)

There are two versions of the model, AraBERTv0.1 and AraBERTv1, with the difference being that AraBERTv1 uses pre-segmented text where prefixes and suffixes were splitted using the [Farasa Segmenter](http://alt.qcri.org/farasa/segmenter.html).


We evalaute AraBERT models on different downstream tasks and compare them to [mBERT]((https://github.com/google-research/bert/blob/master/multilingual.md)), and other state of the art models (*To the extent of our knowledge*). The Tasks were Sentiment Analysis on 6 different datasets ([HARD](https://github.com/elnagara/HARD-Arabic-Dataset), [ASTD-Balanced](https://www.aclweb.org/anthology/D15-1299), [ArsenTD-Lev](https://staff.aub.edu.lb/~we07/Publications/ArSentD-LEV_Sentiment_Corpus.pdf), [LABR](https://github.com/mohamedadaly/LABR)), Named Entity Recognition with the [ANERcorp](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp), and Arabic Question Answering on [Arabic-SQuAD and ARCD](https://github.com/husseinmozannar/SOQAL)


## Results
Task | Metric | AraBERTv0.1 | AraBERTv1 | AraBERTv0.2-base | AraBERTv2-Base | AraBERTv0.2-large | AraBERTv2-large| AraELECTRA-Base
:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
HARD |Acc.|**96.2**|96.1|-|-|-|-|-
ASTD |Acc.|92.2|**92.6**|-|-|-|-|-
ArsenTD-Lev|macro-f1|53.56|-|55.71|-|56.94|-|**57.20**
AJGT|Acc.|93.1|**93.8**|-|-|-|-|-
LABR|Acc.|85.9|**86.7**|-|-|-|-|-
ANERcorp|macro-F1|83.1|82.4|83.70|-|83.08|-|**83.95**
ARCD|EM - F1|31.62 - 67.45|31.7 - 67.8|32.76 - 66.53|31.34 - 67.23|36.89 - **71.32**|34.19 - 68.12|**37.03** - 71.22
TyDiQA-ar|EM - F1|68.51 - 82.86|- |73.07 - 85.41|-|73.72 - 86.03|-|**74.91 - 86.68**


## How to use

You can easily use AraBERT since it is almost fully compatible with existing codebases (Use this repo instead of the official BERT one, the only difference is in the ```tokenization.py``` file where we modify the _is_punctuation function to make it compatible with the "+" symbol and the "[" and "]" characters)


**AraBERTv1 an v2  always needs pre-segmentation**
```python
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

model_name = "aubmindlab/bert-base-arabertv2"
arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

arabert_prep = ArabertPreprocessor(model_name=model_name)

text = "ولن نبالغ إذا قلنا إن هاتف أو كمبيوتر المكتب في زمننا هذا ضروري"
arabert_prep.preprocess(text)
>>>"و+ لن نبالغ إذا قل +نا إن هاتف أو كمبيوتر ال+ مكتب في زمن +نا هذا ضروري"

arabert_tokenizer.tokenize(text_preprocessed)

>>> ['و+', 'لن', 'نبال', '##غ', 'إذا', 'قل', '+نا', 'إن', 'هاتف', 'أو', 'كمبيوتر', 'ال+', 'مكتب', 'في', 'زمن', '+نا', 'هذا', 'ضروري']
```

**AraBERTv0.1 and v0.2 needs no pre-segmentation.**
```python
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv01",do_lower_case=False)
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv01")

text = "ولن نبالغ إذا قلنا إن هاتف أو كمبيوتر المكتب في زمننا هذا ضروري"

model_name = "aubmindlab/bert-base-arabertv01"
arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

arabert_prep = ArabertPreprocessor(model_name=model_name)

arabert_tokenizer.tokenize(text_preprocessed)

>>> ['ولن', 'ن', '##بالغ', 'إذا', 'قلنا', 'إن', 'هاتف', 'أو', 'كمبيوتر', 'المكتب', 'في', 'زمن', '##ن', '##ا', 'هذا', 'ضروري']
```

## Model Weights and Vocab Download

**You can find the PyTorch, TF2 and TF1 models in HuggingFace's Transformer Library under the ```aubmindlab``` username**

- `wget https://huggingface.co/aubmindlab/MODEL_NAME/resolve/main/tf1_model.tar.gz` where `MODEL_NAME` is any model under the `aubmindlab` name
