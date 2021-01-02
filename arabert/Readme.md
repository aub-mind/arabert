# AraBERT v1 & v2 : Pre-training BERT for Arabic Language Understanding
<img src="https://github.com/aub-mind/arabert/blob/master/arabert_logo.png" width="100" align="left"/>

**AraBERT** is an Arabic pretrained lanaguage model based on [Google's BERT architechture](https://github.com/google-research/bert). AraBERT uses the same BERT-Base config. More details are available in the [AraBERT Paper](https://arxiv.org/abs/2003.00104v2) and in the [AraBERT Meetup](https://github.com/WissamAntoun/pydata_khobar_meetup)

There are two versions of the model, AraBERTv0.1 and AraBERTv1, with the difference being that AraBERTv1 uses pre-segmented text where prefixes and suffixes were splitted using the [Farasa Segmenter](http://alt.qcri.org/farasa/segmenter.html).


We evalaute AraBERT models on different downstream tasks and compare them to [mBERT]((https://github.com/google-research/bert/blob/master/multilingual.md)), and other state of the art models (*To the extent of our knowledge*). The Tasks were Sentiment Analysis on 6 different datasets ([HARD](https://github.com/elnagara/HARD-Arabic-Dataset), [ASTD-Balanced](https://www.aclweb.org/anthology/D15-1299), [ArsenTD-Lev](https://staff.aub.edu.lb/~we07/Publications/ArSentD-LEV_Sentiment_Corpus.pdf), [LABR](https://github.com/mohamedadaly/LABR)), Named Entity Recognition with the [ANERcorp](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp), and Arabic Question Answering on [Arabic-SQuAD and ARCD](https://github.com/husseinmozannar/SOQAL)


## Results
Task | Metric | AraBERTv0.1 | AraBERTv1 | AraBERTv0.2-base | AraBERTv2-Base | AraBERTv0.2-large | AraBERTv2-large| AraELECTRA-Base
:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
HARD |Acc.|**96.2**|96.1|soon|soon|soon|soon|soon
ASTD |Acc.|92.2|**92.6**|soon|soon|soon|soon|soon
ArsenTD-Lev|Acc.|58.9|**59.4**|soon|soon|soon|soon|soon
AJGT|Acc.|93.1|**93.8**|soon|soon|soon|soon|soon
LABR|Acc.|85.9|**86.7**|soon|soon|soon|soon|soon
ANERcorp|macro-F1|**83.1**|82.4|soon|soon|soon|soon|soon
ARCD|EM - F1|31.6- 67.4|31.7 - 67.8|32.76 - 66.53|31.34 - 67.23|33.62 - 66.59|34.19 - 68.12 |**35.33 - 68.57**
TyDiQA-ar|EM - F1|58.31 - 78.91|61.11 - 79.36|60.67 - 79.63|61.67 - 81.66|61.56 - 82.38|64.49 - 82.51|**65.91 - 83.65**


## How to use

You can easily use AraBERT since it is almost fully compatible with existing codebases (Use this repo instead of the official BERT one, the only difference is in the ```tokenization.py``` file where we modify the _is_punctuation function to make it compatible with the "+" symbol and the "[" and "]" characters)


**AraBERTv1 an v2  always needs pre-segmentation**
```python
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

model_name = "bert-base-arabertv2"
arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

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

model_name = "bert-base-arabertv01"
arabert_tokenizer = AutoTokenizer.from_pretrained(model_name)
arabert_model = AutoModel.from_pretrained(model_name)

arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

arabert_tokenizer.tokenize(text_preprocessed)

>>> ['ولن', 'ن', '##بالغ', 'إذا', 'قلنا', 'إن', 'هاتف', 'أو', 'كمبيوتر', 'المكتب', 'في', 'زمن', '##ن', '##ا', 'هذا', 'ضروري']
```

**Examples**

`AraBERT_ANERCorp_CamelSplits.ipynb` is a demo of AraBERT for token classification on the ANERCorp dataset.

`araBERT_(Updated_Demo_TF).ipynb` is a demo using the AJGT dataset using TensorFlow Estimators (GPU and TPU compatible).

`AraBERT_PyTorch_Demo.ipynb` is a demo using the AJGT dataset using HuggingFace's Transformers API for PyTorch (GPU compatible)

`AraBERT_with_fast_bert.ipynb` is a demo using the AJGT dataset with Fast-Bert library

`AraBERT_Fill_Mask.ipynb` is a demo of the Masked Language capabilites and how it is better than other models that support Arabic

`AraBert_output_Embeddings_PyTorch.ipynb` is a demo on how to extract word embeddings fro sentences using the Transformers Library

`AraBERT_Text_Classification_with_HF_Trainer_Pytorch_GPU.ipynb` is a demo using the AJGT dataset using HuggingFace's Trainer API for PyTorch (GPU compatible) Note: TPU compatibility should be enabled in the `TrainingArguments` but not tested yet

`MTL_AraBERT_Offensive_Lanaguage_detection.ipynb`  is the code used in the in the [OSACT4 - shared task on Offensive language detection (LREC 2020)](http://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/). Paper [Link](https://www.aclweb.org/anthology/2020.osact-1.16/)

**AraBERT on ARCD**

During the preprocessing step the ```answer_start``` character position needs to be recalculated. You can use the file ```arcd_preprocessing.py``` as shown below to clean, preprocess the ARCD dataset before running ```run_squad.py```. More detailed Colab notebook is available in the [SOQAL repo](https://github.com/husseinmozannar/SOQAL).
```bash
python arcd_preprocessing.py \
    --input_file="/PATH_TO/arcd-test.json" \
    --output_file="arcd-test-pre.json" \
    --do_farasa_tokenization=True \
    --use_farasapy=True \
```
```bash
python run_squad.py \
  --vocab_file="/PATH_TO_PRETRAINED_TF_CKPT/vocab.txt" \
  --bert_config_file="/PATH_TO_PRETRAINED_TF_CKPT/config.json" \
  --init_checkpoint="/PATH_TO_PRETRAINED_TF_CKPT/" \
  --do_train=True \
  --train_file=turk_combined_all_pre.json \
  --do_predict=True \
  --predict_file=arcd-test-pre.json \
  --train_batch_size=32 \
  --predict_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=4 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --do_lower_case=False\
  --output_dir="/PATH_TO/OUTPUT_PATH"/ \
  --use_tpu=True \
  --tpu_name=$TPU_ADDRESS \
```
## Model Weights and Vocab Download

**You can find the PyTorch, TF2 and TF1 models in HuggingFace's Transformer Library under the ```aubmindlab``` username**

The TF1.x model are available in the HuggingFace models repo.
You can download them as follows:
- via git-lfs: clone all the models in a repo
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/aubmindlab/MODEL_NAME
tar -C ./MODEL_NAME -zxvf /content/MODEL_NAME/tf1_model.tar.gz
```
where `MODEL_NAME` is any model under the `aubmindlab` name

- via `wget`:
    - Go to the tf1_model.tar.gz file on huggingface.co/models/aubmindlab/MODEL_NAME.
    - copy the `oid sha256`
    - then run `wget  https://cdn-lfs.huggingface.co/aubmindlab/aragpt2-base/INSERT_THE_SHA_HERE` (ex: for `aragpt2-base`: `wget https://cdn-lfs.huggingface.co/aubmindlab/aragpt2-base/3766fc03d7c2593ff2fb991d275e96b81b0ecb2098b71ff315611d052ce65248`)