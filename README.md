# AraBERT : Pre-training BERT for Arabic Language Understanding
<img src="https://github.com/aub-mind/arabert/blob/master/arabert_logo.png" width="100" align="left"/>

**AraBERT** is an Arabic pretrained lanaguage model based on [Google's BERT architechture](https://github.com/google-research/bert). AraBERT uses the same BERT-Base config. More details are available in the [AraBERT Paper](https://arxiv.org/abs/2003.00104v2) and in the [AraBERT Meetup](https://github.com/WissamAntoun/pydata_khobar_meetup)

There are two versions of the model, AraBERTv0.1 and AraBERTv1, with the difference being that AraBERTv1 uses pre-segmented text where prefixes and suffixes were splitted using the [Farasa Segmenter](http://alt.qcri.org/farasa/segmenter.html).

The model was trained on ~70M sentences or ~23GB of Arabic text with ~3B words. The training corpora are a collection from publically available large scale raw arabic text ([Arabic Wikidumps](https://archive.org/details/arwiki-20190201), [The 1.5B words Arabic Corpus](https://www.semanticscholar.org/paper/1.5-billion-words-Arabic-Corpus-El-Khair/f3eeef4afb81223df96575adadf808fe7fe440b4), [The OSIAN Corpus](https://www.aclweb.org/anthology/W19-4619), Assafir news articles, and 4 other manually crawled news websites (Al-Akhbar, Annahar, AL-Ahram, AL-Wafd) from [the Wayback Machine](http://web.archive.org/))

We evalaute both AraBERT models on different downstream tasks and compare them to [mBERT]((https://github.com/google-research/bert/blob/master/multilingual.md)), and other state of the art models (*To the extent of our knowledge*). The Tasks were Sentiment Analysis on 6 different datasets ([HARD](https://github.com/elnagara/HARD-Arabic-Dataset), [ASTD-Balanced](https://www.aclweb.org/anthology/D15-1299), [ArsenTD-Lev](https://staff.aub.edu.lb/~we07/Publications/ArSentD-LEV_Sentiment_Corpus.pdf), [LABR](https://github.com/mohamedadaly/LABR)), Named Entity Recognition with the [ANERcorp](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp), and Arabic Question Answering on [Arabic-SQuAD and ARCD](https://github.com/husseinmozannar/SOQAL)

**Update 6 (19/11/2020):**
Added [ANERcorp](https://link.springer.com/chapter/10.1007/978-3-540-70939-8_13) with [CamelSplits](https://camel.abudhabi.nyu.edu/anercorp/) and updated the results table with the scores on the camel splits. You can also find our NER notebook under the `examples` folder.

We also added the ability to use the `lamb` optimizer from the albert paper. Which is used when training with batch_sizes>~2K.

**Update 5(2/9/2020):**
~~Added [ANERcorp](https://link.springer.com/chapter/10.1007/978-3-540-70939-8_13) dataset for Arabic NER which we manually cleaned and split into sentences (since the original dataset was missing sentence seperators). We notice that the NER score improved (results table was updated). We added the code used to generate the splits (80/10/10) and an example notebook for Arabic NER under the `examples` folder.~~

**Update 4 (7/7/2020) :**
Added TF2 models under the `aubmindlab` models on `HuggingFace`, you can now use `TFAutoModel` and work with Tensorflow 2

**Update 3 (1/7/2020) :**
You can now use the Transformers Library without adding the extra parameters to the `AutoTokenizer`, the tokens are now loaded automatically:
```python
#----NEW----
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess_arabert import preprocess

arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabert")

#----OLD----
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess_arabert import never_split_tokens, preprocess

arabert_tokenizer = AutoTokenizer.from_pretrained(
    "aubmindlab/bert-base-arabert",
    do_lower_case=False,
    do_basic_tokenize=True,
    never_split=never_split_tokens)
```

**Update 2 (21/5/2020) :**
Added support for the farasapy segmenter https://github.com/MagedSaeed/farasapy in the ``preprocess_arabert.py`` which is ~6x faster than the ``py4j.java_gateway``, consider setting ``use_farasapy=True`` when calling preprocess and pass it an instance of ``FarasaSegmenter(interactive=True)`` with interactive set to ``True`` for faster segmentation.

**Update 1 (21/4/2020) :**
Fixed an issue with ARCD fine-tuning ~~which drastically improved performance.~~ Initially we didn't account for the change of the ```answer_start``` during preprocessing. Results reverted to community submitted results [issue](https://github.com/aub-mind/arabert/issues/33)
## Results
Task | Metric |prev. SOTA | mBERT | AraBERTv0.1 | AraBERTv1
---|:---:|:---:|:---:|:---:|:---:
HARD |Acc.|95.7 [ElJundi et.al.](https://www.aclweb.org/anthology/W19-4608/)|95.7|**96.2**|96.1
ASTD |Acc.|86.5 [ElJundi et.al.](https://www.aclweb.org/anthology/W19-4608/)| 80.1|92.2|**92.6**
ArsenTD-Lev|Acc.|52.4 [ElJundi et.al.](https://www.aclweb.org/anthology/W19-4608/)|51|58.9|**59.4**
AJGT|Acc.|93 [Dahou et.al.](https://dl.acm.org/doi/fullHtml/10.1145/3314941)| 83.6|93.1|**93.8**
LABR|Acc.|**87.5** [Dahou et.al.](https://dl.acm.org/doi/fullHtml/10.1145/3314941)|83|85.9|86.7
ANERcorp|macro-F1|81.7 (BiLSTM-CRF)|78.4|**83.1**|82.4
ARCD|EM - F1|mBERT|**34.2** - 61.3|31.6- 67.4|31.7 - **67.8**

*If you tested AraBERT on a public dataset and you want to add your results to the table above, open a pull request or contact us. Also make sure to have your code available online so we can add it as a reference*

## How to use

You can easily use AraBERT since it is almost fully compatible with existing codebases (Use this repo instead of the official BERT one, the only difference is in the ```tokenization.py``` file where we modify the _is_punctuation function to make it compatible with the "+" symbol and the "[" and "]" characters)


**AraBERTv1 always needs pre-segmentation**
```python
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess_arabert import never_split_tokens, preprocess
from farasa.segmenter import FarasaSegmenter

arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabert")
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabert")

# OLD WAY
# arabert_tokenizer = AutoTokenizer.from_pretrained(
#     "aubmindlab/bert-base-arabert",
#     do_lower_case=False,
#     do_basic_tokenize=True,
#     never_split=never_split_tokens)
# arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabert")

#Preprocess the text to make it compatible with AraBERT using farasapy
farasa_segmenter = FarasaSegmenter(interactive=True)

#or you can use a py4j JavaGateway to the farasa Segmneter .jar but it's slower
#(see update 2)
#from py4j.java_gateway import JavaGateway
#gateway = JavaGateway.launch_gateway(classpath='./PATH_TO_FARASA/FarasaSegmenterJar.jar')
#farasa = gateway.jvm.com.qcri.farasa.segmenter.Farasa()

text = "ولن نبالغ إذا قلنا إن هاتف أو كمبيوتر المكتب في زمننا هذا ضروري"
text_preprocessed = preprocess( text,
                                do_farasa_tokenization = True,
                                farasa = farasa_segmenter,
                                use_farasapy = True)

>>>text_preprocessed: "و+ لن نبالغ إذا قل +نا إن هاتف أو كمبيوتر ال+ مكتب في زمن +نا هذا ضروري"

arabert_tokenizer.tokenize(text_preprocessed)

>>> ['و+', 'لن', 'نبال', '##غ', 'إذا', 'قل', '+نا', 'إن', 'هاتف', 'أو', 'كمبيوتر', 'ال+', 'مكتب', 'في', 'زمن', '+نا', 'هذا', 'ضروري']
```

**AraBERTv0.1 needs no pre-segmentation.**
```python
from transformers import AutoTokenizer, AutoModel

arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv01",do_lower_case=False)
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv01")

text = "ولن نبالغ إذا قلنا إن هاتف أو كمبيوتر المكتب في زمننا هذا ضروري"
arabert_tokenizer.tokenize(text)

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
Models | AraBERTv0.1 | AraBERTv1
---|:---:|:---:
TensorFlow|[Drive Link](https://drive.google.com/open?id=1-kVmTUZZ4DP2rzeHNjTPkY8OjnQCpomO) | [Drive Link](https://drive.google.com/open?id=1-d7-9ljKgDJP5mx73uBtio-TuUZCqZnt)
PyTorch| [Drive_Link](https://drive.google.com/open?id=1-_3te42mQCPD8SxwZ3l-VBL7yaJH-IOv)| [Drive_Link](https://drive.google.com/open?id=1-69s6Pxqbi63HOQ1M9wTcr-Ovc6PWLLo)

**You can find the PyTorch models in HuggingFace's Transformer Library under the ```aubmindlab``` username**

## If you used this model please cite us as :
Google Scholar has our Bibtex wrong (missing name), use this instead
```
@inproceedings{antoun2020arabert,
  title={AraBERT: Transformer-based Model for Arabic Language Understanding},
  author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
  booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
  pages={9}
}
```
## Acknowledgments
Thanks to TensorFlow Research Cloud (TFRC) for the free access to Cloud TPUs, couldn't have done it without this program, and to the [AUB MIND Lab](https://sites.aub.edu.lb/mindlab/) Members for the continous support. Also thanks to [Yakshof](https://www.yakshof.com/#/) and Assafir for data and storage access. Another thanks for Habib Rahal (https://www.behance.net/rahalhabib), for putting a face to AraBERT.

## Contacts
**Wissam Antoun**: [Linkedin](https://www.linkedin.com/in/giulio-ravasio-3a81a9110/) | [Twitter](https://twitter.com/wissam_antoun) | [Github](https://github.com/WissamAntoun) | <wfa07@mail.aub.edu> | <wissam.antoun@gmail.com>

**Fady Baly**: [Linkedin](https://www.linkedin.com/in/fadybaly/) | [Twitter](https://twitter.com/fadybaly) | [Github](https://github.com/fadybaly) | <fgb06@mail.aub.edu> | <baly.fady@gmail.com>

