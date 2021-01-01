# Scripts used to pre_process and create the data for classifier evaluation
#%%
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")

from arabert.preprocess import ArabertPreprocessor


from tqdm import tqdm

tqdm.pandas()

from tokenization import FullTokenizer
from run_classifier import input_fn_builder, model_fn_builder


model_name = "bert-base-arabert"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)


class Dataset:
    def __init__(
        self,
        name,
        train,
        test,
        label_list,
        train_InputExamples=None,
        test_InputExamples=None,
        train_features=None,
        test_features=None,
    ):
        self.name = name
        self.train = train
        self.test = test
        self.label_list = label_list
        self.train_InputExamples = train_InputExamples
        self.test_InputExamples = test_InputExamples
        self.train_features = train_features
        self.test_features = test_features


all_datasets = []
#%%
# *************HARD************
df_HARD = pd.read_csv("Datasets\\HARD\\balanced-reviews-utf8.tsv", sep="\t", header=0)

df_HARD = df_HARD[["rating", "review"]]  # we are interested in rating and review only
# code rating as +ve if > 3, -ve if less, no 3s in dataset
df_HARD["rating"] = df_HARD["rating"].apply(lambda x: 0 if x < 3 else 1)
# rename columns to fit default constructor in fastai
df_HARD.columns = ["label", "text"]
df_HARD["text"] = df_HARD["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
train_HARD, test_HARD = train_test_split(df_HARD, test_size=0.2, random_state=42)
label_list_HARD = [0, 1]

data_Hard = Dataset("HARD", train_HARD, test_HARD, label_list_HARD)
all_datasets.append(data_Hard)

#%%
# *************ASTD-Unbalanced************
df_ASTD_UN = pd.read_csv(
    "Datasets\\ASTD-master\\data\\Tweets.txt", sep="\t", header=None
)

DATA_COLUMN = "text"
LABEL_COLUMN = "label"
df_ASTD_UN.columns = [DATA_COLUMN, LABEL_COLUMN]

df_ASTD_UN[LABEL_COLUMN] = df_ASTD_UN[LABEL_COLUMN].apply(
    lambda x: 0 if (x == "NEG") else x
)
df_ASTD_UN[LABEL_COLUMN] = df_ASTD_UN[LABEL_COLUMN].apply(
    lambda x: 1 if (x == "POS") else x
)
df_ASTD_UN[LABEL_COLUMN] = df_ASTD_UN[LABEL_COLUMN].apply(
    lambda x: 2 if (x == "NEUTRAL") else x
)
df_ASTD_UN[LABEL_COLUMN] = df_ASTD_UN[LABEL_COLUMN].apply(
    lambda x: 3 if (x == "OBJ") else x
)
df_ASTD_UN["text"] = df_ASTD_UN["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
train_ASTD_UN, test_ASTD_UN = train_test_split(
    df_ASTD_UN, test_size=0.2, random_state=42
)
label_list_ASTD_UN = [0, 1, 2, 3]

data_ASTD_UN = Dataset(
    "ASTD-Unbalanced", train_ASTD_UN, test_ASTD_UN, label_list_ASTD_UN
)
all_datasets.append(data_ASTD_UN)
#%%
# *************ASTD-Dahou-Balanced************

df_ASTD_B = pd.read_csv(
    "Datasets\\Dahou\\data_csv_balanced\\ASTD-balanced-not-linked.csv",
    sep=",",
    header=0,
)

df_ASTD_B.columns = [DATA_COLUMN, LABEL_COLUMN]

df_ASTD_B[LABEL_COLUMN] = df_ASTD_B[LABEL_COLUMN].apply(lambda x: 0 if (x == -1) else x)
df_ASTD_B["text"] = df_ASTD_B["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
train_ASTD_B, test_ASTD_B = train_test_split(df_ASTD_B, test_size=0.2, random_state=42)
label_list_ASTD_B = [0, 1]

data_ASTD_B = Dataset(
    "ASTD-Dahou-Balanced", train_ASTD_B, test_ASTD_B, label_list_ASTD_B
)
all_datasets.append(data_ASTD_B)

#%%
# *************ArSenTD-LEV************
df_ArSenTD = pd.read_csv(
    "Datasets\\ArSenTD-LEV\\ArSenTD-LEV-processed-no-emojis2.csv", sep=",", header=0
)

df_ArSenTD.columns = [DATA_COLUMN, LABEL_COLUMN]

df_ArSenTD[LABEL_COLUMN] = df_ArSenTD[LABEL_COLUMN].apply(
    lambda x: 0 if (x == "very_negative") else x
)
df_ArSenTD[LABEL_COLUMN] = df_ArSenTD[LABEL_COLUMN].apply(
    lambda x: 1 if (x == "negative") else x
)
df_ArSenTD[LABEL_COLUMN] = df_ArSenTD[LABEL_COLUMN].apply(
    lambda x: 2 if (x == "neutral") else x
)
df_ArSenTD[LABEL_COLUMN] = df_ArSenTD[LABEL_COLUMN].apply(
    lambda x: 3 if (x == "positive") else x
)
df_ArSenTD[LABEL_COLUMN] = df_ArSenTD[LABEL_COLUMN].apply(
    lambda x: 4 if (x == "very_positive") else x
)
df_ArSenTD["text"] = df_ArSenTD["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
label_list_ArSenTD = [0, 1, 2, 3, 4]

train_ArSenTD, test_ArSenTD = train_test_split(
    df_ArSenTD, test_size=0.2, random_state=42
)

data_ArSenTD = Dataset("ArSenTD-LEV", train_ArSenTD, test_ArSenTD, label_list_ArSenTD)
all_datasets.append(data_ArSenTD)

#%%
# *************AJGT************
df_AJGT = pd.read_excel("Datasets\\Ajgt\\AJGT.xlsx", header=0)

df_AJGT = df_AJGT[["Feed", "Sentiment"]]
df_AJGT.columns = [DATA_COLUMN, LABEL_COLUMN]

df_AJGT[LABEL_COLUMN] = df_AJGT[LABEL_COLUMN].apply(
    lambda x: 0 if (x == "Negative") else x
)
df_AJGT[LABEL_COLUMN] = df_AJGT[LABEL_COLUMN].apply(
    lambda x: 1 if (x == "Positive") else x
)
df_AJGT["text"] = df_AJGT["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
train_AJGT, test_AJGT = train_test_split(df_AJGT, test_size=0.2, random_state=42)
label_list_AJGT = [0, 1]

data_AJGT = Dataset("AJGT", train_AJGT, test_AJGT, label_list_AJGT)
all_datasets.append(data_AJGT)
#%%
# *************LABR-UN-Binary************
from labr import LABR

labr_helper = LABR()

(d_train, y_train, d_test, y_test) = labr_helper.get_train_test(
    klass="2", balanced="unbalanced"
)

train_LABR_B_U = pd.DataFrame({"text": d_train, "label": y_train})
test_LABR_B_U = pd.DataFrame({"text": d_test, "label": y_test})

train_LABR_B_U["text"] = train_LABR_B_U["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
test_LABR_B_U["text"] = test_LABR_B_U["text"].progress_apply(
    lambda x: arabert_prep.preprocess(
        x
    )
)
label_list_LABR_B_U = [0, 1]

data_LABR_B_U = Dataset(
    "LABR-UN-Binary", train_LABR_B_U, test_LABR_B_U, label_list_LABR_B_U
)
# all_datasets.append(data_LABR_B_U)

#%%
for data in tqdm(all_datasets):
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    data.train_InputExamples = data.train.apply(
        lambda x: run_classifier.InputExample(
            guid=None,  # Globally unique ID for bookkeeping, unused in this example
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN],
        ),
        axis=1,
    )

    data.test_InputExamples = data.test.apply(
        lambda x: run_classifier.InputExample(
            guid=None, text_a=x[DATA_COLUMN], text_b=None, label=x[LABEL_COLUMN]
        ),
        axis=1,
    )
#%%
# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 256

VOC_FNAME = "./64000_vocab_sp_70m.txt"
tokenizer = FullTokenizer(VOC_FNAME)

for data in tqdm(all_datasets):
    # Convert our train and test features to InputFeatures that BERT understands.
    data.train_features = run_classifier.convert_examples_to_features(
        data.train_InputExamples, data.label_list, MAX_SEQ_LENGTH, tokenizer
    )
    data.test_features = run_classifier.convert_examples_to_features(
        data.test_InputExamples, data.label_list, MAX_SEQ_LENGTH, tokenizer
    )

# %%
import pickle

with open("all_datasets_64k_farasa_256.pickle", "wb") as fp:  # Pickling
    pickle.dump(all_datasets, fp)


# %%
