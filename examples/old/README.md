**Examples**

`AraBERT_ANERCorp_CamelSplits.ipynb` is a demo of AraBERT for token classification on the ANERCorp dataset.

`araBERT_(Updated_Demo_TF).ipynb` is a demo using the AJGT dataset using TensorFlow Estimators (GPU and TPU compatible).

`AraBERT_with_fast_bert.ipynb` is a demo using the AJGT dataset with Fast-Bert library

`AraBERT_Fill_Mask.ipynb` is a demo of the Masked Language capabilites and how it is better than other models that support Arabic

`AraBERT_Text_Classification_with_HF_Trainer_Pytorch_GPU.ipynb` is a demo using the AJGT dataset using HuggingFace's Trainer API for PyTorch (GPU compatible) Note: TPU compatibility should be enabled in the `TrainingArguments` but not tested yet

`MTL_AraBERT_Offensive_Lanaguage_detection.ipynb`  is the code used in the in the [OSACT4 - shared task on Offensive language detection (LREC 2020)](http://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/). Paper [Link](https://www.aclweb.org/anthology/2020.osact-1.16/)