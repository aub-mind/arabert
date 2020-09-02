#%%
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
ANERCorp_path = './ANERCorp_manually_cleaned.tsv'

with open(ANERCorp_path,'r',encoding='utf-8') as f:
  data = []
  sentence = []
  label = []
  for line in f:
    if line=='\n':
      if len(sentence) > 0:
        data.append((sentence,label))
        sentence = []
        label = []
      continue
    splits = line.split('\t')
    sentence.append(splits[0].replace('\u200f','').replace('·','.'))
    label.append(splits[1][:-1])
    # if you need to cut long sentence, instead of throwing them out in the creation of examples section
    # if len(sentence) > 100 and label[-1] == 'O':
    #   data.append((sentence,label))
    #   sentence = []
    #   label = []
  
  if len(sentence) > 0:
    data.append((sentence,label))

#%%

print('Number of sentences',len(data))
label_counter = Counter([ label for sentence in data for label in sentence[1]])
print('Label Count: ',label_counter)

#%%
plt.hist([len(sentence[0])for sentence in data],bins=range(0,256,3))
plt.show()

#%%
label_list = list(label_counter.keys())
print(label_list)

#%%
data_train_dev , data_test = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)
data_train , data_dev = train_test_split(data_train_dev, test_size=0.1, random_state=42, shuffle=True)

#%%
print(Counter([ label for sentence in data_test for label in sentence[1]]))
print(Counter([ label for sentence in data_dev for label in sentence[1]]))
print(Counter([ label for sentence in data_train for label in sentence[1]]))
# %%
with open('../ANERCorp_manually_cleaned_train.tsv','w',encoding='utf-8') as f:
    for row in data_train:
        for word, label in zip(row[0],row[1]):
            f.write(word+'\t'+label+'\n')
        f.write('\n')

with open('../ANERCorp_manually_cleaned_dev.tsv','w',encoding='utf-8') as f:
    for row in data_dev:
        for word, label in zip(row[0],row[1]):
            f.write(word+'\t'+label+'\n')
        f.write('\n')

with open('../ANERCorp_manually_cleaned_test.tsv','w',encoding='utf-8') as f:
    for row in data_test:
        for word, label in zip(row[0],row[1]):
            f.write(word+'\t'+label+'\n')
        f.write('\n')

