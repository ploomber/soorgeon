# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### All what you need to apply Named Entity Recognition (NER)

# %% [markdown]
# **Named Entity Recognition:** is the task of identifying and categorizing key information (entities) in text. An entity can be any word or series of words that consistently refers to the same thing.

# %% [markdown]
# Find the Aggregated dataset at: https://www.kaggle.com/naseralqaydeh/named-entity-recognition-ner-corpus

# %% [markdown]
# ## 1- Importing Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# %% [markdown]
# ## 2- Importing Data

# %%
data_path = "../input/entity-annotated-corpus/ner_dataset.csv"

data = pd.read_csv(data_path, encoding='unicode_escape')
# filling the first column that determines which sentence each word belongs to.
data.fillna(method='ffill', inplace=True)
data.head()

# %%
ready_dist_path = "../input/named-entity-recognition-ner-corpus/ner.csv"
ready_data = pd.read_csv(ready_dist_path)
ready_data.head()

# %% [markdown]
# ## 3- Get to know our data


# %%
def join_a_sentence(sentence_number, data):
    """
    Args.:
          sentence_number: sentence number we want to join and return. 
          
    Returns:
          The joined sentence.
    """

    sentence_number = str(sentence_number)
    the_sentence_words_list = list(data[
        data['Sentence #'] == 'Sentence: {}'.format(sentence_number)]['Word'])

    return ' '.join(the_sentence_words_list)


# %%
join_a_sentence(sentence_number=1, data=data)

# %%
join_a_sentence(sentence_number=100, data=data)

# %%
# Data Shape
data.shape

# %%
# Number of unique sentences
len(np.unique(data['Sentence #']))

# %%
print("Number of unique words in the dataset: {}".format(data.Word.nunique()))
print("Number of unique tags in the dataset: {}".format(data.Tag.nunique()))

# %%
tags = data.Tag.unique()
tags


# %%
def num_words_tags(tags, data):
    """This functions takes the tags we want to count and the datafram 
    and return a dict where the key is the tag and the value is the frequency
    of that tag"""

    tags_count = {}

    for tag in tags:
        len_tag = len(data[data['Tag'] == tag])
        tags_count[tag] = len_tag

    return tags_count


# %%
tags_count = num_words_tags(tags, data)
tags_count

# %%
plt.figure(figsize=(10, 6))
plt.hist(data.Tag, log=True, label='Tags', color='olive', bins=50)
plt.xlabel('Tags', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title("Tags Frequency", fontsize=20)
plt.grid(alpha=0.3)
plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# **Code that I used to produce  ready_data**

# %%
# def process_Data():

#     data_dict = {}

#     for sn in range(1, len(np.unique(data['Sentence #']))+1):

#         all_sen_data = []

#         se_data = data[data['Sentence #']  == 'Sentence: {}'.format(sn)]
#         sentence = ' '.join(list(se_data['Word']))
#         all_sen_data.append(sentence)

#         sen_pos = list(se_data['POS'])
#         all_sen_data.append(sen_pos)

#         sen_tags = list(se_data['Tag'])
#         all_sen_data.append(sen_tags)

#         data_dict['Sentence: {}'.format(sn)] = all_sen_data

#         if sn % 10000 == 0:
#             print("{} sentences are processed".format(sn))

#     return data_dict

# %% [markdown]
# ## 4- Data Preprocessing

# %%
ready_data.head()

# %%
X = list(ready_data['Sentence'])
Y = list(ready_data['Tag'])

# %%
from ast import literal_eval

Y_ready = []

for sen_tags in Y:
    Y_ready.append(literal_eval(sen_tags))

# %%
print("First three sentences: \n")
print(X[:3])

# %%
print("First three Tags: \n")
print(Y_ready[:3])

# %% [markdown]
# We need to tokenize the sentences by mapping each word to a unique identifier, then we need to pad them because NN need the input sentences to have the same lenght.

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
print("Number of examples: {}".format(len(X)))

# %% [markdown]
# - **Toknize sentences**

# %%
# cutoff reviews after 110 words
maxlen = 110

# consider the top 36000 words in the dataset
max_words = 36000

# tokenize each sentence in the dataset
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

# %%
word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))
ind2word = dict([(value, key) for (key, value) in word_index.items()])

# %%
word2id = word_index

# %%
# dict. that map each identifier to its word
id2word = {}
for key, value in word2id.items():
    id2word[value] = key

# %% [markdown]
# - **Sentences padding**

# %%
# pad the sequences so that all sequences are of the same size
X_preprocessed = pad_sequences(sequences, maxlen=maxlen, padding='post')

# %%
# first example after tokenization and padding.
X_preprocessed[0]

# %%
# 22479 example after tokenization and padding.
X_preprocessed[22479]

# %% [markdown]
# - **Preprocess tags**

# %% [markdown]
# we need to preprocess tags by assigning a unique identifier for each one of them.
#
# Since also tags for each example have different lenght we need to fine a way to slove this problem.
#
# - We can assign a new tag for the zeros that we used in padding
# - We can use the O tag for them.
#
# I will try the second choice of using the O tag to pad the tag list.

# %%
# dict. that map each tag to its identifier
tags2id = {}
for i, tag in enumerate(tags):
    tags2id[tag] = i

# %%
tags2id

# %%
# dict. that map each identifier to its tag
id2tag = {}
for key, value in tags2id.items():
    id2tag[value] = key

# %%
id2tag


# %%
def preprocess_tags(tags2id, Y_ready):

    Y_preprocessed = []
    maxlen = 110
    # for each target
    for y in Y_ready:

        # place holder to store the new preprocessed tag list
        Y_place_holder = []

        # for each tag in rhe tag list
        for tag in y:
            # append the id of the tag in the place holder list
            Y_place_holder.append(tags2id[tag])

        # find the lenght of the new preprocessed tag list
        len_new_tag_list = len(Y_place_holder)
        # find the differance in length between the len of tag list and padded sentences
        num_O_to_add = maxlen - len_new_tag_list

        # add 'O's to padd the tag lists
        padded_tags = Y_place_holder + ([tags2id['O']] * num_O_to_add)
        Y_preprocessed.append(padded_tags)

    return Y_preprocessed


# %%
Y_preprocessed = preprocess_tags(tags2id, Y_ready)

# %%
print(Y_preprocessed[0])

# %%
print(Y_ready[0])

# %% [markdown]
# ### By now we have the data ready for training our model
#
#
# **We have X_preprocessed and Y_preprocessed that we will use to train our model**

# %% [markdown]
# The las step is to **split** the data into:
#
# - Training dataset
# - Valisdation dataset
# - testing dataset

# %% [markdown]
# - **Data shuffling and splitting**

# %%
print("The Lenght of training examples: {}".format(len(X_preprocessed)))
print("The Lenght of training targets: {}".format(len(Y_preprocessed)))

# %%
X_preprocessed = np.asarray(X_preprocessed)
Y_preprocessed = np.asarray(Y_preprocessed)

# %%
# 70% of the datat will be used for training
training_samples = 0.7
# 15% of the datat will be used for validation
validation_samples = 0.15
# 15% of the datat will be used for testing
testing_samples = 0.15

# %%
indices = np.arange(len(Y_preprocessed))

# %%
np.random.seed(seed=555)
np.random.shuffle(indices)

# %%
X_preprocessed = X_preprocessed[indices]
Y_preprocessed = Y_preprocessed[indices]

# %%
X_train = X_preprocessed[:int(0.7 * len(X_preprocessed))]
print("Number of training examples: {}".format(len(X_train)))

X_val = X_preprocessed[int(0.7 *
                           len(X_preprocessed)):int(0.7 *
                                                    len(X_preprocessed)) +
                       (int(0.15 * len(X_preprocessed)) + 1)]
print("Number of validation examples: {}".format(len(X_val)))

X_test = X_preprocessed[int(0.7 * len(X_preprocessed)) +
                        (int(0.15 * len(X_preprocessed)) + 1):]
print("Number of testing examples: {}".format(len(X_test)))

Y_train = Y_preprocessed[:int(0.7 * len(X_preprocessed))]
Y_val = Y_preprocessed[int(0.7 *
                           len(X_preprocessed)):int(0.7 *
                                                    len(X_preprocessed)) +
                       (int(0.15 * len(X_preprocessed)) + 1)]
Y_test = Y_preprocessed[int(0.7 * len(X_preprocessed)) +
                        (int(0.15 * len(X_preprocessed)) + 1):]

print("Total number of examples after shuffling and splitting: {}".format(
    len(X_train) + len(X_val) + len(X_test)))

# %% [markdown]
# ## 5- Model Training and Evaluation

# %%
X_train[1000]

# %%
Y_train[1000]

# %%
id2word[729]

# %% [markdown]
# ## Load dataset to the model using train_dataset = tf.data.Dataset
#

# %% _kg_hide-output=true
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# %%
BATCH_SIZE = 132
SHUFFLE_BUFFER_SIZE = 132

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# %%
embedding_dim = 300
maxlen = 110
max_words = 36000
num_tags = len(tags)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=100,
                             activation='tanh',
                             return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=100,
                             activation='tanh',
                             return_sequences=True)),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_tags, activation='softmax'))
])

# %%
model.summary()

# %%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# %%
history = model.fit(train_dataset, validation_data=val_dataset, epochs=15)

# %%
model.evaluate(test_dataset)

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 4), dpi=80)

ax[0].plot(epochs, acc, label="Training Accuracy", color='darkblue')
ax[0].plot(epochs, val_acc, label="Validation Accuracy", color='darkgreen')
ax[0].grid(alpha=0.3)
ax[0].title.set_text('Training Vs Validation Accuracy')
ax[0].fill_between(epochs, acc, val_acc, color='crimson', alpha=0.3)
plt.setp(ax[0], xlabel='Epochs')
plt.setp(ax[0], ylabel='Accuracy')

ax[1].plot(epochs, loss, label="Training Loss", color='darkblue')
ax[1].plot(epochs, val_loss, label="Validation Loss", color='darkgreen')
ax[1].grid(alpha=0.3)
ax[1].title.set_text('Training Vs Validation Loss')
ax[1].fill_between(epochs, loss, val_loss, color='crimson', alpha=0.3)
plt.setp(ax[1], xlabel='Epochs')
plt.setp(ax[1], ylabel='Loss')

plt.show()


# %%
def make_prediction(model, preprocessed_sentence, id2word, id2tag):

    #if preprocessed_sentence.shape() != (1, 110):
    preprocessed_sentence = preprocessed_sentence.reshape((1, 110))

    # return preprocessed sentence to its orginal form
    sentence = preprocessed_sentence[preprocessed_sentence > 0]
    word_list = []
    for word in list(sentence):
        word_list.append(id2word[word])
    orginal_sententce = ' '.join(word_list)

    len_orginal_sententce = len(word_list)

    # make prediction
    prediction = model.predict(preprocessed_sentence)
    prediction = np.argmax(prediction[0], axis=1)

    # return the prediction to its orginal form
    prediction = list(prediction)[:len_orginal_sententce]

    pred_tag_list = []
    for tag_id in prediction:
        pred_tag_list.append(id2tag[tag_id])

    return orginal_sententce, pred_tag_list


# %%
orginal_sententce, pred_tag_list = make_prediction(
    model=model,
    preprocessed_sentence=X_test[520],
    id2word=id2word,
    id2tag=id2tag)

# %%
print(orginal_sententce)

# %%
print(pred_tag_list)

# %% [markdown]
# ## 6- Thank you

# %% [markdown]
# **Thank you for reading, I hope you enjoyed and benefited from it.**
#
# **If you have any questions or notes please leave it in the comment section.**
#
# **If you like this notebook please press upvote and thanks again.**
