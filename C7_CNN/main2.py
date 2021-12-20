import numpy as np
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
import glob
import os
from random import shuffle


def pre_process_data(filepath):
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    pos_label = 1
    neg_label = 0
    dataset = []
    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as f:
            dataset.append([pos_label, f.read()])
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as f:
            dataset.append((neg_label, f.read()))
    shuffle(dataset)
    return dataset


dataset = pre_process_data(
    r'D:\Onedrive\OneDrive - alumni.albany.edu\Pycharm项目\Deep Learning 2021.10\深度学习用于文本和序列\aclImdb\aclImdb\train')

word_vectors = KeyedVectors.load_word2vec_format(
    r"D:\Onedrive\OneDrive - alumni.albany.edu\Pycharm项目\GoogleNews-vectors-negative300.bin.gz", binary=True)


def tokenizer_and_vectorize(filename, tokenizers):
    sample_vec = []
    with open(filename, 'r') as f:
        tokens = tokenizers.tokenize(f.read())
        for token in tokens:
            try:
                sample_vec.append(word_vectors[token])
            except KeyError:
                pass
    return sample_vec


split_point = int(len(dataset) * 0.8)
trainData = dataset[:split_point]
testData = dataset[split_point:]
tokenizer = TreebankWordTokenizer()


def pad_trunc(data, maxlen=400):
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    if len(data) > maxlen:
        temp = data[:maxlen]
    elif len(data) < maxlen:
        temp = data
        additional_elems = maxlen - len(data)
        for _ in range(additional_elems):
            temp.append(zero_vector)
    else:
        temp = data
    return temp


max_len = 400
batch_size = 32
embedding_dims = 300
filter = 250
kernal_size = 3
hidden_dims = 250
epochs = 2


def data_generator(data_store, tokenizers, batchsize=32, maxlen=400, embedding_dims=300):
    X, Y = [], []
    while True:
        for i in range(len(data_store)):
            if (i % batchsize == 0 and X and Y) or (i == len(data_store)):
                X = np.reshape(X, (len(X), maxlen, embedding_dims))
                Y = np.array(Y)
                yield X, Y
                X, Y = [], []
            x, y = data_store[i][1], data_store[i][0]
            x = pad_trunc(tokenizer_and_vectorize(x, tokenizers), maxlen=maxlen)
            X.append(x)
            Y.append(y)


model = Sequential()
model.add(
    Conv1D(filter, kernal_size, padding='valid', activation='relu', strides=1, input_shape=(max_len, embedding_dims)))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_test = data_generator(testData, tokenizer, batchsize=batch_size, maxlen=max_len)

import math

# history=model.fit_generator(generator=data_generator(trainData,tokenizer,batchsize=batch_size,maxlen=maxlen),steps_per_epoch=math.ceil(trainlen/batch_size),epochs=2,validation_data=X_test,validation_steps=math.ceil(testlen/batch_size))

history = model.fit(data_generator(trainData, tokenizer, batchsize=batch_size, maxlen=max_len),
                    steps_per_epoch=math.ceil(len(trainData) / batch_size), epochs=10, validation_data=X_test,
                    validation_steps=math.ceil(len(testData) / batch_size))

model_structure = model.to_json()
with open('cnn_model.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('cnn_weights.h5')
