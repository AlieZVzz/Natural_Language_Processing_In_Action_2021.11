import numpy as np
# from tensorflow.keras.preprocessing import sequence
import tensorflow.keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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


word_vectors = KeyedVectors.load_word2vec_format(
    r"D:\Onedrive\OneDrive - alumni.albany.edu\Pycharm Project\GoogleNews-vectors-negative300.bin.gz", binary=True,
    limit=10000)


# word_vector = get_data('word2vec')


def tokenizer_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    # expected=[]
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


def pad_trunc(data, maxlen):
    embedding_dims = 300
    return [smp[:maxlen] + [[0.] * embedding_dims] * (maxlen - len(smp)) for smp in data]


dataset = pre_process_data(
    r'D:\Onedrive\OneDrive - alumni.albany.edu\Pycharm Project\Deep Learning 2021.10\深度学习用于文本和序列\aclImdb\aclImdb\train')
vectorized_data = tokenizer_and_vectorize(dataset)

expected = collect_expected(dataset)
split_point = int(len(vectorized_data) * 0.8)
x_train = vectorized_data[:5000]
y_train = expected[:5000]

x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

max_len = 100
batch_size = 32
embedding_dims = 300
epochs = 2
num_nerons = 50

x_train = pad_trunc(x_train, max_len)
x_test = pad_trunc(x_test, max_len)

x_train = np.reshape(x_train, (len(x_train), max_len, embedding_dims))
x_test = np.reshape(x_test, (len(x_test), max_len, embedding_dims))
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(tensorflow.keras.layers.SimpleRNN(num_nerons, return_sequences=True, input_shape=(max_len, embedding_dims)))
model.add(Dropout(0.2))
model.add(tensorflow.keras.layers.Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
model_structure = model.to_json()
with open('simple_rnn_model.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('simple_rnn_weights.h5')
