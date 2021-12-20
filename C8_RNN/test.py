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

dataset = pre_process_data(
    r'D:\Onedrive\OneDrive - alumni.albany.edu\Pycharm项目\Deep Learning 2021.10\深度学习用于文本和序列\aclImdb\aclImdb\train')
