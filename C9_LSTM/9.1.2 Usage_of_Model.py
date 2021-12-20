from tensorflow.keras.models import model_from_json
import numpy as np

import C9_LSTM.LSTM

with open("lstm_model.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

model.load_weights("lstm_weights.h5")
sample_1 = "I hate that dismal weather had me down for so long, when will it break! Ugh, when does happiness return? " \
           "The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend. "
vec_list = C9_LSTM.LSTM.tokenizer_and_vectorize([(1, sample_1)])
test_vec_list = C9_LSTM.LSTM.pad_trunc(vec_list, C9_LSTM.LSTM.max_len)

test_vec = np.reshape(test_vec_list, (len(test_vec_list), C9_LSTM.LSTM.max_len, C9_LSTM.LSTM.embedding_dims))
print("Sample's Sentiment, 1 -pos, 2-neg : {}".format(model.predict_classes(test_vec)))
