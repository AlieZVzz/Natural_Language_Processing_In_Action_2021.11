from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

output_vocab_size = 500
num_neurons = 20
encoder_inputs = Input(shape=(None, output_vovab_size))
encoder = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = (state_h, state_c)

decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(num_neurons, return_state=True, return_sequences=True)
decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
