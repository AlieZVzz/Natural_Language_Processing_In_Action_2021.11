from nlpia.loaders import get_data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

df = get_data('moviedialog')
# 数组保存从语料库文件中读取的输入文本和目标文本
input_texts, target_texts = [], []
# 这个集合保存输入文本和目标文本中出现的字符
input_vocabulary = set()
output_vocabulary = set()
# 起始和终止的字符
start_token = '\t'
stop_token = '\n'
# max_training_samples定义了训练使用的行数。它是用户定义的最大值和文件中加载总行数中较小的数
max_training_samples = min(25000, len(df) - 1)

for input_text, target_text in zip(df.statement, df.reply):
    # target_text需要用起始字符和终止字符包装起来
    target_text = start_token + target_text + stop_token
    input_texts.append(input_text)
    target_texts.append(target_text)
    # 编译词汇表--input text中出现过的唯一字符集合
    for char in input_text:
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)

# 将字符集转换为排序后的字符列表，然后使用该列表生成字典
input_vocabulary = sorted(input_vocabulary)
output_vocabulary = sorted(output_vocabulary)
# 确定唯一字符的数量，用于构建独热矩阵
input_vocab_size = len(input_vocabulary)
output_vocab_size = len(output_vocabulary)
# 需要确定序列词条的最大数量
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
# 创建查找字典，用于生成独热向量
input_token_index = dict([(char, i) for i, char in enumerate(input_vocabulary)])
target_token_index = dict([(char, i) for i, char in enumerate(output_vocabulary)])
# 循环创建的字典创建反向查找表
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# 初始化0张量 [num_samples, max_len_seq, num_unique_tokens_in_vocab]
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, input_vocab_size),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, output_vocab_size),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, output_vocab_size),
    dtype='float32')

# 对训练样本进行循环遍历，输入文本和目标文本需要对应
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # 循环遍历每个样本的每个字符
    for t, char in enumerate(input_text):
        # 将每个时刻字符的索引设置为1，其他索引为0；浙江创建训练样本的独热编码
        encoder_input_data[i, t, input_token_index[char]] = 1.
    # 对解码器的训练数据，创建decoder_input_data和decoder_target_data
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1

batch_size = 64
epochs = 100
# 神经元数量为256
num_neurons = 256

encoder_inputs = Input(shape=(None, input_vocab_size))
encoder = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(num_neurons, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
# 每个epoch之后，留下10%的样本用于验证测试
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs,
          validation_split=0.1)
model.save('chatbot.h5')

encoder_model = Model(encoder_inputs, encoder_states)
thought_input = [Input(shape=(num_neurons,)), Input(shape=(num_neurons,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=thought_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(inputs=[decoder_inputs] + thought_input, outputs=[decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # 生成思想向量作为解码器的输入
    thought = encoder_model.predict(input_seq)
    # 与训练相反，target_seq一开始是一个零张量
    target_seq = np.zeros((1, 1, output_vocab_size))
    # 解码器的第一个输入词条是初始词条
    target_seq[0, 0, target_token_index[stop_token]] = 1.
    stop_condition = False
    generated_sequence = ''

    while not stop_condition:
        # 将已生成的词条和最新状态传递给解码器，以预测下一个序列元素
        output_tokens, h, c = decoder_model.predict([target_seq] + thought)

        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = reverse_target_char_index[generated_token_idx]
        generated_sequence += generated_char
        # stop_condition = True 停止循环
        if (generated_char == stop_token or len(generated_sequence) > max_decoder_seq_length):
            stop_condition = True
        # 更新目标序列，并使用最后生成的词条作为下一生成步骤的输入
        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, generated_token_idx] = 1.
        # 更新thought向量状态
        thought = [h, c]

    return generated_sequence


def response(input_text):
    input_seq = np.zeros((1, max_encoder_seq_length, input_vocab_size), dtype='float32')
    # 对输入文本的每个字符进行循环遍历，生成独热张量，以便编码器中生成thought向量
    for t, char in enumerate(input_text):
        input_seq[0, t, input_token_index[char]] = 1.
    # 使用decode_sequence函数调用训练好的模型生成回复序列
    decode_sentence = decode_sequence(input_seq)
    print('Reply (Decode sentence):', decode_sentence)


