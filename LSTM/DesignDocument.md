# LSTM 模型实现

## 项目概述

本项目旨在使用 Python 和 NumPy 从零开始实现一个长短期记忆（LSTM）网络。LSTM 是一种特殊的循环神经网络（RNN），能够学习长期依赖关系，广泛应用于序列数据处理任务，如自然语言处理、时间序列分析等。

此实现包括基本的 LSTM 单元、LSTM 层、序列到序列（Seq2Seq）模型架构、注意力机制以及相关的训练和评估组件。

## 功能特点

- **模块化设计**: 代码结构清晰，各组件（LSTM单元、层、Seq2Seq模型、注意力、优化器、损失函数、激活函数）分离，易于理解和扩展。
- **NumPy 实现**: 核心计算逻辑完全基于 NumPy，不依赖深度学习框架，有助于理解底层原理。
- **LSTM 单元与层**: 实现了标准的 LSTM 单元 (`LSTMCell`) 和可处理序列数据的 LSTM 层 (`LSTMLayer` 和 `VectorizedLSTMLayer`)。
- **Seq2Seq 架构**: 包含了编码器（Encoder）和解码器（Decoder）的实现，构成了 Seq2Seq 模型的基础。
- **注意力机制**: 实现了 Bahdanau 注意力机制 (`BahdanauAttention`)，增强 Seq2Seq 模型处理长序列的能力。
- **训练与评估**: 提供了 `Trainer` 类来管理模型的训练过程，并支持基本的评估。
- **可定制组件**: 包含可插拔的激活函数、损失函数和优化器。

## 模块组成

项目主要由以下模块构成（位于 `src` 目录下）：

- `lstm.py`: 定义了 `LSTMCell`（单个 LSTM 单元）和 `LSTMLayer`/`VectorizedLSTMLayer`（完整的 LSTM 层）。
- `seq2seq.py`: 实现了 `Encoder`、`Decoder` 类，以及将它们组合起来的 `Seq2Seq` 模型。
- `attention.py`: 实现了 `BahdanauAttention` 注意力机制。
- `training.py`: 包含 `Trainer` 类，用于模型的训练和评估流程。
- `model.py`: (可能包含基础的 `Model` 类，作为其他模型组件的基类)。
- `layers.py`: (可能包含如 `Embedding`、`Dense` 等基础层定义)。
- `activations.py`: 包含各种激活函数（如 Sigmoid, Tanh, ReLU, Softmax）的实现。
- `losses.py`: 包含各种损失函数（如均方误差, 交叉熵损失）的实现。
- `optimizers.py`: 包含优化算法（如 SGD, Adam）的实现。
- `initializers.py`: (可能包含权重初始化方法)。
- `utils.py`: (可能包含数据处理、评估指标如 BLEU 分数等辅助工具)。

## 核心组件详解

### 1. LSTM 单元 (`LSTMCell`)

`LSTMCell` 是 LSTM 网络的基本构建块，负责在单个时间步处理输入、前一隐藏状态和前一单元状态，并计算当前时间步的输出隐藏状态和单元状态。它内部包含输入门、遗忘门、输出门和候选单元状态的计算逻辑。

**关键代码片段 (`LSTM/src/lstm.py`):**

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # ... 权重初始化 ...

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x_t, h_prev, c_prev):
        # 输入门
        i_t = self.sigmoid(np.dot(x_t, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i)
        # 遗忘门
        f_t = self.sigmoid(np.dot(x_t, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f)
        # 输出门
        o_t = self.sigmoid(np.dot(x_t, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o)
        # 候选单元状态
        c_tilde_t = self.tanh(np.dot(x_t, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c)
        # 当前单元状态
        c_t = f_t * c_prev + i_t * c_tilde_t
        # 当前隐藏状态
        h_t = o_t * self.tanh(c_t)
        # ... cache for backward ...
        return h_t, c_t, cache

    def backward(self, dh_next, dc_next, cache):
        # ... 梯度计算 ...
        return gradients
```

### 2. LSTM 层 (`LSTMLayer` / `VectorizedLSTMLayer`)

`LSTMLayer` (或其向量化版本 `VectorizedLSTMLayer`) 将多个 `LSTMCell` 串联起来，以处理整个输入序列。它负责管理序列中每个时间步的状态传递，并可以配置为返回所有时间步的输出或仅返回最后一个时间步的输出。

**关键代码片段 (`LSTM/src/lstm.py` - `VectorizedLSTMLayer` 概念):**

```python
class VectorizedLSTMLayer(Layer):
    def __init__(self, units, return_sequences=True, return_state=False, **kwargs):
        # ... 初始化 ...
        self.lstm_cell = LSTMCell(input_dim_at_t, units) # 假设 input_dim 已知

    def forward(self, X_input, initial_h=None, initial_c=None, training=True):
        # X_input shape: (batch_size, sequence_length, input_features)
        batch_size, seq_len, _ = X_input.shape
        # ... 初始化 h 和 c ...
        
        H_all = np.zeros((batch_size, seq_len, self.units))
        C_all = np.zeros((batch_size, seq_len, self.units)) # For caching cell states if needed
        
        h_t = initial_h
        c_t = initial_c

        for t in range(seq_len):
            x_t = X_input[:, t, :]
            h_t, c_t, cache_cell = self.lstm_cell.forward(x_t, h_t, c_t) # 假设 cell 支持批处理
            H_all[:, t, :] = h_t
            C_all[:, t, :] = c_t
            # ... 存储缓存 ...
        
        # ... 根据 return_sequences 和 return_state 返回结果 ...
        return outputs, cache

    def backward(self, d_output, d_next_h, d_next_c, cache):
        # ... BPTT 实现 ...
        return param_grads, d_input, d_initial_h, d_initial_c
```

### 3. 编码器 (`Encoder`)

编码器是 Seq2Seq 模型的一部分，负责将输入序列（如源语言句子）编码成一个固定长度的上下文向量（context vector），通常是 LSTM 最后一个时间步的隐藏状态和单元状态。

**关键代码片段 (`LSTM/src/seq2seq.py`):**

```python
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, **kwargs):
        # ...
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = VectorizedLSTMLayer(units=lstm_units, return_sequences=False, return_state=True)
        # ...

    def forward(self, X_encoder, initial_states=None, training=True):
        embedded_input, _ = self.embedding.forward(X_encoder, training=training)
        lstm_output, cache_lstm = self.lstm.forward(embedded_input, initial_h=h_0, initial_c=c_0, training=training)
        final_hidden_state, final_cell_state = lstm_output
        # ...
        return final_hidden_state, final_cell_state
```

### 4. 解码器 (`Decoder`)

解码器接收编码器生成的上下文向量，并逐个生成输出序列中的元素（如目标语言句子中的词）。在每个时间步，它会考虑前一个生成的内容和当前的隐藏状态来预测下一个元素。

**关键代码片段 (`LSTM/src/seq2seq.py`):**

```python
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, output_dim, **kwargs):
        # ...
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = VectorizedLSTMLayer(units=lstm_units, return_sequences=True, return_state=True)
        self.dense = Dense(output_units=output_dim, activation='softmax')
        # ...

    def forward(self, X_decoder_token, initial_states, training=True):
        # X_decoder_token: 当前输入 token (batch_size, 1)
        # initial_states: (h_prev, c_prev) from encoder or previous step
        embedded_token, _ = self.embedding.forward(X_decoder_token, training=training)
        (all_h, next_h, next_c), _ = self.lstm.forward(embedded_token, initial_h=h_prev, initial_c=c_prev, training=training)
        lstm_output_for_dense = all_h[:, 0, :] # Get current step's LSTM output
        output_token_logits, _ = self.dense.forward(lstm_output_for_dense, training=training)
        return output_token_logits, next_h, next_c
```

### 5. Bahdanau 注意力机制 (`BahdanauAttention`)

Bahdanau 注意力（也称加性注意力）允许解码器在生成每个输出词时，动态地关注输入序列的不同部分。它通过计算一个对齐分数来衡量解码器当前状态与编码器各输出状态的相关性，然后基于这些分数对编码器输出进行加权求和，得到上下文向量。

**关键代码片段 (`LSTM/src/attention.py`):**

```python
class BahdanauAttention(Attention):
    def __init__(self, units):
        # ...
        self.W1 = Dense(units, activation=None) # For query (decoder state)
        self.W2 = Dense(units, activation=None) # For values (encoder states)
        self.V = Dense(1, activation=None)      # To compute score
        self.tanh_activation = get_activation('tanh')
        # ...

    def forward(self, query, values):
        # query: (batch_size, query_dim) - decoder hidden state
        # values: (batch_size, seq_len_encoder, values_dim) - encoder hidden states
        query_proj, _ = self.W1.forward(query)
        query_proj_expanded = np.expand_dims(query_proj, axis=1)
        
        values_reshaped = values.reshape(-1, values_dim)
        values_proj_reshaped, _ = self.W2.forward(values_reshaped)
        values_proj = values_proj_reshaped.reshape(batch_size, seq_len_encoder, self.units)
        
        score_input_tanh = self.tanh_activation(query_proj_expanded + values_proj)
        score_reshaped, _ = self.V.forward(score_input_tanh.reshape(-1, self.units))
        score = score_reshaped.reshape(batch_size, seq_len_encoder)
        
        self.attention_weights = np.exp(score - np.max(score, axis=1, keepdims=True)) # Softmax
        self.attention_weights /= np.sum(self.attention_weights, axis=1, keepdims=True)
        
        expanded_attention_weights = np.expand_dims(self.attention_weights, axis=2)
        self.context_vector = np.sum(expanded_attention_weights * values, axis=1)
        # ...
        return self.context_vector, self.attention_weights, cache
```

### 6. 训练器 (`Trainer`)

`Trainer` 类封装了模型训练的主要逻辑，包括数据批处理、前向传播、损失计算、反向传播和参数更新。它使得训练过程更加规范和易于管理。

**关键代码片段 (`LSTM/src/training.py`):**

```python
class Trainer:
    def __init__(self, model, optimizer, loss_function, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = get_loss(loss_function) if isinstance(loss_function, str) else loss_function
        # ...

    def train_step(self, X_batch, Y_batch):
        # For Seq2Seq: X_batch = (encoder_input, decoder_input)
        if isinstance(X_batch, tuple):
            predictions_logits, cache = self.model.forward(*X_batch, training=True)
        else:
            predictions_logits, cache = self.model.forward(X_batch, training=True)
        
        loss_value, d_loss_logits = self.loss_func(predictions_logits, Y_batch)
        param_gradients = self.model.backward(d_loss_logits, cache)
        all_params = self.model.get_parameters()
        self.optimizer.update(all_params, param_gradients)
        return loss_value, predictions_logits

    def fit(self, X_train, Y_train, epochs, batch_size, X_val=None, Y_val=None, ...):
        for epoch in range(epochs):
            # ... batch iteration ...
            for X_batch_current, Y_batch_current in batch_iterator(...):
                loss, _ = self.train_step(X_batch_current, Y_batch_current)
            # ... logging and validation ...
```

## 关键设计与实现

- **序列数据处理**: LSTM 网络的核心在于其处理序列数据的能力。通过内部的门控机制和单元状态，LSTM 能够有效地捕捉和传递序列中的长期依赖信息。
- **状态管理**: 在 `LSTMLayer` 和 Seq2Seq 模型中，隐藏状态 (hidden state) 和单元状态 (cell state) 在时间步之间的传递至关重要。编码器的最终状态通常作为解码器的初始状态。
- **BPTT (Backpropagation Through Time)**: LSTM 的训练依赖于 BPTT 算法，它将损失梯度从序列的末端反向传播到序列的开端，以更新网络权重。`backward` 方法在各层中实现了这一逻辑。
- **向量化操作**: 为了提高计算效率，许多操作（尤其是在 `VectorizedLSTMLayer` 中）都尽可能地使用了 NumPy 的向量化特性，以同时处理批次中的多个样本和序列中的多个时间步。
- **注意力机制集成**: Bahdanau 注意力机制被集成到 Seq2Seq 模型中，通过在解码的每一步动态计算上下文向量，使模型能够更灵活地关注输入序列的相关部分，尤其对于长序列翻译任务效果显著。
- **教师强制 (Teacher Forcing)**: 在 Seq2Seq 模型的训练过程中，通常使用教师强制策略，即在解码的每个时间步，使用真实的目标序列中的词作为下一个时间步的输入，而不是使用模型自己前一步的预测输出。这有助于稳定训练过程。
- **模块化与可扩展性**: 通过将不同的功能（如激活、损失、优化、层类型）分解到独立的模块和类中，整个框架更易于理解、维护和扩展新功能。

## 使用示例

```python
# (此处为伪代码，展示如何组装和使用模型)
# from LSTM.src.seq2seq import Seq2Seq
# from LSTM.src.optimizers import Adam
# from LSTM.src.training import Trainer

# 1. 准备数据 (X_encoder_train, X_decoder_input_train, Y_decoder_output_train)
# ... (数据预处理，词汇表构建，序列填充等) ...

# 2. 定义模型参数
input_vocab_size = 5000
target_vocab_size = 5000
embedding_dim = 128
lstm_units = 256

# 3. 初始化 Seq2Seq 模型
# model = Seq2Seq(input_vocab_size, target_vocab_size, embedding_dim, lstm_units, lstm_units)

# 4. 初始化优化器和损失函数
# optimizer = Adam(learning_rate=0.001)
# loss_function = 'categorical_crossentropy' # 或自定义损失对象

# 5. 初始化训练器
# trainer = Trainer(model, optimizer, loss_function)

# 6. 开始训练
# trainer.fit(
#     (X_encoder_train, X_decoder_input_train), 
#     Y_decoder_output_train, 
#     epochs=10, 
#     batch_size=64, 
#     # X_val=(X_encoder_val, X_decoder_input_val), 
#     # Y_val=Y_decoder_output_val
# )

# 7. 进行预测/推断
# input_sequence = ... # 准备单个输入序列
# predicted_sequence = model.predict(input_sequence, start_token_id, end_token_id, max_length=50)
# print(f"Predicted sequence: {predicted_sequence}")
```

## 训练与评估结果

(此部分用于展示模型在特定数据集上的训练曲线、损失变化、评估指标如 BLEU 分数等。)

**示例图表 (占位符):**

- **训练损失曲线**: `[Image of training_loss_vs_epochs.png]`
- **验证集 BLEU 分数**: `[Image of validation_bleu_score.png]`

## 未来工作

- 实现更高级的注意力机制（如 Luong Attention）。
- 增加对双向 LSTM (BiLSTM) 的支持。
- 实现更复杂的解码策略（如 Beam Search）。
- 完善梯度裁剪、正则化等训练技巧。
- 提供更详细的示例和预训练模型。
- 优化性能，例如通过 Cython 或其他方式加速计算密集部分。
- 增加对 GPU 计算的支持（目前仅 NumPy CPU）。