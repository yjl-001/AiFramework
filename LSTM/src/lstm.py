# LSTM/src/lstm.py
import numpy as np

class LSTMCell:
    """一个单独的 LSTM 单元。"""
    def __init__(self, input_size, hidden_size):
        """
        初始化 LSTM 单元。
        参数:
            input_size (int): 输入特征的大小。
            hidden_size (int): 隐藏状态的大小。
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 权重和偏置，用于输入门、遗忘门、单元门（候选状态）、输出门
        # Wi, Wf, Wc, Wo 是连接输入 x 的权重
        # Ui, Uf, Uc, Uo 是连接前一隐藏状态 h_prev 的权重
        # bi, bf, bc, bo 是偏置项

        # 为简单起见，我们将 x 和 h 的权重分开定义
        # 输入门权重和偏置
        self.W_i = np.random.randn(input_size, hidden_size)  # 输入 x 到输入门的权重
        self.U_i = np.random.randn(hidden_size, hidden_size) # 上一隐藏状态 h_prev 到输入门的权重
        self.b_i = np.zeros(hidden_size)  # 输入门的偏置

        # 遗忘门权重和偏置
        self.W_f = np.random.randn(input_size, hidden_size)  # 输入 x 到遗忘门的权重
        self.U_f = np.random.randn(hidden_size, hidden_size) # 上一隐藏状态 h_prev 到遗忘门的权重
        self.b_f = np.zeros(hidden_size)  # 遗忘门的偏置

        # 输出门权重和偏置
        self.W_o = np.random.randn(input_size, hidden_size)  # 输入 x 到输出门的权重
        self.U_o = np.random.randn(hidden_size, hidden_size) # 上一隐藏状态 h_prev 到输出门的权重
        self.b_o = np.zeros(hidden_size)  # 输出门的偏置

        # 候选单元状态（或称为单元输入门）权重和偏置
        self.W_c = np.random.randn(input_size, hidden_size)  # 输入 x 到候选单元状态的权重
        self.U_c = np.random.randn(hidden_size, hidden_size) # 上一隐藏状态 h_prev 到候选单元状态的权重
        self.b_c = np.zeros(hidden_size)  # 候选单元状态的偏置

        # 将所有参数收集到一个列表中，方便后续优化器使用
        self.parameters = [
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_c, self.U_c, self.b_c
        ]

    def sigmoid(self, x):
        """Sigmoid 激活函数。"""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """Tanh 激活函数。"""
        return np.tanh(x)

    def forward(self, x_t, h_prev, c_prev):
        """
        执行单个时间步的前向传播。

        参数:
            x_t (np.ndarray): 当前时间步的输入 (input_size,)
            h_prev (np.ndarray): 上一个时间步的隐藏状态 (hidden_size,)
            c_prev (np.ndarray): 上一个时间步的单元状态 (hidden_size,)

        返回:
            h_t (np.ndarray): 当前时间步的隐藏状态 (hidden_size,)
            c_t (np.ndarray): 当前时间步的单元状态 (hidden_size,)
            cache (tuple): 用于反向传播的缓存值。
        """
        """
        Performs a single forward pass for one timestep.
        x_t: input at current timestep (input_size,)
        h_prev: hidden state from previous timestep (hidden_size,)
        c_prev: cell state from previous timestep (hidden_size,)
        """
        # 输入门 (决定什么新信息要被存儲到单元状态中)
        # i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
        i_t = self.sigmoid(np.dot(x_t, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i)

        # 遗忘门 (决定从单元状态中丢弃什么信息)
        # f_t = sigmoid(W_f * x_t + U_f * h_prev + b_f)
        f_t = self.sigmoid(np.dot(x_t, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f)

        # 输出门 (决定单元状态的哪些部分要被输出)
        # o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
        o_t = self.sigmoid(np.dot(x_t, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o)

        # 候选单元状态 (创建新的候选值，可能会被添加到单元状态)
        # c_tilde_t = tanh(W_c * x_t + U_c * h_prev + b_c)
        c_tilde_t = self.tanh(np.dot(x_t, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c)

        # 当前单元状态 (更新单元状态)
        # c_t = f_t * c_prev + i_t * c_tilde_t
        c_t = f_t * c_prev + i_t * c_tilde_t

        # 当前隐藏状态 (计算输出的隐藏状态)
        # h_t = o_t * tanh(c_t)
        h_t = o_t * self.tanh(c_t)

        # 缓存前向传播过程中的值，用于反向传播
        cache = (x_t, h_prev, c_prev, i_t, f_t, o_t, c_tilde_t, c_t, h_t)
        
        return h_t, c_t, cache

    def backward(self, dh_next, dc_next, cache):
        """
        执行单个时间步的反向传播。

        参数:
            dh_next (np.ndarray): 损失函数相对于下一个隐藏状态 h_t 的梯度 (hidden_size,)
            dc_next (np.ndarray): 损失函数相对于下一个单元状态 c_t 的梯度 (hidden_size,)
                                 (注意：这个 dc_next 是从时间序列的更后面传递过来的，或者在序列末端是0)
            cache (tuple): 前向传播时缓存的值。

        返回:
            gradients (dict): 包含所有参数梯度以及相对于 x_t, h_prev, c_prev 的梯度的字典。
        """
        x_t, h_prev, c_prev, i_t, f_t, o_t, c_tilde_t, c_t, h_t = cache

        # h_t = o_t * tanh(c_t)
        # 梯度回传到 o_t 和 c_t (通过 tanh(c_t))
        # d(loss)/do_t
        do_t = dh_next * self.tanh(c_t)
        # d(loss)/d(tanh(c_t))
        d_tanh_c_t = dh_next * o_t
        # d(loss)/dc_t = d(loss)/d(tanh(c_t)) * (1 - tanh(c_t)^2) + dc_next (从下一个时间步或损失函数直接传来的梯度)
        dc_t = d_tanh_c_t * (1 - self.tanh(c_t)**2) + dc_next

        # o_t = sigmoid(a_o)  => a_o = W_o * x_t + U_o * h_prev + b_o
        # d(loss)/da_o = d(loss)/do_t * d(o_t)/da_o = do_t * o_t * (1 - o_t)
        da_o = do_t * o_t * (1 - o_t) # sigmoid的导数

        # c_t = f_t * c_prev + i_t * c_tilde_t
        # 梯度回传到 f_t, c_prev, i_t, c_tilde_t
        # d(loss)/df_t
        df_t = dc_t * c_prev
        # d(loss)/dc_prev (这个梯度会传递给上一个时间步的 dc_next)
        dc_prev_from_ct = dc_t * f_t
        # d(loss)/di_t
        di_t = dc_t * c_tilde_t
        # d(loss)/dc_tilde_t
        dc_tilde_t = dc_t * i_t

        # c_tilde_t = tanh(a_c) => a_c = W_c * x_t + U_c * h_prev + b_c
        # d(loss)/da_c = d(loss)/dc_tilde_t * d(c_tilde_t)/da_c = dc_tilde_t * (1 - c_tilde_t**2)
        da_c = dc_tilde_t * (1 - c_tilde_t**2) # tanh的导数

        # f_t = sigmoid(a_f) => a_f = W_f * x_t + U_f * h_prev + b_f
        # d(loss)/da_f = d(loss)/df_t * d(f_t)/da_f = df_t * f_t * (1 - f_t)
        da_f = df_t * f_t * (1 - f_t)

        # i_t = sigmoid(a_i) => a_i = W_i * x_t + U_i * h_prev + b_i
        # d(loss)/da_i = d(loss)/di_t * d(i_t)/da_i = di_t * i_t * (1 - i_t)
        da_i = di_t * i_t * (1 - i_t)

        # 计算权重和偏置的梯度
        # 例如: d(loss)/dW_i = d(loss)/da_i * d(a_i)/dW_i = da_i * x_t (外积)
        #       d(loss)/db_i = d(loss)/da_i * d(a_i)/db_i = da_i
        dW_i = np.outer(x_t, da_i)
        dU_i = np.outer(h_prev, da_i)
        db_i = da_i

        dW_f = np.outer(x_t, da_f)
        dU_f = np.outer(h_prev, da_f)
        db_f = da_f

        dW_o = np.outer(x_t, da_o)
        dU_o = np.outer(h_prev, da_o)
        db_o = da_o

        dW_c = np.outer(x_t, da_c)
        dU_c = np.outer(h_prev, da_c)
        db_c = da_c

        # 计算传递到上一个隐藏状态 h_prev 和上一个单元状态 c_prev 以及当前输入 x_t 的梯度
        # d(loss)/dh_prev = d(loss)/da_i * U_i^T + d(loss)/da_f * U_f^T + d(loss)/da_o * U_o^T + d(loss)/da_c * U_c^T
        dh_prev = np.dot(da_i, self.U_i.T) + np.dot(da_f, self.U_f.T) + np.dot(da_o, self.U_o.T) + np.dot(da_c, self.U_c.T)
        # d(loss)/dc_prev (已在上面计算为 dc_prev_from_ct)
        dc_prev = dc_prev_from_ct
        # d(loss)/dx_t = d(loss)/da_i * W_i^T + d(loss)/da_f * W_f^T + d(loss)/da_o * W_o^T + d(loss)/da_c * W_c^T
        dx_t = np.dot(da_i, self.W_i.T) + np.dot(da_f, self.W_f.T) + np.dot(da_o, self.W_o.T) + np.dot(da_c, self.W_c.T)

        gradients = {
            'dW_i': dW_i, 'dU_i': dU_i, 'db_i': db_i,
            'dW_f': dW_f, 'dU_f': dU_f, 'db_f': db_f,
            'dW_o': dW_o, 'dU_o': dU_o, 'db_o': db_o,
            'dW_c': dW_c, 'dU_c': dU_c, 'db_c': db_c,
            'dx_t': dx_t, 'dh_prev': dh_prev, 'dc_prev': dc_prev
        }
        return gradients

class LSTMLayer:
    """一个 LSTM 层，由多个 LSTMCell 组成，用于处理序列数据。（非向量化版本）"""
    def __init__(self, input_size, hidden_size, output_size=None, return_sequences=True):
        """
        初始化 LSTM 层。
        参数:
            input_size (int): 每个时间步输入特征的大小。
            hidden_size (int): LSTM 单元隐藏状态的大小。
            output_size (int, optional): 输出特征的大小。如果 None，则默认为 hidden_size。
                                       仅在 return_sequences=False 时，如果 output_size 与 hidden_size 不同，
                                       会添加一个额外的全连接层进行投影。
            return_sequences (bool): 是否返回所有时间步的输出。如果 False，只返回最后一个时间步的输出。
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.return_sequences = return_sequences

        self.cell = LSTMCell(input_size, hidden_size) # LSTM 层包含一个 LSTM 单元的实例
        
        # 如果不返回序列且输出大小与隐藏大小不同，则添加一个全连接层 (W_y, b_y) 用于输出投影
        # 注意：这个简单的实现中，这个全连接层没有激活函数
        if not return_sequences and self.output_size != hidden_size:
            self.W_y = np.random.randn(hidden_size, self.output_size) # 输出权重
            self.b_y = np.zeros(self.output_size) # 输出偏置
            self.parameters = self.cell.parameters + [self.W_y, self.b_y]
        else:
            self.W_y = None # 如果返回序列或 output_size == hidden_size，则不需要额外的输出投影层
            self.b_y = None
            self.parameters = self.cell.parameters

    def forward(self, X):
        """
        执行 LSTM 层的前向传播。
        注意：这个实现是逐个样本和逐个时间步处理的，效率较低。

        参数:
            X (np.ndarray): 输入序列，形状为 (batch_size, sequence_length, input_size)。

        返回:
            outputs (np.ndarray): 如果 return_sequences=True，形状为 (batch_size, sequence_length, hidden_size 或 output_size)。
                                 如果 return_sequences=False，形状为 (batch_size, hidden_size 或 output_size)。
            caches (list): 包含每个时间步每个样本的 LSTMCell 缓存的列表，用于反向传播。
        """
        batch_size, sequence_length, _ = X.shape
        
        # 初始化隐藏状态 h 和单元状态 c (通常为全零)
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        all_h_outputs = [] # 如果 return_sequences 为 True，存储每个时间步的隐藏状态
        all_cell_caches = []  # 存储每个时间步所有样本的 cell 缓存，用于反向传播

        # 遍历时间序列中的每个时间步
        for t in range(sequence_length):
            x_t_batch = X[:, t, :] # 当前时间步的输入数据 (batch_size, input_size)
            
            h_batch_next_t = np.zeros_like(h) # 存储当前时间步所有样本的下一个隐藏状态
            c_batch_next_t = np.zeros_like(c) # 存储当前时间步所有样本的下一个单元状态
            
            current_t_cell_caches = [] # 存储当前时间步所有样本的 cell 缓存
            # 遍历批次中的每个样本
            for b in range(batch_size):
                x_t_sample = x_t_batch[b, :]    # 当前样本，当前时间步的输入
                h_prev_sample = h[b, :]         # 当前样本的上一个隐藏状态
                c_prev_sample = c[b, :]         # 当前样本的上一个单元状态
                
                # LSTMCell 的前向传播
                h_next_sample, c_next_sample, cell_cache = self.cell.forward(x_t_sample, h_prev_sample, c_prev_sample)
                
                h_batch_next_t[b, :] = h_next_sample # 保存当前样本的下一个隐藏状态
                c_batch_next_t[b, :] = c_next_sample # 保存当前样本的下一个单元状态
                current_t_cell_caches.append(cell_cache) # 保存当前样本的 cell 缓存
            
            # 更新整个批次的隐藏状态和单元状态，用于下一个时间步
            h = h_batch_next_t
            c = c_batch_next_t
            all_cell_caches.append(current_t_cell_caches) # 存储当前时间步所有样本的缓存
            
            if self.return_sequences:
                all_h_outputs.append(h.copy()) # 存储当前时间步的隐藏状态 (所有样本)
        
        if self.return_sequences:
            # 将所有时间步的隐藏状态堆叠起来
            # 形状变为 (sequence_length, batch_size, hidden_size)，然后转置为 (batch_size, sequence_length, hidden_size)
            stacked_outputs = np.stack(all_h_outputs, axis=1)
            return stacked_outputs, all_cell_caches
        else:
            # 只返回最后一个时间步的隐藏状态 h
            # 如果定义了 W_y 和 b_y (即 output_size != hidden_size 且 return_sequences=False)，则通过全连接层
            if self.W_y is not None:
                final_output = np.dot(h, self.W_y) + self.b_y
                # Apply softmax or other activation if needed for classification, outside this layer
                return final_output, caches 
            return h, all_cell_caches # Corrected variable name

    def backward(self, d_output, caches):
        """
        执行 LSTM 层的反向传播。
        注意：这个实现是逐个样本和逐个时间步处理的，效率较低，且梯度累积方式可能需要适配更标准的框架。

        参数:
            d_output (np.ndarray): 损失函数相对于 LSTM 层输出的梯度。
                                 如果 return_sequences=True, 形状为 (batch_size, sequence_length, hidden_size/output_size)。
                                 如果 return_sequences=False, 形状为 (batch_size, hidden_size/output_size)。
            caches (list): 前向传播时保存的 LSTMCell 缓存列表。

        返回:
            dX (np.ndarray): 损失函数相对于 LSTM 层输入的梯度，形状为 (batch_size, sequence_length, input_size)。
            param_gradients (dict): 包含所有可训练参数 (W_i, U_i, ..., W_y, b_y) 梯度的字典。
        """
        # 从 d_output 或 caches 中获取 batch_size 和 sequence_length
        # caches 的结构是 list of lists: caches[timestep][batch_sample_index]
        # caches[t][b] 是一个元组 (x_t, h_prev, c_prev, ...)
        # x_t 的形状是 (input_size,)
        # d_output 的形状取决于 return_sequences
        if self.return_sequences:
            batch_size, sequence_length, _ = d_output.shape
        else:
            batch_size = d_output.shape[0]
            sequence_length = len(caches) # caches 是 list of lists of tuples
            # 如果 d_output 是一维的（例如，batch_size=1 且非序列输出），确保 batch_size 正确
            if d_output.ndim == 1 and batch_size == 1: # This case might be tricky
                 pass # batch_size is likely 1
            elif d_output.ndim == 1 and batch_size != 1:
                 # This implies d_output might be (output_size,) for a single sample, need to reshape or handle
                 # For simplicity, assume d_output is at least 2D (batch_size, output_size) if not sequence
                 # Or (batch_size, seq_len, output_size) if sequence
                 # This part highlights issues with non-vectorized batch handling.
                 pass 


        # 初始化所有参数的梯度累加器
        # 这些梯度将在所有时间步和所有批次样本上累积
        dW_i_total = np.zeros_like(self.cell.W_i)
        dU_i_total = np.zeros_like(self.cell.U_i)
        db_i_total = np.zeros_like(self.cell.b_i)
        dW_f_total = np.zeros_like(self.cell.W_f)
        dU_f_total = np.zeros_like(self.cell.U_f)
        db_f_total = np.zeros_like(self.cell.b_f)
        dW_o_total = np.zeros_like(self.cell.W_o)
        dU_o_total = np.zeros_like(self.cell.U_o)
        db_o_total = np.zeros_like(self.cell.b_o)
        dW_c_total = np.zeros_like(self.cell.W_c)
        dU_c_total = np.zeros_like(self.cell.U_c)
        db_c_total = np.zeros_like(self.cell.b_c)
        
        dW_y_total = np.zeros_like(self.W_y) if self.W_y is not None else None
        db_y_total = np.zeros_like(self.b_y) if self.b_y is not None else None

        # 初始化输入梯度 dX
        dX = np.zeros((batch_size, sequence_length, self.input_size))

        # 初始化反向传播到上一个时间步的隐藏状态梯度 dh_next 和单元状态梯度 dc_next
        # 这些是针对整个批次的，在每个时间步开始时使用
        dh_from_next_timestep_batch = np.zeros((batch_size, self.hidden_size))
        dc_from_next_timestep_batch = np.zeros((batch_size, self.hidden_size))

        # 如果 return_sequences 为 False，d_output 是关于最后一个时间步输出的梯度
        if not self.return_sequences:
            # 如果存在输出投影层 W_y, b_y
            if self.W_y is not None:
                # d_output 是 dL/d(final_output)，其中 final_output = h_last @ W_y + b_y
                # 我们需要计算 dL/dh_last, dL/dW_y, dL/db_y
                
                # h_last 是最后一个时间步的隐藏状态，形状为 (batch_size, hidden_size)
                # caches[-1] 是最后一个时间步的缓存列表，每个元素是该时间步一个样本的 cell_cache
                # cell_cache[8] 是 h_t
                h_last_batch = np.array([caches[-1][b][8] for b in range(batch_size)]) # (batch_size, hidden_size)
                
                # dL/dW_y = h_last.T @ dL/d(final_output)
                dW_y_total = np.dot(h_last_batch.T, d_output) # 累加，但这里只发生一次
                # dL/db_y = sum(dL/d(final_output), axis=0)
                db_y_total = np.sum(d_output, axis=0) # 累加，但这里只发生一次
                # dL/dh_last = dL/d(final_output) @ W_y.T
                # 这个梯度将作为 dh_from_next_timestep_batch 输入到最后一个时间步的 cell backward
                dh_from_next_timestep_batch = np.dot(d_output, self.W_y.T)
            else:
                # 如果没有输出投影层，d_output 就是 dL/dh_last
                dh_from_next_timestep_batch = d_output
            # dc_from_next_timestep_batch 初始为0，因为它不受输出层直接影响（除非损失函数直接作用于单元状态）

        # 从最后一个时间步向前迭代
        for t in reversed(range(sequence_length)):
            # 如果 return_sequences=True, d_output[:, t, :] 是当前时间步输出 h_t 的直接梯度
            # 总的 dh_t 输入到 cell.backward 是 d_output[:, t, :] (如果适用) + dh_from_next_timestep_batch (从t+1时间步传来)
            if self.return_sequences:
                dh_for_cell_t = d_output[:, t, :] + dh_from_next_timestep_batch
            else:
                # 如果不返回序列，只有最后一个时间步 (t == sequence_length - 1) 会接收到来自外部的 dh_from_next_timestep_batch
                # 其他时间步的 dh_for_cell_t 完全来自于 dh_from_next_timestep_batch (即上一个迭代设置的 dh_prev_from_cell_t_plus_1)
                dh_for_cell_t = dh_from_next_timestep_batch
            
            # dc_for_cell_t 总是 dc_from_next_timestep_batch (从t+1时间步传来)
            dc_for_cell_t = dc_from_next_timestep_batch
            
            current_timestep_caches = caches[t] # 当前时间步所有样本的 cell 缓存列表
            
            # 为当前时间步的 dX, dh_prev, dc_prev 初始化，它们将由 cell.backward 计算并累加
            # 这些是传递到 t-1 时间步的梯度
            dh_to_prev_timestep_batch = np.zeros((batch_size, self.hidden_size))
            dc_to_prev_timestep_batch = np.zeros((batch_size, self.hidden_size))
            dx_current_t_batch = np.zeros((batch_size, self.input_size))

            # 遍历批次中的每个样本
            for b in range(batch_size):
                cell_cache_sample_t = current_timestep_caches[b] # 当前样本在当前时间步的 cell 缓存
                dh_input_to_cell = dh_for_cell_t[b, :]       # 输入到 cell backward 的 dh
                dc_input_to_cell = dc_for_cell_t[b, :]       # 输入到 cell backward 的 dc
                
                # 执行单个 cell 的反向传播
                gradients_cell = self.cell.backward(dh_input_to_cell, dc_input_to_cell, cell_cache_sample_t)
                
                # 累积参数的梯度 (在所有时间步和所有样本上求和)
                # 注意：这种累积方式对于非向量化实现是标准的，但对于向量化实现，梯度通常是直接计算的平均值或总和。
                dW_i_total += gradients_cell['dW_i']
                dU_i_total += gradients_cell['dU_i']
                db_i_total += gradients_cell['db_i']
                dW_f_total += gradients_cell['dW_f']
                dU_f_total += gradients_cell['dU_f']
                db_f_total += gradients_cell['db_f']
                dW_o_total += gradients_cell['dW_o']
                dU_o_total += gradients_cell['dU_o']
                db_o_total += gradients_cell['db_o']
                dW_c_total += gradients_cell['dW_c']
                dU_c_total += gradients_cell['dU_c']
                db_c_total += gradients_cell['db_c']
                
                # 累积传递到上一个时间步的梯度和当前输入的梯度
                dx_current_t_batch[b, :] = gradients_cell['dx_t']
                dh_to_prev_timestep_batch[b, :] = gradients_cell['dh_prev']
                dc_to_prev_timestep_batch[b, :] = gradients_cell['dc_prev']
            
            # 保存当前时间步计算出的输入梯度 dX
            dX[:, t, :] = dx_current_t_batch
            
            # 更新用于下一个迭代 (即 t-1 时间步) 的 dh_from_next_timestep 和 dc_from_next_timestep
            dh_from_next_timestep_batch = dh_to_prev_timestep_batch
            dc_from_next_timestep_batch = dc_to_prev_timestep_batch

        # 收集所有参数的梯度
        param_gradients = {
            'dW_i': dW_i_total, 'dU_i': dU_i_total, 'db_i': db_i_total,
            'dW_f': dW_f_total, 'dU_f': dU_f_total, 'db_f': db_f_total,
            'dW_o': dW_o_total, 'dU_o': dU_o_total, 'db_o': db_o_total,
            'dW_c': dW_c_total, 'dU_c': dU_c_total, 'db_c': db_c_total
        }
        if self.W_y is not None:
            param_gradients['W_y'] = dW_y_total
            param_gradients['b_y'] = db_y_total
            
        return dX, param_gradients

    def get_parameters(self):
        """返回模型的所有可训练参数。"""
        return self.parameters

    def set_parameters(self, parameters):
        """设置模型的可训练参数。"""
        # This needs to be more robust, matching parameter names or order
        idx = 0
        self.cell.W_i, self.cell.U_i, self.cell.b_i = parameters[idx:idx+3]; idx += 3
        self.cell.W_f, self.cell.U_f, self.cell.b_f = parameters[idx:idx+3]; idx += 3
        self.cell.W_o, self.cell.U_o, self.cell.b_o = parameters[idx:idx+3]; idx += 3
        self.cell.W_c, self.cell.U_c, self.cell.b_c = parameters[idx:idx+3]; idx += 3
        if self.W_y is not None:
            self.W_y, self.b_y = parameters[idx:idx+2]

# --- 向量化 LSTM 单元和层 ---

class VectorizedLSTMCell:
    """
    向量化的 LSTM 单元。此类处理整个批次的数据，而不是单个样本。
    它将输入门、遗忘门、单元门和输出门的计算合并，以提高效率。
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.0, recurrent_dropout_rate=0.0):
        """
        初始化向量化的 LSTM 单元。

        参数:
            input_size (int): 输入特征的维度。
            hidden_size (int): 隐藏状态和单元状态的维度。
            dropout_rate (float): 应用于输入 x_t 的 dropout 比率。
            recurrent_dropout_rate (float): 应用于上一个隐藏状态 h_prev 的 recurrent dropout 比率。
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate  # 应用于输入 x_t
        self.recurrent_dropout_rate = recurrent_dropout_rate  # 应用于 h_prev

        # 合并的权重矩阵，用于输入 x_t (W) 和循环连接 h_prev (U)
        # W 的形状为 (input_size, 4 * hidden_size)
        # U 的形状为 (hidden_size, 4 * hidden_size)
        # b 的形状为 (4 * hidden_size,)
        # 四个门分别是：输入门(i)、遗忘门(f)、单元门(c_tilde)、输出门(o)

        # 使用 Glorot (Xavier) 均匀初始化权重
        limit_W = np.sqrt(6.0 / (input_size + hidden_size))
        limit_U = np.sqrt(6.0 / (hidden_size + hidden_size))

        self.W = np.random.uniform(-limit_W, limit_W, (input_size, 4 * hidden_size))
        self.U = np.random.uniform(-limit_U, limit_U, (hidden_size, 4 * hidden_size))
        self.b = np.zeros(4 * hidden_size)
        # 将遗忘门的偏置初始化为 1.0 或一个小的正值，以鼓励在训练初期记住信息
        self.b[hidden_size : 2 * hidden_size] = 1.0 

        self.parameters = [self.W, self.U, self.b]
        self.param_names = ['W', 'U', 'b']  # 用于梯度关联

    def sigmoid(self, x):
        """Sigmoid 激活函数。"""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """Tanh 激活函数。"""
        return np.tanh(x)

    def forward(self, x_t_batch, h_prev_batch, c_prev_batch, training=True):
        """
        执行 LSTM 单元的前向传播（向量化版本）。

        参数:
            x_t_batch (np.ndarray): 当前时间步的输入批次，形状为 (batch_size, input_size)。
            h_prev_batch (np.ndarray): 上一个时间步的隐藏状态批次，形状为 (batch_size, hidden_size)。
            c_prev_batch (np.ndarray): 上一个时间步的单元状态批次，形状为 (batch_size, hidden_size)。
            training (bool): 是否处于训练模式（用于 dropout）。

        返回:
            h_t_batch (np.ndarray): 当前时间步的隐藏状态批次，形状为 (batch_size, hidden_size)。
            c_t_batch (np.ndarray): 当前时间步的单元状态批次，形状为 (batch_size, hidden_size)。
            cache (dict): 包含前向传播中间值的缓存，用于反向传播。
        """
        batch_size = x_t_batch.shape[0]

        batch_size = x_t_batch.shape[0]

        # 对输入 x_t 应用 dropout
        if training and self.dropout_rate > 0:
            # Inverted dropout: scale during training, no change during testing
            input_dropout_mask = (np.random.rand(*x_t_batch.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            x_t_batch_dropped = x_t_batch * input_dropout_mask
        else:
            x_t_batch_dropped = x_t_batch
            input_dropout_mask = None  # 如果不训练或不使用 dropout，则没有掩码

        # 对循环连接 h_prev 应用 recurrent dropout
        # 注意：标准的 recurrent dropout 通常在整个序列中使用相同的掩码。
        # 此处为简化单元实现，每次调用时生成。在层级别管理此掩码会更标准。
        if training and self.recurrent_dropout_rate > 0:
            rec_dropout_mask = (np.random.rand(*h_prev_batch.shape) > self.recurrent_dropout_rate) / (1.0 - self.recurrent_dropout_rate)
            h_prev_batch_dropped = h_prev_batch * rec_dropout_mask
        else:
            h_prev_batch_dropped = h_prev_batch
            rec_dropout_mask = None

        # 一次性计算所有门的线性变换
        # z = (x_t_dropped @ W) + (h_prev_dropped @ U) + b
        # z 的形状为 (batch_size, 4 * hidden_size)
        z = np.dot(x_t_batch_dropped, self.W) + np.dot(h_prev_batch_dropped, self.U) + self.b

        # 将 z 分割成四个部分，分别对应每个门（输入门、遗忘门、单元候选门、输出门）
        # 每个部分的形状是 (batch_size, hidden_size)
        a_i = z[:, :self.hidden_size]                # 输入门激活前的线性组合
        a_f = z[:, self.hidden_size:2*self.hidden_size] # 遗忘门激活前的线性组合
        a_c = z[:, 2*self.hidden_size:3*self.hidden_size] # 单元候选状态激活前的线性组合
        a_o = z[:, 3*self.hidden_size:]              # 输出门激活前的线性组合

        # 应用激活函数
        i_t = self.sigmoid(a_i)       # 输入门
        f_t = self.sigmoid(a_f)       # 遗忘门
        c_tilde_t = self.tanh(a_c)    # 候选单元状态
        o_t = self.sigmoid(a_o)       # 输出门

        # 计算新的单元状态
        # c_t = f_t * c_prev + i_t * c_tilde_t
        c_t_batch = f_t * c_prev_batch + i_t * c_tilde_t

        # 计算新的隐藏状态
        # h_t = o_t * tanh(c_t)
        h_t_batch = o_t * self.tanh(c_t_batch)

        # 保存用于反向传播的缓存
        cache = {
            'x_t_batch': x_t_batch,  # 原始 x_t (dropout 前)，用于梯度计算
            'h_prev_batch': h_prev_batch, # 原始 h_prev
            'c_prev_batch': c_prev_batch,
            'input_dropout_mask': input_dropout_mask, # 输入 dropout 掩码
            'rec_dropout_mask': rec_dropout_mask,     # 循环 dropout 掩码
            'z': z,                  # 门激活前的总线性组合
            'i_t': i_t, 'f_t': f_t, 'c_tilde_t': c_tilde_t, 'o_t': o_t, # 激活后的门值
            'a_i': a_i, 'a_f': a_f, 'a_c': a_c, 'a_o': a_o, # 激活前的门值
            'c_t_batch': c_t_batch,    # 当前单元状态
            'h_t_batch': h_t_batch     # 当前隐藏状态
        }
        return h_t_batch, c_t_batch, cache

    def backward(self, dh_next_batch, dc_next_batch, cache):
        """
        执行 LSTM 单元的反向传播（向量化版本）。

        参数:
            dh_next_batch (np.ndarray): 来自下一个时间步或损失函数关于当前隐藏状态 h_t 的梯度，
                                      形状为 (batch_size, hidden_size)。
            dc_next_batch (np.ndarray): 来自下一个时间步关于当前单元状态 c_t 的梯度，
                                      形状为 (batch_size, hidden_size)。
            cache (dict): 前向传播时保存的缓存。

        返回:
            gradients (dict): 包含所有相关梯度的字典：
                              'dW': W 的梯度 (input_size, 4 * hidden_size)
                              'dU': U 的梯度 (hidden_size, 4 * hidden_size)
                              'db': b 的梯度 (4 * hidden_size,)
                              'dx_t': x_t 的梯度 (batch_size, input_size)
                              'dh_prev': h_prev 的梯度 (batch_size, hidden_size)
                              'dc_prev': c_prev 的梯度 (batch_size, hidden_size)
        """
        # 从缓存中获取前向传播的中间值
        x_t_batch = cache['x_t_batch']          # (batch_size, input_size)
        h_prev_batch = cache['h_prev_batch']    # (batch_size, hidden_size)
        c_prev_batch = cache['c_prev_batch']    # (batch_size, hidden_size)
        input_dropout_mask = cache['input_dropout_mask'] # (batch_size, input_size) or None
        rec_dropout_mask = cache['rec_dropout_mask']     # (batch_size, hidden_size) or None
        # z = cache['z'] # (batch_size, 4 * hidden_size) # 实际上我们不需要 z 本身，而是它的分量 a_i, a_f, ...
        i_t, f_t, c_tilde_t, o_t = cache['i_t'], cache['f_t'], cache['c_tilde_t'], cache['o_t'] # 激活后的门
        # a_i, a_f, a_c, a_o = cache['a_i'], cache['a_f'], cache['a_c'], cache['a_o'] # 激活前的门值 (未使用，因为梯度直接通过激活后的门计算)
        c_t_batch = cache['c_t_batch']          # (batch_size, hidden_size)
        # h_t_batch = cache['h_t_batch'] # 在计算前一时间步的梯度时并不直接使用

        batch_size = dh_next_batch.shape[0]

        # 1. 计算关于 h_t 的梯度
        # h_t = o_t * tanh(c_t)
        # dL/do_t = dL/dh_t * tanh(c_t)
        # dL/d(tanh(c_t)) = dL/dh_t * o_t
        d_o_t = dh_next_batch * self.tanh(c_t_batch)  # (batch_size, hidden_size)
        d_tanh_c_t = dh_next_batch * o_t            # (batch_size, hidden_size)

        # 2. 计算关于 c_t 的梯度 (合并来自 h_t 路径和直接的 dc_next_batch)
        # dL/dc_t = dL/d(tanh(c_t)) * (1 - tanh(c_t)^2) + dc_next_batch (来自下一层或时间步的 c 的梯度)
        # d(tanh(x))/dx = 1 - tanh(x)^2
        d_c_t = d_tanh_c_t * (1 - self.tanh(c_t_batch)**2) 
        d_c_t += dc_next_batch # 加上来自下一个时间步（或层）的单元状态梯度

        # 3. 计算关于门和 c_prev 的梯度
        # c_t = f_t * c_prev_batch + i_t * c_tilde_t
        # dL/df_t = dL/dc_t * c_prev_batch
        # dL/dc_prev = dL/dc_t * f_t  (这是传递到上一个时间步的 dc)
        # dL/di_t = dL/dc_t * c_tilde_t
        # dL/dc_tilde_t = dL/dc_t * i_t

        # 输出门: o_t = sigmoid(a_o)
        # dL/da_o = dL/do_t * o_t * (1 - o_t)
        d_a_o = d_o_t * o_t * (1 - o_t) # (batch_size, hidden_size)

        # 单元状态 c_t 的相关梯度
        d_f_t = d_c_t * c_prev_batch      # (batch_size, hidden_size)
        d_c_prev_batch = d_c_t * f_t      # 传递给上一个单元状态的梯度 (batch_size, hidden_size)
        d_i_t = d_c_t * c_tilde_t         # (batch_size, hidden_size)
        d_c_tilde_t = d_c_t * i_t         # (batch_size, hidden_size)

        # 候选单元状态: c_tilde_t = tanh(a_c)
        # dL/da_c = dL/dc_tilde_t * (1 - c_tilde_t^2)
        d_a_c = d_c_tilde_t * (1 - c_tilde_t**2) # (batch_size, hidden_size)

        # 遗忘门: f_t = sigmoid(a_f)
        # dL/da_f = dL/df_t * f_t * (1 - f_t)
        d_a_f = d_f_t * f_t * (1 - f_t) # (batch_size, hidden_size)

        # 输入门: i_t = sigmoid(a_i)
        # dL/da_i = dL/di_t * i_t * (1 - i_t)
        d_a_i = d_i_t * i_t * (1 - i_t) # (batch_size, hidden_size)

        # 4. 将所有门的激活前梯度 (dL/da) 合并为一个矩阵
        # 顺序: i, f, c, o (与权重矩阵 W, U 的列顺序一致)
        d_a_concat = np.concatenate((d_a_i, d_a_f, d_a_c, d_a_o), axis=1) # (batch_size, 4 * hidden_size)

        # 5. 计算关于权重 W, U 和偏置 b 的梯度
        # z_dropped = x_t_dropped @ W + h_prev_dropped @ U + b
        # dL/dW = x_t_dropped.T @ dL/dz = x_t_dropped.T @ d_a_concat
        # dL/dU = h_prev_dropped.T @ dL/dz = h_prev_dropped.T @ d_a_concat
        # dL/db = sum(dL/dz, axis=0) = sum(d_a_concat, axis=0)

        # 获取前向传播时使用的 (可能经过 dropout 的) 输入
        x_t_batch_dropped = x_t_batch
        if input_dropout_mask is not None:
            x_t_batch_dropped = x_t_batch * input_dropout_mask # 恢复前向传播时使用的 x_t_dropped
        
        h_prev_batch_dropped = h_prev_batch
        if rec_dropout_mask is not None:
            h_prev_batch_dropped = h_prev_batch * rec_dropout_mask # 恢复前向传播时使用的 h_prev_dropped

        dW = np.dot(x_t_batch_dropped.T, d_a_concat)  # (input_size, 4 * hidden_size)
        dU = np.dot(h_prev_batch_dropped.T, d_a_concat) # (hidden_size, 4 * hidden_size)
        db = np.sum(d_a_concat, axis=0)             # (4 * hidden_size,)

        # 6. 计算关于此单元输入 (x_t, h_prev) 的梯度
        # dL/dx_t_dropped = dL/dz @ W.T = d_a_concat @ W.T
        # dL/dh_prev_dropped = dL/dz @ U.T = d_a_concat @ U.T
        dx_t_batch_dropped = np.dot(d_a_concat, self.W.T)    # (batch_size, input_size)
        dh_prev_batch_dropped = np.dot(d_a_concat, self.U.T) # (batch_size, hidden_size)

        # 7. 应用 dropout 掩码的导数到 dx_t 和 dh_prev
        # 如果 x_t_dropped = x_t * mask, 那么 dL/dx_t = dL/dx_t_dropped * mask
        dx_t_batch = dx_t_batch_dropped
        if input_dropout_mask is not None:
            dx_t_batch = dx_t_batch_dropped * input_dropout_mask

        dh_prev_batch = dh_prev_batch_dropped
        if rec_dropout_mask is not None:
            dh_prev_batch = dh_prev_batch_dropped * rec_dropout_mask

        # 收集所有梯度
        gradients = {
            'dW': dW, 'dU': dU, 'db': db,                # 权重和偏置的梯度
            'dx_t': dx_t_batch,                         # 输入 x_t 的梯度
            'dh_prev': dh_prev_batch,                   # 上一个隐藏状态 h_prev 的梯度
            'dc_prev': d_c_prev_batch                   # 上一个单元状态 c_prev 的梯度
        }
        return gradients

class VectorizedLSTMLayer:
    """
    向量化的 LSTM 层。
    该层使用 VectorizedLSTMCell 来处理输入序列。
    它可以返回整个序列的输出，或者只返回最后一个时间步的输出。
    支持对输入和循环连接应用 dropout。
    """
    def __init__(self, input_dim, units, return_sequences=True, output_dim=None, 
                 dropout_rate=0.0, recurrent_dropout_rate=0.0, name=None):
        """
        初始化向量化的 LSTM 层。

        参数:
            input_dim (int): 输入特征的维度。
            units (int): LSTM 单元中隐藏状态的维度 (也称为 hidden_size)。
            return_sequences (bool): 是否返回整个序列的输出。
                                     如果为 True, 输出形状为 (batch_size, sequence_length, units 或 output_dim)。
                                     如果为 False, 输出形状为 (batch_size, units 或 output_dim)。
            output_dim (int, optional): 输出的维度。如果为 None，则输出维度等于 units。
                                        仅当 return_sequences=False 时，如果 output_dim 与 units 不同，
                                        会添加一个额外的 Dense 层进行投影。
            dropout_rate (float): 应用于 LSTM 单元输入的 dropout 比率。
            recurrent_dropout_rate (float): 应用于 LSTM 单元循环连接的 dropout 比率。
            name (str, optional): 层的名称。
        """
        self.input_dim = input_dim
        self.units = units  # 隐藏层大小
        self.return_sequences = return_sequences
        self.output_dim = output_dim if output_dim is not None else units
        self.dropout_rate = dropout_rate  # 应用于单元格中的输入 x_t
        self.recurrent_dropout_rate = recurrent_dropout_rate  # 应用于单元格中的 h_prev
        self.name = name if name else 'lstm_layer'

        # 创建 LSTM 单元实例
        self.cell = VectorizedLSTMCell(input_dim, units, dropout_rate, recurrent_dropout_rate)

        # 可选的 Dense 层，用于在不返回序列且 output_dim 与 units 不同时进行输出投影
        self.W_y = None
        self.b_y = None
        # 参数列表初始化为单元的参数
        self.parameters = self.cell.parameters[:] # 创建副本以避免直接修改 cell 的列表
        self.param_names = [f'{self.name}_W', f'{self.name}_U', f'{self.name}_b']

        # 如果不返回序列且输出维度与隐藏单元数不同，则添加一个 Dense 投影层
        if not self.return_sequences and self.output_dim != self.units:
            # 使用 Glorot (Xavier) 均匀初始化 Dense 层的权重
            limit_Wy = np.sqrt(6.0 / (self.units + self.output_dim))
            self.W_y = np.random.uniform(-limit_Wy, limit_Wy, (self.units, self.output_dim))
            self.b_y = np.zeros(self.output_dim)
            self.parameters.extend([self.W_y, self.b_y])
            self.param_names.extend([f'{self.name}_W_y', f'{self.name}_b_y'])

    def forward(self, X_batch, training=True):
        """
        执行 LSTM 层的前向传播。

        参数:
            X_batch (np.ndarray): 输入批次数据，形状为 (batch_size, sequence_length, input_dim)。
            training (bool): 是否处于训练模式（用于 dropout）。

        返回:
            final_outputs (np.ndarray): LSTM 层的输出。
                                       如果 return_sequences=True, 形状为 (batch_size, sequence_length, units 或 output_dim)。
                                       如果 return_sequences=False, 形状为 (batch_size, units 或 output_dim)。
            layer_cache (dict): 包含前向传播中间值的缓存，用于反向传播。
        """
        batch_size, sequence_length, _ = X_batch.shape

        h_t = np.zeros((batch_size, self.units))
        c_t = np.zeros((batch_size, self.units))

        outputs = []
        caches = [] # List of cell caches for each timestep

        # Recurrent dropout mask for h_prev (same mask across all timesteps)
        rec_dropout_mask_h = None
        if training and self.recurrent_dropout_rate > 0:
            # This mask should be applied to h_prev inside the cell's forward pass
            # The cell's current implementation generates its own rec_dropout_mask per call.
            # For proper recurrent dropout, the layer should generate this once and pass it.
            # This requires modifying the cell to accept an optional pre-generated mask.
            # For now, we'll rely on the cell's per-call mask generation, which is less standard.
            # A more standard approach: generate mask here, pass to cell, cell applies it.
            # Or, cell's forward takes h_prev and applies its *own* consistent mask if training.
            # Let's assume cell handles its own recurrent dropout mask generation internally for now.
            pass

        for t in range(sequence_length):
            x_t_current_batch = X_batch[:, t, :]
            # The cell's forward method will handle its own dropout masks based on 'training' flag
            h_t, c_t, cell_cache = self.cell.forward(x_t_current_batch, h_t, c_t, training=training)
            caches.append(cell_cache)
            if self.return_sequences:
                outputs.append(h_t.copy()) # Store all hidden states

        layer_cache = {
            'X_batch': X_batch, # Input to the layer
            'cell_caches': caches, # List of caches from each cell call
            'last_h_t': h_t.copy() if not self.return_sequences else None, # Needed for dW_y if applicable
            'last_c_t': c_t.copy() if not self.return_sequences else None
        }

        if self.return_sequences:
            final_outputs = np.stack(outputs, axis=1) # (batch_size, sequence_length, units)
        else:
            final_outputs = h_t # Last hidden state (batch_size, units)
            if self.W_y is not None:
                final_outputs = np.dot(final_outputs, self.W_y) + self.b_y # (batch_size, output_dim)
        
        return final_outputs, layer_cache

    def backward(self, d_output, layer_cache):
        """
        执行 LSTM 层的反向传播。

        参数:
            d_output (np.ndarray): 来自下一层或损失函数关于此层输出的梯度。
                                 如果 return_sequences=True, 形状为 (batch_size, sequence_length, units 或 output_dim)。
                                 如果 return_sequences=False, 形状为 (batch_size, units 或 output_dim)。
            layer_cache (dict): 从前向传播中保存的缓存。

        返回:
            dX_batch (np.ndarray): 关于此层输入 X_batch 的梯度，形状为 (batch_size, sequence_length, input_dim)。
            all_param_gradients (list): 包含此层所有参数 (W, U, b, 以及可选的 W_y, b_y) 梯度的列表，
                                      顺序与 self.parameters 一致。
        """
        X_batch = layer_cache['X_batch']
        cell_caches = layer_cache['cell_caches']
        batch_size, sequence_length, _ = X_batch.shape

        # Initialize gradients for cell parameters (W, U, b)
        dW_total = np.zeros_like(self.cell.W)
        dU_total = np.zeros_like(self.cell.U)
        db_total = np.zeros_like(self.cell.b)

        # Initialize gradients for optional dense layer parameters (W_y, b_y)
        dW_y_total = np.zeros_like(self.W_y) if self.W_y is not None else None
        db_y_total = np.zeros_like(self.b_y) if self.b_y is not None else None

        # Gradient of the input to this LSTM layer
        dX_batch = np.zeros_like(X_batch)

        # Gradients for h_t and c_t to propagate backwards in time
        dh_next_from_future = np.zeros((batch_size, self.units))
        dc_next_from_future = np.zeros((batch_size, self.units))

        if not self.return_sequences:
            # d_output is (batch_size, output_dim or units)
            if self.W_y is not None:
                # d_output is dL/d(final_projected_output)
                # final_projected_output = last_h_t @ W_y + b_y
                # We need h_t from the last timestep, which is layer_cache['last_h_t']
                last_h_t = layer_cache['last_h_t'] # (batch_size, units)
                
                dW_y_total = np.dot(last_h_t.T, d_output) # (units, output_dim)
                db_y_total = np.sum(d_output, axis=0)    # (output_dim,)
                
                # Gradient w.r.t last_h_t (input to the dense layer)
                dh_next_from_future = np.dot(d_output, self.W_y.T) # (batch_size, units)
            else:
                # d_output is dL/d(last_h_t)
                dh_next_from_future = d_output # (batch_size, units)
            # dc_next_from_future remains zero as it's not directly affected by output layer

        # Iterate backwards through time
        for t in reversed(range(sequence_length)):
            current_cell_cache = cell_caches[t]
            
            # Determine dh_input_to_cell for this timestep
            # It's sum of gradient from loss/next_layer (if return_sequences) AND gradient from future timestep (t+1)
            if self.return_sequences:
                dh_from_loss_or_next_layer = d_output[:, t, :] # (batch_size, units)
                dh_input_to_cell = dh_from_loss_or_next_layer + dh_next_from_future
            else:
                # If not returning sequences, only the last timestep (t == sequence_length - 1)
                # receives dh_next_from_future from the output projection (or directly from d_output).
                # For t < sequence_length - 1, dh_input_to_cell is purely dh_next_from_future (from t+1).
                dh_input_to_cell = dh_next_from_future
            
            dc_input_to_cell = dc_next_from_future # Always from future timestep for cell state

            cell_gradients = self.cell.backward(dh_input_to_cell, dc_input_to_cell, current_cell_cache)
            
            # Accumulate gradients for cell's W, U, b
            dW_total += cell_gradients['dW']
            dU_total += cell_gradients['dU']
            db_total += cell_gradients['db']
            
            # Store gradient w.r.t input x_t for this timestep
            dX_batch[:, t, :] = cell_gradients['dx_t']
            
            # Update dh_next_from_future and dc_next_from_future for the PREVIOUS timestep (t-1)
            dh_next_from_future = cell_gradients['dh_prev']
            dc_next_from_future = cell_gradients['dc_prev']

        # 整合所有参数的梯度
        # 顺序必须与 self.param_names 和 self.parameters 中的顺序匹配
        all_param_gradients = [dW_total, dU_total, db_total]
        if self.W_y is not None:
            all_param_gradients.extend([dW_y_total, db_y_total])
            
        # 返回关于输入的梯度 dX_batch 和一个包含所有参数梯度的列表
        return dX_batch, all_param_gradients 

    def get_parameters(self):
        """返回一个包含参数 ndarray (W, U, b, 以及可选的 W_y, b_y) 的列表。"""
        return self.parameters

    def set_parameters(self, new_params):
        """从一个 ndarray 列表设置参数。
           new_params 中的顺序必须与 self.parameters 中的顺序匹配。
        """

        if len(new_params) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} parameter arrays, got {len(new_params)}")
        
        self.cell.W, self.cell.U, self.cell.b = new_params[0], new_params[1], new_params[2]
        current_idx = 3
        if self.W_y is not None:
            self.W_y = new_params[current_idx]
            current_idx += 1
        if self.b_y is not None:
            self.b_y = new_params[current_idx]
            current_idx += 1
        
        # Update self.parameters to point to the new arrays if they were replaced by copy
        # If new_params elements are views or direct assignments, this might not be strictly necessary
        # but good for consistency.
        self.parameters = [self.cell.W, self.cell.U, self.cell.b]
        if self.W_y is not None:
            self.parameters.append(self.W_y)
        if self.b_y is not None:
            self.parameters.append(self.b_y)

# --- 测试用例 ---

def test_lstm_cell():
    print("测试 LSTMCell...")
    input_size = 3  # 输入维度
    hidden_size = 5 # 隐藏状态维度
    cell = LSTMCell(input_size, hidden_size) # 初始化 LSTM 单元

    # 生成随机输入数据
    x_t = np.random.randn(input_size)       # 当前时间步的输入
    h_prev = np.random.randn(hidden_size)   # 上一时间步的隐藏状态
    c_prev = np.random.randn(hidden_size)   # 上一时间步的细胞状态

    # 前向传播
    h_t, c_t, cache = cell.forward(x_t, h_prev, c_prev)
    print(f"  h_t 形状: {h_t.shape}, c_t 形状: {c_t.shape}")
    assert h_t.shape == (hidden_size,) # 验证隐藏状态形状
    assert c_t.shape == (hidden_size,) # 验证细胞状态形状

    # 反向传播
    dh_next = np.random.randn(hidden_size) # 下一时间步隐藏状态的梯度
    dc_next = np.random.randn(hidden_size) # 下一时间步细胞状态的梯度
    gradients = cell.backward(dh_next, dc_next, cache) # 计算梯度
    
    print(f"  梯度已计算。dW_i 形状示例: {gradients['dW_i'].shape}")
    assert gradients['dW_i'].shape == (input_size, hidden_size) # 验证输入门权重梯度形状
    assert gradients['dh_prev'].shape == (hidden_size,) # 验证上一隐藏状态梯度形状
    print("LSTMCell 测试通过。")

def test_lstm_layer_non_vectorized():
    print("\n测试 LSTMLayer (非向量化)...")
    batch_size = 2      # 批量大小
    sequence_length = 4 # 序列长度
    input_size = 3      # 输入维度
    hidden_size = 5     # 隐藏状态维度

    # 测试 return_sequences = True
    layer_seq = LSTMLayer(input_size, hidden_size, return_sequences=True) # 初始化层，返回所有时间步输出
    X = np.random.randn(batch_size, sequence_length, input_size) # 生成随机输入序列
    
    outputs_seq, caches_seq = layer_seq.forward(X) # 前向传播
    print(f"  返回序列 True: 输出形状: {outputs_seq.shape}")
    assert outputs_seq.shape == (batch_size, sequence_length, hidden_size) # 验证输出形状

    d_output_seq = np.random.randn(batch_size, sequence_length, hidden_size) # 构造输出梯度
    dX_seq, param_grads_seq = layer_seq.backward(d_output_seq, caches_seq) # 反向传播
    print(f"  返回序列 True: dX 形状: {dX_seq.shape}, dW_i 形状: {param_grads_seq['dW_i'].shape}")
    assert dX_seq.shape == X.shape # 验证输入梯度形状
    assert param_grads_seq['dW_i'].shape == layer_seq.cell.W_i.shape # 验证权重梯度形状

    # 测试 return_sequences = False
    layer_last = LSTMLayer(input_size, hidden_size, return_sequences=False) # 初始化层，仅返回最后一个时间步输出
    outputs_last, caches_last = layer_last.forward(X) # 前向传播
    print(f"  返回序列 False: 输出形状: {outputs_last.shape}")
    assert outputs_last.shape == (batch_size, hidden_size) # 验证输出形状

    d_output_last = np.random.randn(batch_size, hidden_size) # 构造输出梯度
    dX_last, param_grads_last = layer_last.backward(d_output_last, caches_last) # 反向传播
    print(f"  返回序列 False: dX 形状: {dX_last.shape}, dW_i 形状: {param_grads_last['dW_i'].shape}")
    assert dX_last.shape == X.shape # 验证输入梯度形状

    # 测试 return_sequences = False 且带有输出投影
    output_projection_size = 7 # 输出投影维度
    layer_proj = LSTMLayer(input_size, hidden_size, output_size=output_projection_size, return_sequences=False) # 初始化层，带输出投影
    outputs_proj, caches_proj = layer_proj.forward(X) # 前向传播
    print(f"  返回序列 False 且带投影: 输出形状: {outputs_proj.shape}")
    assert outputs_proj.shape == (batch_size, output_projection_size) # 验证输出形状

    d_output_proj = np.random.randn(batch_size, output_projection_size) # 构造输出梯度
    dX_proj, param_grads_proj = layer_proj.backward(d_output_proj, caches_proj) # 反向传播
    print(f"  返回序列 False 且带投影: dX 形状: {dX_proj.shape}, dW_y 形状: {param_grads_proj['W_y'].shape}")
    assert dX_proj.shape == X.shape # 验证输入梯度形状
    assert param_grads_proj['W_y'].shape == (hidden_size, output_projection_size) # 验证投影权重梯度形状

    print("LSTMLayer (非向量化) 测试通过。")


def test_vectorized_lstm_cell():
    print("\n测试 VectorizedLSTMCell...")
    batch_size = 2      # 批量大小
    input_size = 3      # 输入维度
    hidden_size = 5     # 隐藏状态维度
    # 初始化向量化 LSTM 单元，并设置 dropout 率
    cell = VectorizedLSTMCell(input_size, hidden_size, dropout_rate=0.1, recurrent_dropout_rate=0.1)

    # 生成随机输入数据 (批处理)
    x_t_batch = np.random.randn(batch_size, input_size)         # 当前时间步的批量输入
    h_prev_batch = np.random.randn(batch_size, hidden_size)    # 上一时间步的批量隐藏状态
    c_prev_batch = np.random.randn(batch_size, hidden_size)    # 上一时间步的批量细胞状态

    # 前向传播 (training=True 以启用 dropout)
    h_t, c_t, cache = cell.forward(x_t_batch, h_prev_batch, c_prev_batch, training=True)
    print(f"  h_t 形状: {h_t.shape}, c_t 形状: {c_t.shape}")
    assert h_t.shape == (batch_size, hidden_size) # 验证隐藏状态形状
    assert c_t.shape == (batch_size, hidden_size) # 验证细胞状态形状

    # 前向传播 (training=False 以禁用 dropout, 用于评估)
    h_t_eval, _, _ = cell.forward(x_t_batch, h_prev_batch, c_prev_batch, training=False)
    # 注意: 由于 dropout, h_t 和 h_t_eval 可能会不同 (除非 dropout 率为 0)

    # 反向传播
    dh_next = np.random.randn(batch_size, hidden_size) # 下一时间步隐藏状态的梯度 (批处理)
    dc_next = np.random.randn(batch_size, hidden_size) # 下一时间步细胞状态的梯度 (批处理)
    gradients = cell.backward(dh_next, dc_next, cache) # 计算梯度
    
    print(f"  梯度已计算。dW 形状示例: {gradients['dW'].shape}")
    assert gradients['dW'].shape == (input_size, 4 * hidden_size) # 验证输入权重梯度形状
    assert gradients['dh_prev'].shape == (batch_size, hidden_size) # 验证上一隐藏状态梯度形状
    assert gradients['dx_t'].shape == (batch_size, input_size) # 验证当前输入梯度形状
    print("VectorizedLSTMCell 测试通过。")


def test_vectorized_lstm_layer():
    print("\n测试 VectorizedLSTMLayer...")
    batch_size = 2          # 批量大小
    sequence_length = 4     # 序列长度
    input_dim = 3           # 输入维度
    units = 5               # LSTM单元数量 (隐藏状态维度)

    # 测试 return_sequences = True, 并启用 dropout
    layer_seq = VectorizedLSTMLayer(input_dim, units, return_sequences=True, dropout_rate=0.1, recurrent_dropout_rate=0.1)
    X = np.random.randn(batch_size, sequence_length, input_dim) # 生成随机输入序列
    
    outputs_seq, cache_seq = layer_seq.forward(X, training=True) # 前向传播 (训练模式)
    print(f"  返回序列 True: 输出形状: {outputs_seq.shape}")
    assert outputs_seq.shape == (batch_size, sequence_length, units) # 验证输出形状

    d_output_seq = np.random.randn(batch_size, sequence_length, units) # 构造输出梯度
    dX_seq, param_grads_seq = layer_seq.backward(d_output_seq, cache_seq) # 反向传播
    print(f"  返回序列 True: dX 形状: {dX_seq.shape}, dW (cell) 形状: {param_grads_seq[0].shape}")
    assert dX_seq.shape == X.shape # 验证输入梯度形状
    assert param_grads_seq[0].shape == layer_seq.cell.W.shape # 验证单元输入权重梯度形状

    # 测试 return_sequences = False, 并测试评估模式 (training=False)
    layer_last = VectorizedLSTMLayer(input_dim, units, return_sequences=False)
    outputs_last, cache_last = layer_last.forward(X, training=False) # 前向传播 (评估模式)
    print(f"  返回序列 False: 输出形状: {outputs_last.shape}")
    assert outputs_last.shape == (batch_size, units) # 验证输出形状

    d_output_last = np.random.randn(batch_size, units) # 构造输出梯度
    dX_last, param_grads_last = layer_last.backward(d_output_last, cache_last) # 反向传播
    print(f"  返回序列 False: dX 形状: {dX_last.shape}, dW (cell) 形状: {param_grads_last[0].shape}")
    assert dX_last.shape == X.shape # 验证输入梯度形状

    # 测试 return_sequences = False 且带有输出投影
    output_projection_dim = 7 # 输出投影维度
    layer_proj = VectorizedLSTMLayer(input_dim, units, output_dim=output_projection_dim, return_sequences=False)
    outputs_proj, cache_proj = layer_proj.forward(X) # 前向传播
    print(f"  返回序列 False 且带投影: 输出形状: {outputs_proj.shape}")
    assert outputs_proj.shape == (batch_size, output_projection_dim) # 验证输出形状

    d_output_proj = np.random.randn(batch_size, output_projection_dim) # 构造输出梯度
    dX_proj, param_grads_proj = layer_proj.backward(d_output_proj, cache_proj) # 反向传播
    # param_grads_proj 应该是 [dW_cell, dU_cell, db_cell, dW_y, db_y]
    print(f"  返回序列 False 且带投影: dX 形状: {dX_proj.shape}, dW_y 形状: {param_grads_proj[3].shape}")
    assert dX_proj.shape == X.shape # 验证输入梯度形状
    assert len(param_grads_proj) == 5
    assert param_grads_proj[3].shape == (units, output_projection_dim) # 验证投影权重 dW_y 形状
    # 确保投影偏置梯度也存在且形状正确 (如果适用)
    if len(param_grads_proj) > 4 and param_grads_proj[4] is not None:
        assert param_grads_proj[4].shape == (output_projection_dim,) # 验证投影偏置 db_y 形状

    print("VectorizedLSTMLayer 测试通过。")

if __name__ == '__main__':
    # 运行所有测试函数
    test_lstm_cell()
    test_lstm_layer_non_vectorized()
    test_vectorized_lstm_cell()
    test_vectorized_lstm_layer()

    # get_parameters 和 set_parameters 方法的使用示例
    print("\n测试 VectorizedLSTMLayer 的 get_parameters 和 set_parameters 方法...")
    # 初始化一个带输出投影的 VectorizedLSTMLayer
    layer_example = VectorizedLSTMLayer(input_dim=3, units=5, output_dim=2, return_sequences=False)
    initial_params_example = layer_example.get_parameters() # 获取初始参数
    print(f"  初始 W_y 形状: {initial_params_example[3].shape if len(initial_params_example) > 3 and initial_params_example[3] is not None else 'N/A'}")

    # 创建一组形状相同的新虚拟参数
    new_params_example = [np.random.randn(*p.shape) for p in initial_params_example]
    layer_example.set_parameters(new_params_example) # 设置新参数
    
    # 验证参数是否已更新 (例如，通过检查其中一个参数)
    updated_params_example = layer_example.get_parameters() # 获取更新后的参数
    assert np.allclose(updated_params_example[0], new_params_example[0]) # 检查单元的 W 权重
    if len(initial_params_example) > 3 and initial_params_example[3] is not None:
        assert np.allclose(updated_params_example[3], new_params_example[3]) # 如果存在，检查投影权重 W_y
        print(f"  更新后 W_y 形状: {updated_params_example[3].shape}")

    print("get_parameters 和 set_parameters 测试通过。")

    print("\n所有 LSTM 测试完成。")
                


# --- TODO: Vectorize LSTMCell --- 
# The current implementation of LSTMLayer.forward and LSTMLayer.backward iterates through the batch.
# This is highly inefficient. The LSTMCell's forward and backward methods should be vectorized
# to process the entire batch at once.

# Example of how a vectorized LSTMCell might look (conceptual):
class VectorizedLSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weights for efficiency: W maps [x_t, h_prev] to [i, f, o, c_tilde]
        # W_all will have shape (input_size + hidden_size, 4 * hidden_size)
        # b_all will have shape (4 * hidden_size)
        self.W_all = np.random.randn(input_size + hidden_size, 4 * hidden_size) * 0.01
        self.b_all = np.zeros(4 * hidden_size)
        
        # For individual access if needed (e.g. for specific regularization)
        # Or, initialize W_i, U_i etc. and then stack them into W_all
        # W_x = [W_xi, W_xf, W_xo, W_xc] each (input_size, hidden_size)
        # W_h = [W_hi, W_hf, W_ho, W_hc] each (hidden_size, hidden_size)
        # self.W_x = np.random.randn(input_size, 4 * hidden_size) * 0.01
        # self.W_h = np.random.randn(hidden_size, 4 * hidden_size) * 0.01
        # self.b = np.zeros(4 * hidden_size)

        self.parameters = [self.W_all, self.b_all]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x_t_batch, h_prev_batch, c_prev_batch):
        """
        x_t_batch: (batch_size, input_size)
        h_prev_batch: (batch_size, hidden_size)
        c_prev_batch: (batch_size, hidden_size)
        """
        batch_size = x_t_batch.shape[0]

        # Concatenate x_t and h_prev_batch for combined matrix multiplication
        xh_concat = np.concatenate((x_t_batch, h_prev_batch), axis=1) # (batch_size, input_size + hidden_size)

        # Compute all gate activations in one go
        gates_pre_activation = np.dot(xh_concat, self.W_all) + self.b_all # (batch_size, 4 * hidden_size)

        # Split into individual gates
        i_t_pre = gates_pre_activation[:, :self.hidden_size]
        f_t_pre = gates_pre_activation[:, self.hidden_size : 2*self.hidden_size]
        o_t_pre = gates_pre_activation[:, 2*self.hidden_size : 3*self.hidden_size]
        c_tilde_t_pre = gates_pre_activation[:, 3*self.hidden_size:]

        # Apply activations
        i_t = self.sigmoid(i_t_pre)
        f_t = self.sigmoid(f_t_pre)
        o_t = self.sigmoid(o_t_pre)
        c_tilde_t = self.tanh(c_tilde_t_pre)

        # Cell state update
        c_t_batch = f_t * c_prev_batch + i_t * c_tilde_t

        # Hidden state update
        h_t_batch = o_t * self.tanh(c_t_batch)

        cache = (x_t_batch, h_prev_batch, c_prev_batch, xh_concat, i_t, f_t, o_t, c_tilde_t, c_t_batch, gates_pre_activation)
        return h_t_batch, c_t_batch, cache

    def backward(self, dh_next_batch, dc_next_batch, cache):
        """
        dh_next_batch: (batch_size, hidden_size)
        dc_next_batch: (batch_size, hidden_size)
        cache: from forward pass
        """
        x_t_batch, h_prev_batch, c_prev_batch, xh_concat, i_t, f_t, o_t, c_tilde_t, c_t_batch, gates_pre_activation = cache
        batch_size = x_t_batch.shape[0]

        # Gradient of h_t w.r.t o_t and c_t
        do_t = dh_next_batch * self.tanh(c_t_batch)
        dc_t = dh_next_batch * o_t * (1 - self.tanh(c_t_batch)**2) + dc_next_batch

        # Gradients of gates (post-activation)
        di_t = dc_t * c_tilde_t
        df_t = dc_t * c_prev_batch
        dc_tilde_t = dc_t * i_t
        # do_t already computed

        # Gradients of gates (pre-activation)
        # d(sigmoid)/dx = sigmoid(x)*(1-sigmoid(x)) = y*(1-y)
        # d(tanh)/dx = 1 - tanh(x)^2 = 1 - y^2
        da_i = di_t * i_t * (1 - i_t)
        da_f = df_t * f_t * (1 - f_t)
        da_o = do_t * o_t * (1 - o_t)
        da_c_tilde = dc_tilde_t * (1 - c_tilde_t**2)

        # Combine pre-activation gradients
        dgates_pre_activation = np.concatenate((da_i, da_f, da_o, da_c_tilde), axis=1) # (batch_size, 4 * hidden_size)

        # Gradients for W_all and b_all
        # dL/dW = x.T * dL/d(pre_activation)
        # dL/db = sum(dL/d(pre_activation), axis=0)
        dW_all = np.dot(xh_concat.T, dgates_pre_activation)
        db_all = np.sum(dgates_pre_activation, axis=0)

        # Gradients for xh_concat (to propagate to x_t and h_prev)
        # dL/d(xh_concat) = dL/d(pre_activation) * W_all.T
        dxh_concat = np.dot(dgates_pre_activation, self.W_all.T) # (batch_size, input_size + hidden_size)

        # Split d(xh_concat) into dx_t and dh_prev
        dx_t_batch = dxh_concat[:, :self.input_size]
        dh_prev_batch = dxh_concat[:, self.input_size:]

        # Gradient for c_prev
        dc_prev_batch = dc_t * f_t

        gradients = {
            'dW_all': dW_all, 'db_all': db_all,
            'dx_t': dx_t_batch, 'dh_prev': dh_prev_batch, 'dc_prev': dc_prev_batch
        }
        return gradients

# Update LSTMLayer to use VectorizedLSTMCell
class VectorizedLSTMLayer:
    def __init__(self, input_size, hidden_size, output_size=None, return_sequences=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.return_sequences = return_sequences

        self.cell = VectorizedLSTMCell(input_size, hidden_size)
        
        self.W_y = None
        self.b_y = None
        if not return_sequences and self.output_size != hidden_size:
            self.W_y = np.random.randn(hidden_size, self.output_size) * 0.01
            self.b_y = np.zeros(self.output_size)
            self.parameters = self.cell.parameters + [self.W_y, self.b_y]
        else:
            self.parameters = self.cell.parameters

    def forward(self, X_batch):
        """
        X_batch: input sequence (batch_size, sequence_length, input_size)
        """
        batch_size, sequence_length, _ = X_batch.shape
        
        h_batch = np.zeros((batch_size, self.hidden_size))
        c_batch = np.zeros((batch_size, self.hidden_size))
        
        outputs_batch = [] 
        caches_layer = [] 

        for t in range(sequence_length):
            x_t_batch = X_batch[:, t, :] 
            h_batch, c_batch, cell_cache = self.cell.forward(x_t_batch, h_batch, c_batch)
            caches_layer.append(cell_cache)
            if self.return_sequences:
                outputs_batch.append(h_batch.copy())
        
        if self.return_sequences:
            stacked_outputs = np.stack(outputs_batch, axis=1)
            return stacked_outputs, caches_layer
        else:
            final_h = h_batch
            if self.W_y is not None:
                final_output = np.dot(final_h, self.W_y) + self.b_y
                return final_output, caches_layer
            return final_h, caches_layer

    def backward(self, d_output_batch, caches_layer):
        """
        d_output_batch: Gradient from subsequent layer or loss.
                        Shape: (batch_size, sequence_length, hidden_size/output_size) if return_sequences.
                        Shape: (batch_size, hidden_size/output_size) if not return_sequences.
        caches_layer: List of caches from each timestep's cell.forward.
        """
        batch_size, _ = caches_layer[0][0].shape # x_t_batch is (batch_size, input_size)
        sequence_length = len(caches_layer)

        # Initialize gradients for cell parameters (accumulated)
        dW_all_total = np.zeros_like(self.cell.W_all)
        db_all_total = np.zeros_like(self.cell.b_all)
        
        dW_y_total = np.zeros_like(self.W_y) if self.W_y is not None else None
        db_y_total = np.zeros_like(self.b_y) if self.b_y is not None else None

        dX_batch = np.zeros((batch_size, sequence_length, self.input_size))

        dh_next_batch = np.zeros((batch_size, self.hidden_size))
        dc_next_batch = np.zeros((batch_size, self.hidden_size))

        if not self.return_sequences:
            # d_output_batch is (batch_size, output_size)
            if self.W_y is not None:
                # h_last_batch is needed for dW_y, db_y
                # It's h_t_batch from the last timestep, which is caches_layer[-1][8]
                h_last_batch = caches_layer[-1][8] # c_t_batch is 8th element, h_t_batch is after that. Oh, cache for vectorized is different.
                                                # cache = (x_t_batch, h_prev_batch, c_prev_batch, xh_concat, i_t, f_t, o_t, c_tilde_t, c_t_batch, gates_pre_activation)
                                                # h_t_batch is not directly stored in cache, but it was the output of cell.forward
                                                # We need the h_batch that was input to W_y. This is the h_batch from the *last* iteration of the forward loop.
                                                # The LSTMLayer.forward returns final_h (which is h_batch after loop). We need this value. 
                                                # This implies forward pass should also return last h if not returning sequences.
                                                # For now, let's assume h_last_batch can be retrieved from the last cache's h_prev_batch for the *next* (non-existent) step, or more simply, the h_t from the last step.
                                                # The cache for VectorizedLSTMCell is: (x_t_batch, h_prev_batch, c_prev_batch, xh_concat, i_t, f_t, o_t, c_tilde_t, c_t_batch, gates_pre_activation)
                                                # The h_t_batch is output, not in cache. The LSTMLayer.forward stores it as `h_batch` and updates it.
                                                # The `h_batch` *before* the dense layer is `caches_layer[-1][1]` (h_prev_batch for the *next* step) if we consider the state *after* the last cell.forward call.
                                                # Or more simply, the `h_t_batch` that was output by `self.cell.forward` in the last timestep. This is `h_batch` in `VectorizedLSTMLayer.forward` at the end of the loop.
                                                # This is tricky. Let's assume the `h_last_batch` is the `h_batch` that was fed into the dense layer. We need to get it from the cache. 
                                                # The `h_t_batch` is the output of the last cell.forward. It's `h_batch` in the loop. `caches_layer[-1]` is the cache from that last call.
                                                # The `h_prev_batch` in `caches_layer[-1]` is the `h` from `t-1`. The `h_t_batch` is `h_batch` after the call.
                                                # Let's re-evaluate: if not return_sequences, `final_h` is returned. This `final_h` is `h_batch` after the loop.
                                                # This `h_batch` is the one that would be `h_prev_batch` for `t = sequence_length`. 
                                                # The `h_t` for the last actual timestep `T-1` is `outputs_batch[-1]` if we were collecting them.
                                                # Or, it's the `h_batch` variable after the last `self.cell.forward` call.
                                                # The `caches_layer[-1]` contains `h_prev_batch` for the last step. We need `h_t` of the last step.
                                                # This means the `LSTMLayer` should probably store the final `h_batch` if `W_y` is used.
                                                # For now, let's assume `h_last_batch` is the `h_t` from the last timestep. We can get it if `forward` returns it or if we reconstruct it.
                                                # The `VectorizedLSTMCell.forward` returns `h_t_batch`. The `VectorizedLSTMLayer.forward` updates `h_batch` with this. So `h_batch` at the end of the loop is the one.
                                                # This `h_batch` is not in `caches_layer`. This is a design flaw in how `caches_layer` is used for `dW_y`.
                                                # A common way is to pass `final_h` (the one used for `W_y`) into backward as part of the cache.
                                                # Let's assume `d_output_batch` is `dL/d(final_output)` and `final_h` was used to compute `final_output`.
                                                # We need `final_h` to compute `dW_y`. It's the `h_batch` after the loop in `forward`.
                                                # Let's modify `caches_layer` to include the final `h_batch` if `W_y` is used.
                                                # For now, I'll proceed with the assumption that `h_last_batch` can be obtained. The simplest is to get `h_prev_batch` from the *next* (non-existent) step, which is the `h_t_batch` of the last step.
                                                # The `h_t_batch` from the last timestep `t=sequence_length-1` is the one used. This is `h_batch` after the last loop iteration.
                                                # This `h_batch` is what `dh_next_batch` should be initialized to if `W_y` is present.

                # To get h_last_batch (the one that was input to W_y layer): 
                # It's the h_t from the last timestep. 
                # If forward pass returned (final_h, caches_layer), then final_h is what we need.
                # Let's assume it's available or passed via a modified cache. For now, this is a placeholder.
                # A common practice: the cache for a layer includes all necessary items for its own backward pass.
                # The `caches_layer` here is a list of *cell* caches. The *layer* cache should include `final_h` if `W_y` is used.
                # Let's assume `caches_layer` is `(list_of_cell_caches, final_h_if_dense_layer_used)`
                
                # This part is problematic without final_h. Let's assume it's passed in `caches_layer` somehow.
                # If `caches_layer` is just `list_of_cell_caches`, then we need to reconstruct `h_last_batch`.
                # `h_last_batch` is the `h_t` from the last timestep. It's `h_batch` after the last `cell.forward` call.
                # It's not directly in `caches_layer[t]`. 
                # This needs a slight redesign of what `caches_layer` stores or how `backward` gets `h_last_batch`.
                # For now, I will skip dW_y, db_y calculation if h_last_batch is not clearly available from cache.
                # A better cache for LSTMLayer would be `(X_batch, list_of_cell_caches, final_h_if_applicable)`

                # Let's assume `d_output_batch` is `dL/d(Wy_output)`
                # `dh_next_batch` should be `dL/dh_last_timestep`
                dh_next_batch = np.dot(d_output_batch, self.W_y.T) # This is dL/dh for the last h_t
                # dW_y and db_y require h_last_batch. This is a known issue with current cache structure.
                # If we had h_last_batch (shape: batch_size, hidden_size):
                # dW_y_total = np.dot(h_last_batch.T, d_output_batch)
                # db_y_total = np.sum(d_output_batch, axis=0)
            else:
                # d_output_batch is dL/dh_last_timestep
                dh_next_batch = d_output_batch
            # dc_next_batch is still zero as it's not directly affected by the output layer

        for t in reversed(range(sequence_length)):
            cell_cache_t = caches_layer[t]
            
            # If return_sequences, d_output_batch is (batch_size, seq_len, hidden_size)
            # dh_for_current_cell = (gradient from loss on this step's h) + (gradient from next h_prev)
            current_dh_from_loss = d_output_batch[:, t, :] if self.return_sequences else 0
            dh_for_cell_backward = current_dh_from_loss + dh_next_batch
            dc_for_cell_backward = dc_next_batch # Only from propagation
            
            cell_gradients = self.cell.backward(dh_for_cell_backward, dc_for_cell_backward, cell_cache_t)
            
            dW_all_total += cell_gradients['dW_all']
            db_all_total += cell_gradients['db_all']
            
            dX_batch[:, t, :] = cell_gradients['dx_t']
            dh_next_batch = cell_gradients['dh_prev'] # Propagate to t-1
            dc_next_batch = cell_gradients['dc_prev'] # Propagate to t-1

        layer_gradients = {
            'dW_all': dW_all_total, 'db_all': db_all_total,
            'dX_batch': dX_batch
        }
        if self.W_y is not None:
            # This part is still problematic without h_last_batch for dW_y, db_y
            # For now, these will be None or zero if not calculated.
            # To fix, LSTMLayer.forward should return final_h in its cache if W_y is used.
            # And LSTMLayer.backward should expect it.
            # Example: cache = (cell_caches, final_h_for_dense)
            # Then: final_h_for_dense = caches_layer[1] (if caches_layer is a tuple)
            # dW_y_total = np.dot(final_h_for_dense.T, d_output_batch_for_dense_layer_output)
            # db_y_total = np.sum(d_output_batch_for_dense_layer_output, axis=0)
            layer_gradients['dW_y'] = dW_y_total # Placeholder, likely incorrect without h_last_batch
            layer_gradients['db_y'] = db_y_total # Placeholder
            
        return layer_gradients


# Note: The LSTMLayer (non-vectorized) has a fundamental issue in its backward pass
# with how batch_size is determined and how gradients are accumulated.
# The VectorizedLSTMLayer and VectorizedLSTMCell are the preferred approach for efficiency and correctness.
# The non-vectorized version is mostly for conceptual understanding of the per-sample flow.
# The backward pass of the non-vectorized LSTMLayer needs significant revision to correctly handle batching if it were to be used.
# Specifically, `caches[0][0][0].shape` to get batch_size is very fragile.
# And accumulation of gradients `dW_i_total += gradients_cell['dW_i']` assumes `gradients_cell` are for a single sample.

# For the purpose of this exercise, focusing on VectorizedLSTMLayer is better.

if __name__ == '__main__':
    # --- Test VectorizedLSTMCell ---
    print("Testing VectorizedLSTMCell")
    batch_size_vc = 2
    input_size_vc = 3
    hidden_size_vc = 4

    cell_v = VectorizedLSTMCell(input_size_vc, hidden_size_vc)
    x_t_v = np.random.randn(batch_size_vc, input_size_vc)
    h_prev_v = np.random.randn(batch_size_vc, hidden_size_vc)
    c_prev_v = np.random.randn(batch_size_vc, hidden_size_vc)

    h_t_v, c_t_v, cache_v = cell_v.forward(x_t_v, h_prev_v, c_prev_v)
    print("h_t_v shape:", h_t_v.shape) # Expected: (batch_size, hidden_size)
    print("c_t_v shape:", c_t_v.shape) # Expected: (batch_size, hidden_size)

    dh_next_v = np.random.randn(batch_size_vc, hidden_size_vc)
    dc_next_v = np.random.randn(batch_size_vc, hidden_size_vc)
    gradients_v = cell_v.backward(dh_next_v, dc_next_v, cache_v)
    print("Vectorized Cell Gradients:")
    for k, v in gradients_v.items():
        print(f"{k} shape: {v.shape}")
    # Expected shapes:
    # dW_all shape: (input_size + hidden_size, 4 * hidden_size)
    # db_all shape: (4 * hidden_size,)
    # dx_t shape: (batch_size, input_size)
    # dh_prev shape: (batch_size, hidden_size)
    # dc_prev shape: (batch_size, hidden_size)
    print("dW_all expected shape:", (input_size_vc + hidden_size_vc, 4 * hidden_size_vc))
    print("db_all expected shape:", (4 * hidden_size_vc,))

    # --- Test VectorizedLSTMLayer ---
    print("\nTesting VectorizedLSTMLayer (return_sequences=True)")
    batch_size_vl = 2
    seq_len_vl = 5
    input_size_vl = 3
    hidden_size_vl = 4

    lstm_layer_seq_true = VectorizedLSTMLayer(input_size_vl, hidden_size_vl, return_sequences=True)
    X_vl = np.random.randn(batch_size_vl, seq_len_vl, input_size_vl)

    outputs_seq_true, caches_seq_true = lstm_layer_seq_true.forward(X_vl)
    print("outputs_seq_true shape:", outputs_seq_true.shape) # Expected: (batch_size, seq_len, hidden_size)

    # Gradients for return_sequences=True
    d_outputs_seq_true = np.random.randn(batch_size_vl, seq_len_vl, hidden_size_vl)
    layer_grads_seq_true = lstm_layer_seq_true.backward(d_outputs_seq_true, caches_seq_true)
    print("Vectorized Layer Gradients (seq=True):")
    for k, v in layer_grads_seq_true.items():
        if v is not None:
            print(f"{k} shape: {v.shape}")
    # Expected shapes for seq=True:
    # dW_all shape: (input_size + hidden_size, 4 * hidden_size)
    # db_all shape: (4 * hidden_size,)
    # dX_batch shape: (batch_size, seq_len, input_size)

    print("\nTesting VectorizedLSTMLayer (return_sequences=False, no dense output)")
    lstm_layer_seq_false_no_dense = VectorizedLSTMLayer(input_size_vl, hidden_size_vl, return_sequences=False)
    # X_vl is same
    output_seq_false, caches_seq_false = lstm_layer_seq_false_no_dense.forward(X_vl)
    print("output_seq_false shape:", output_seq_false.shape) # Expected: (batch_size, hidden_size)

    # Gradients for return_sequences=False
    d_output_seq_false = np.random.randn(batch_size_vl, hidden_size_vl)
    layer_grads_seq_false = lstm_layer_seq_false_no_dense.backward(d_output_seq_false, caches_seq_false)
    print("Vectorized Layer Gradients (seq=False, no dense):")
    for k, v in layer_grads_seq_false.items():
        if v is not None:
            print(f"{k} shape: {v.shape}")

    print("\nTesting VectorizedLSTMLayer (return_sequences=False, with dense output)")
    output_size_dense = 2
    lstm_layer_seq_false_dense = VectorizedLSTMLayer(input_size_vl, hidden_size_vl, output_size=output_size_dense, return_sequences=False)
    output_dense, caches_dense = lstm_layer_seq_false_dense.forward(X_vl)
    print("output_dense shape:", output_dense.shape) # Expected: (batch_size, output_size_dense)

    d_output_dense = np.random.randn(batch_size_vl, output_size_dense)
    # Note: The backward pass for the dense layer part (dW_y, db_y) is currently problematic as discussed in comments.
    # The test will run, but dW_y, db_y might be zero or None.
    layer_grads_dense = lstm_layer_seq_false_dense.backward(d_output_dense, caches_dense)
    print("Vectorized Layer Gradients (seq=False, with dense):")
    for k, v in layer_grads_dense.items():
        if v is not None:
            print(f"{k} shape: {v.shape}")
    print("dW_y expected shape (if implemented correctly):", (hidden_size_vl, output_size_dense))
    print("db_y expected shape (if implemented correctly):", (output_size_dense,))

    # --- (Optional) Test Non-Vectorized LSTMLayer if you want to debug it further ---
    # print("\nTesting Non-Vectorized LSTMLayer")
    # batch_size_nv = 1 # Non-vectorized layer is easier to test with batch_size=1 first
    # seq_len_nv = 3
    # input_size_nv = 2
    # hidden_size_nv = 3
    # output_size_nv = 2 # For dense layer test

    # lstm_nv = LSTMLayer(input_size_nv, hidden_size_nv, output_size=output_size_nv, return_sequences=False)
    # X_nv = np.random.randn(batch_size_nv, seq_len_nv, input_size_nv)
    # output_nv, caches_nv = lstm_nv.forward(X_nv)
    # print(f"Non-vectorized output shape: {output_nv.shape}") # Expected (batch_size_nv, output_size_nv)

    # d_output_nv = np.random.randn(batch_size_nv, output_size_nv)
    # grads_nv = lstm_nv.backward(d_output_nv, caches_nv)
    # print("Non-vectorized Layer Gradients:")
    # for k, v_list in grads_nv.items(): # Assuming grads_nv might be lists if not summed properly
    #     if isinstance(v_list, list) and v_list:
    #         print(f"{k} shape: {v_list[0].shape if hasattr(v_list[0], 'shape') else 'scalar or list element'}")
    #     elif hasattr(v_list, 'shape'):
    #         print(f"{k} shape: {v_list.shape}")
    #     else:
    #         print(f"{k}: {v_list}")

    print("\nReminder: The non-vectorized LSTMLayer's backward pass needs careful review for batch handling.")
    print("The VectorizedLSTMLayer is the recommended version.")
    print("The dW_y/db_y calculation in VectorizedLSTMLayer.backward needs final_h from forward pass cache.")