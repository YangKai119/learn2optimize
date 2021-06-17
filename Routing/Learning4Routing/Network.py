import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_log_softmax(vector:torch.Tensor, mask:torch.Tensor, dim:int=-1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in log space, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-40).log()
    return F.log_softmax(vector, dim=dim)

def masked_max(vector:torch.Tensor, mask:torch.Tensor, dim:int, keep_dim:bool=False, min_val:float=-1e7) -> (torch.Tensor, torch.Tensor):
    """
    计算最大值：在masked值的特定的维度
    :param vector: 计算最大值的vector，假定没有mask的部分全是0
    :param mask: vector的mask，必须是可以扩展到vector的形状
    :param dim: 计算max的维度
    :param keep_dim: 是否保持dim
    :param min_val: paddings的最小值
    :return: 包括最大值的Tensor
    """
    one_minus_mask = ~mask
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keep_dim)
    return max_value, max_index

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(Encoder, self).__init__()
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths):
        # 打包RNN的填充序列
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths.cpu(), batch_first=self.batch_first)
        # forward pass through RNN
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # 返回输出和最终状态
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)
        # (batch_size, 1(unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # softmax with only valid inputs, excluding zero padded parts
        # log_softmax for a better numerical stability
        log_score = masked_log_softmax(u_i, mask, dim=-1)
        return log_score

class PointerNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, bidirectional=True, batch_first=True):
        super(PointerNet, self).__init__()
        # Embedding dimension
        self.embedding_dim = embedding_dim
        # decoder hidden size
        self.hidden_size = hidden_size
        # bidirectional encoder
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = 1
        self.batch_first = batch_first

        # 我们将嵌入层用于以后更复杂的应用程序用法，例如单词序列。
        self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim,bias=False)
        self.encoder = Encoder(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=self.num_layers, bidirectional=bidirectional,batch_first=batch_first)
        self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = Attention(hidden_size=hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_seq, input_lengths):
        if self.batch_first:
            batch_size = input_seq.size(0)
            max_seq_len = input_seq.size(1)
        else:
            batch_size = input_seq.size(1)
            max_seq_len = input_seq.size(0)

        # embedding
        embedded = self.embedding(input_seq)
        # (batch_size, max_seq_len, embedding_dim)

        # encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first else (max_seq_len, batch_size, hidden_size)
        # hidden_size is usually set same as embedding size
        # encoder_hidden => (num_layers * num_directions, batch_size, hidden_size) for each of h_n and c_n
        encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)

        if self.bidirectional:
            # Optionally, Sum bidirectional RNN outputs
            # (batch_size, max_seq_len, hidden_size)
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

        encoder_h_n, encoder_c_n = encoder_hidden
        # (1, 2, batch_size, hidden_size)
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        # Let's use zeros as an initial input
        # (batch_size, hidden_size)
        decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        # ((batch_size, hidden_size), (batch_size, hidden_size))
        decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())

        # (batch_size, max_seq_len, max_seq_len)
        range_tensor = torch.arange(max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(batch_size, max_seq_len, max_seq_len)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)

        # (batch_size, max_seq_len, max_seq_len)
        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            # we will simply mask out when calculating attention or max (and loss later)
            # not all input and hidden, just for simplicity
            # (batch_size, max_seq_len)
            sub_mask = mask_tensor[:, i, :]

            # h,c is both (batch_size, hidden_size)
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

            # next hidden
            decoder_hidden = (h_i, c_i)

            # get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
            pointer_log_scores.append(log_pointer_score)

            # get the indices of maximum pointer
            # (batch_size, 1)
            _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keep_dim=True)

            pointer_argmaxs.append(masked_argmax)

            # (batch_size, 1, hidden_size)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

            # encoder_outputs为(batch_size, max_seq_len, hidden_size)
            # index为(batch_size, 1, hidden_size)，且所有hidden_size个的数据都是一样的，都是0-30的数字
            # decoder_input: (batch_size , 1, hidden_size).squeeze(1)即(batch_size, hidden_size)
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        # stack是叠加，会增加一个维度
        # t * (batch_size, max_seq_len) t为max_seq_len, stack之后变成(batch_size, max_seq_len, max_seq_len)
        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        # cat是在现有维度上续接，不会产生新维度
        # t * (batch_size, 1) cat之后变成 (batch_size, max_seq_len)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)

        return pointer_log_scores, pointer_argmaxs, mask_tensor








