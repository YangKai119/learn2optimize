import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_log_softmax(vector:torch.Tensor, mask:torch.Tensor, dim:int=-1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-40).log()
    return F.log_softmax(vector, dim=dim)

def masked_max(vector:torch.Tensor, mask:torch.Tensor, dim:int, keep_dim:bool=False, min_val:float=-1e7) -> (torch.Tensor, torch.Tensor):
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
        outputs, hidden = self.rnn(packed)
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

class PointerNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, bidirectional=True, batch_first=True):
        super(PointerNetwork, self).__init__()
        # Embedding dimension
        self.embedding_dim = embedding_dim
        # decoder hidden size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = 1
        self.batch_first = batch_first
        
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
        encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

        encoder_h_n, encoder_c_n = encoder_hidden
        # (1, 2, batch_size, hidden_size)
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        # ((batch_size, hidden_size), (batch_size, hidden_size))
        decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())

        range_tensor = torch.arange(max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(batch_size, max_seq_len, max_seq_len)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)

        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            sub_mask = mask_tensor[:, i, :]
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)
            decoder_hidden = (h_i, c_i)
            log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
            pointer_log_scores.append(log_pointer_score)
            # (batch_size, 1)
            _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keep_dim=True)
            pointer_argmaxs.append(masked_argmax)
            # (batch_size, 1, hidden_size)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)
        # stack是叠加，会增加一个维度
        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        # cat是在现有维度上续接，不会产生新维度
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)

        return pointer_log_scores, pointer_argmaxs, mask_tensor








