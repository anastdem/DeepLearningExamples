import torch
import torch.nn as nn
import torch.nn.functional as F


class GST(nn.Module):
    def __init__(self, ref_enc_filters, n_mel_channels, encoder_embedding_dim, gst_tokens, gst_num_heads):
        super().__init__()
        self.encoder = ReferenceEncoder(ref_enc_filters, n_mel_channels, encoder_embedding_dim)
        self.stl = STL(encoder_embedding_dim, gst_tokens, gst_num_heads)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed, alphas = self.stl(enc_out)
        return style_embed, alphas


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''
    def __init__(self, encoder_embedding_dim, gst_tokens, gst_num_heads):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(gst_tokens, encoder_embedding_dim // gst_num_heads))
        self.attention = MultiHeadAttention(query_dim=encoder_embedding_dim // 2,
                                            key_dim=encoder_embedding_dim // gst_num_heads,
                                            num_units=encoder_embedding_dim,
                                            num_heads=gst_num_heads)

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, alphas = self.attention(query, keys)
        return style_embed, alphas


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = torch.softmax(scores, dim=-1)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        return out, scores


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''
    def __init__(self, ref_enc_filters, n_mel_channels, encoder_embedding_dim):
        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)])
        self.n_mels = n_mel_channels

        self.conv_params = {
            "kernel_size": 3,
            "stride": 2,
            "pad": 1,
            "n_convs": K
        }

        out_channels = self.calculate_size(self.n_mels, **self.conv_params)

        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=encoder_embedding_dim // 2,
                          batch_first=True)

    def forward(self, inputs, input_lengths=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            _input_lengths = self.calculate_size(input_lengths, **self.conv_params)
            out = nn.utils.rnn.pack_padded_sequence(
                out, _input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    @staticmethod
    def calculate_size(dim_size, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            dim_size = (dim_size - kernel_size + 2 * pad) // stride + 1
        return dim_size
