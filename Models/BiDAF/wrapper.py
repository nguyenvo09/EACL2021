import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s' % i))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s' % i))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s' % i), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s' % i), val=0)
            getattr(self.rnn, 'bias_hh_l%s' % i).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s_reverse' % i))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s_reverse' % i))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s_reverse' % i), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s_reverse' % i), val=0)
                getattr(self.rnn, 'bias_hh_l%s_reverse' % i).chunk(4)[1].fill_(1)

    def forward(self, x, return_h=True, max_len=None):
        x, x_len, d_new_indices, d_restoring_indices = x
        x = self.dropout(x)
        # x_idx = d_new_indices
        x_len_sorted = x_len[d_new_indices]
        # x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x[d_new_indices]  # x.index_select(dim=0, index=x_idx)
        x_ori_idx = d_restoring_indices
        # _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=max_len)[0]
        # x = x.index_select(dim=0, index=x_ori_idx)
        x = x[x_ori_idx]
        if return_h:
            h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)) #.squeeze()
            # h = h.index_select(dim=0, index=x_ori_idx)
            h = h[x_ori_idx]
        return x, h


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s' % i))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s' % i))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s' % i), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s' % i), val=0)
            getattr(self.rnn, 'bias_hh_l%s' % i).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s_reverse' % i))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s_reverse' % i))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s_reverse' % i), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s_reverse' % i), val=0)
                getattr(self.rnn, 'bias_hh_l%s_reverse' % i).chunk(4)[1].fill_(1)

    def forward(self, x, return_h=True, max_len=None):
        x, x_len, d_new_indices, d_restoring_indices = x
        x = self.dropout(x)
        # x_idx = d_new_indices
        x_len_sorted = x_len[d_new_indices]
        # x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x[d_new_indices]  # x.index_select(dim=0, index=x_idx)
        x_ori_idx = d_restoring_indices
        # _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        # x_packed, (h, c) = self.rnn(x_packed)
        x_packed, h = self.rnn(x_packed)  # this is for GRU not LSTM

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=max_len)[0]
        # x = x.index_select(dim=0, index=x_ori_idx)
        x = x[x_ori_idx]
        if return_h:
            h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2))  # .squeeze()
            # h = h.index_select(dim=0, index=x_ori_idx)
            h = h[x_ori_idx]
        return x, h


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'): x = self.dropout(x)
        x = self.linear(x)
        return x
