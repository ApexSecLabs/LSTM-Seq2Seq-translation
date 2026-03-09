import torch
import torch.nn as nn
from torchinfo import summary


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, bidirectional, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # LSTM 中的 dropout 参数仅在 num_layers > 1 时有效
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        if bidirectional:
            # 为每一层创建独立的线性变换层
            self.fc_hidden_layers = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)])
            self.fc_cell_layers = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)])
            self.dropout_fc = nn.Dropout(dropout)   # <--- 新增变换后的 Dropout

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))   # <--- 应用 Dropout
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        if self.bidirectional:
            batch_size = hidden.size(1)
            hidden_layers = []
            cell_layers = []
            for i in range(self.num_layers):
                h_fwd = hidden[2 * i, :, :]
                h_bwd = hidden[2 * i + 1, :, :]
                h_cat = torch.cat((h_fwd, h_bwd), dim=1)
                h_transformed = self.dropout_fc(self.fc_hidden_layers[i](h_cat))  # <--- 应用 Dropout
                hidden_layers.append(h_transformed)

                c_fwd = cell[2 * i, :, :]
                c_bwd = cell[2 * i + 1, :, :]
                c_cat = torch.cat((c_fwd, c_bwd), dim=1)
                c_transformed = self.dropout_fc(self.fc_cell_layers[i](c_cat))    # <--- 应用 Dropout
                cell_layers.append(c_transformed)

            hidden = torch.stack(hidden_layers, dim=0)
            cell = torch.stack(cell_layers, dim=0)
        # 非双向时无需额外处理
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)          # <--- 新增 Dropout 层
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell):
        embedded = self.dropout(self.embedding(input_token))   # <--- 应用 Dropout
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.target_vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_len)
        input_token = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = (trg[:, t] if teacher_force else top1).unsqueeze(1)

        return outputs