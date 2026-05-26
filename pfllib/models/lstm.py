import torch
import torch.nn.functional as F
from torch import nn


class LSTMNet(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers=2,
        bidirectional=False,
        dropout=0.2,
        padding_idx=0,
        vocab_size=98635,
        num_classes=10,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if isinstance(x, list):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]

        embedded = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if isinstance(x, list):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out


class TextCNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_channels=100,
        kernel_size=None,
        max_len=200,
        dropout=0.8,
        padding_idx=0,
        vocab_size=98635,
        num_classes=10,
    ):
        super(TextCNN, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        if isinstance(x, list):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out
