import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_levels, num_layers=1, bidirectional=True, dropout=0, eps=1e-5,
                 momentum=0.9):
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )
        self.readout = nn.Linear(
            in_features=hidden_size * (1 + bidirectional),
            out_features=num_levels
        )

    def forward(self, sentences):
        lengths = np.array([len(sentence) for sentence in sentences])

        padded_sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)

        packed_sentences = nn.utils.rnn.pack_padded_sequence(padded_sentences, lengths=lengths, batch_first=True)

        rnn_output, hidden = self.rnn(packed_sentences)

        unpacked_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        return self.readout(unpacked_output)
