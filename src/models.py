import numpy as np
import torch
import torch.nn as nn


class RNN_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_labels, num_layers=1, bidirectional=True, dropout=0):
        super(RNN_Model, self).__init__()
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
            out_features=num_labels
        )

    def forward(self, sentences):
        lengths = torch.tensor([len(sentence) for sentence in sentences])

        padded_sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)

        packed_sentences = nn.utils.rnn.pack_padded_sequence(padded_sentences, lengths=lengths, batch_first=True)

        rnn_output, hidden = self.rnn(packed_sentences.float())

        unpacked_output, size_list = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = I.expand(unpacked_output.size(0), 1, unpacked_output.size(-1)) - 1
        last_output = unpacked_output.gather(dim=1, index=I).squeeze(1)

        return self.readout(last_output)
