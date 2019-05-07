import numpy as np
import torch

UNK = "unk"


def batch_generator(sentences, labels, embeddings, batch_size, shuffle=True):
    indx = list(range(len(labels)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(labels), batch_size):
        end_idx = min(start_idx + batch_size, len(labels))
        batch_sentences, batch_labels = sentences[indx[start_idx: end_idx]], labels[indx[start_idx: end_idx]]

        padded_sentences = pad_batch(batch_sentences)
        tensor_sentences = embed(padded_sentences, embeddings)

        tensor_labels = torch.tensor(batch_labels)

        yield tensor_sentences, tensor_labels


def pad_batch(batch_sentences):
    lengths = [len(sentence) for sentence in batch_sentences]
    padded = np.chararray((len(lengths), max(lengths)))
    padded[:] = UNK
    for i, length in enumerate(lengths):
        sentence = batch_sentences[i]
        padded[i, 0:length] = sentence[:length]
    return padded


def embed(padded_sentences, embeddings):
    sentence_embeddings = []
    for sentence in padded_sentences:
        sentence_embedding = [embeddings.get(word, embeddings[UNK]) for word in sentence]
        sentence_embeddings.append(torch.tensor(sentence_embedding))
    return torch.stack(sentence_embedding)
