import numpy as np
import torch


UNK = 'unk'

def iterate_data(sentences, labels, embeddings, batch_size, shuffle= True):
    indx = list(range(len(labels)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(labels), batch_size):
        end_idx = min(start_idx + batch_size, len(labels))
        batch_sentences, batch_labels = sentences[indx[start_idx: end_idx]], labels[indx[start_idx: end_idx]]

        batch_sentences, batch_labels = sort_batch(batch_sentences, batch_labels)
        tensor_sentences = embed(batch_sentences, embeddings)
        tensor_labels = torch.tensor(batch_labels).long()
        yield tensor_sentences, tensor_labels # Ordered sequences of tensorized embedded sentences and labels.


def sort_batch(batch_sentences, batch_labels):
    lengths = np.array([len(sentence) for sentence in batch_sentences])
    idx = np.argsort(-lengths)
    return batch_sentences[idx], batch_labels[idx]


def embed(batch_sentences, embeddings):
    embedded_sentences= []
    for sentence in batch_sentences:
        sentence_embedding = [embeddings.get(word, embeddings[UNK]) for word in sentence]
        embedded_sentences.append(torch.tensor(sentence_embedding))
    return embedded_sentences
