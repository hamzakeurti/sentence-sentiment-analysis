import numpy as np
import re

import torch

GLOVE = "../data/glove.6B.300d.txt"
TREES_DIR = "../data/trainDevTestTrees_PTB/"
STANFORD_DIR = "../data/stanfordSentimentTreebank/"
TREES_T = TREES_DIR + "train.txt"
TREES_D = TREES_DIR + "dev.txt"
TREES_S = TREES_DIR + "test.txt"

RE_WORD = re.compile(' (\w+)\)')

UNK = 'unk'

# Taken from answer https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def load_glove_embedding(glove_file):
    print("Loading Glove Model")
    with open(glove_file, 'r', encoding="utf8") as f:
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def parse_tree(tree_file):
    sentences = []
    labels = []
    with open(tree_file, 'r', encoding="utf8") as f:
        for line in f:
            found = re.findall(RE_WORD, line)
            if not found:
                continue
            sentences.append(np.char.lower(found))
            labels.append(int(line[1]))
    return np.array(sentences), labels


def lists_to_tensors(sentences, labels, embeddings):
    labels_tensor = torch.tensor(labels)
    sentence_sequence = []
    for sentence in sentences:
        sentence_embedding = [embeddings.get(word,embeddings[UNK]) for word in sentence]
        sentence_sequence.append(torch.tensor(sentence_embedding))
    return sentence_sequence, labels_tensor
