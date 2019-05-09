import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import loading
import batching
import models
from utils import LOG_INFO



embeddings = loading.load_glove_embedding(loading.GLOVE)
train_sentences, train_labels = loading.parse_tree(loading.TREES_T)

test_sentences, test_labels = loading.parse_tree(loading.TREES_S)

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
LOG_INTERVAL = 1
EPOCHS = 10

# Params 1
LR = 0.01
MM = 0.9
WD = 1e-6

# # Params 2
# LR = 0.1
# MM = 0.9
# WD = 1e-6


#  Model 1

model = models.RNN_Model(
    embedding_dim=300,
    hidden_size=300,
    num_labels=5,
    num_layers=1,
    bidirectional=True,
    dropout=0
)

# #  Model 2
# kernel_size = 3
# avg_window = 2
# model = nn.Sequential(OrderedDict([
#     ('conv1', nn.Conv2d(1,4,kernel_size,padding=1)), # out (,4,28,28)
#     # ('batch1', BatchNorm2d(4)),
#     ('relu1', nn.ReLU()),
#     ('avg1', nn.AvgPool2d(avg_window)), # out (,4,14,14)
#     ('conv2', nn.Conv2d(4,4,kernel_size,padding = 1)),
#     # ('batch2', BatchNorm2d(4)),
#     ('relu2', nn.ReLU()),
#     ('avg2', nn.AvgPool2d(avg_window)), #out (,4,7,7)
#     ('flatten',Reshape((-1 ,4*7*7))),
#     ('fc',nn.Linear(4*7*7,10))
# ])).to(device)


optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MM, weight_decay=WD)


def train(model, optimizer, epoch, sentences, labels, verbosity = True):
    model.train()
    loss_list = []
    acc_list = []
    batch_idx = 0
    for data, target in batching.iterate_data(sentences,labels,embeddings,TRAIN_BATCH_SIZE):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        if batch_idx % LOG_INTERVAL == 0:
            if verbosity:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(sentences),
                           100. * batch_idx / len(sentences), np.mean(loss_list), np.mean(acc_list))
                LOG_INFO(msg)
                loss_list.clear()
                acc_list.clear()
        batch_idx += 1
    return loss_list,acc_list


def test(model, sentences, labels,verbosity = True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in batching.iterate_data(sentences,labels,embeddings,TEST_BATCH_SIZE):
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(sentences)
    if verbosity:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(sentences),
            100. * correct / len(sentences)))
    return test_loss, correct / len(sentences)


if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(model=model,
              optimizer=optimizer,
              epoch=epoch,
              sentences=train_sentences,
              labels=train_labels)
        test(model = model,
            sentences = test_sentences,
            labels = test_labels)
