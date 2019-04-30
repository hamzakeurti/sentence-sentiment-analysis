import numpy as np


# Taken from answer https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def loadGloveEmbedding(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model