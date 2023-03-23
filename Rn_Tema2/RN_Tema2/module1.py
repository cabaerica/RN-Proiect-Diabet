import numpy as np
import pickle as _pickle
import gzip
import numpy as np

globalNrIterations = 1
learningRate = 0.05

def getTrainSet():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = _pickle.load(f , encoding = "latin1")
    f.close()
    images = train_set[0]
    labels = train_set[1]
    return images, labels

def getTestSet():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = _pickle.load(f , encoding = "latin1")
    images = test_set[0]
    labels = test_set[1]
    f.close()
    return test_set

def sigmoid(x):
    y = np.exp(-x)
    return np.divide(1.0, 1 + y)

def softmax(x):
    y = np.exp(x)
    return y / y.sum()

def trainNeuralNetwork():
    train_set = getTrainSet()
    input = np.array(train_set[0])
    target = np.array(train_set[1])
    firstLayer = np.random.normal(0, np.power(np.sqrt(784), (-1)), (100, 784))
    lastLayer = np.random.normal(0, np.power(np.sqrt(100), (-1)), (10, 100))

    firstBiases = np.random.normal(0, 1, 100)
    lastBiases = np.random.normal(0, 1, 10)

    nrIterations = globalNrIterations

    while(nrIterations > 0):
        for index in range(len(input)):
            print(index)
            firstOutput = sigmoid(firstLayer.dot(input[index]) + firstBiases)
            lastOutput = softmax(lastLayer.dot(firstOutput) + lastBiases)

            t = np.zeros(10)
            t[target[index]] = 1
            lastError = lastOutput - t
            lastError = lastError.reshape((10, 1))

            firstError = (firstOutput * (1 - firstOutput)) * (lastLayer * lastError).sum(axis=0)
            firstError = firstError.reshape((100, 1))

            lastLayer += -learningRate * lastError * firstOutput
            firstLayer += -learningRate * firstError * input[index]
            lastBiases += -learningRate * lastError.flatten()
            firstBiases += -learningRate * firstError.flatten()
        nrIterations -= 1
    return firstLayer, lastLayer, firstBiases, lastBiases


def testNeuralNetwork(firstLayer, lastLayer, firstBiases, lastBiases):
    test_set = getTestSet()
    success = 0
    failure = 0
    inputValues = np.array(test_set[0])
    target = np.array(test_set[1])
    for index in range(len(inputValues)):
        print(index)
        firstOutput = sigmoid(firstLayer.dot(inputValues[index]) + firstBiases)
        lastOutput = softmax(lastLayer.dot(firstOutput)+lastBiases)
        if np.argmax(lastOutput) == target[index]:
            success += 1
    
    print("Success: {}".format(success))

def main():
    print("somthinfsdf")
    firstLayer, firstBiases, lastLayer, lastBiases = trainNeuralNetwork()
    testNeuralNetwork(firstLayer, firstBiases, lastLayer, lastBiases)

main()