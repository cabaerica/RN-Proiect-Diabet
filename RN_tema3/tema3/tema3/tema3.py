import numpy as np
import pickle as cPickle
import gzip
import numpy as np

globalNrIterations = 1
learningRate = 0.07

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
def getTrainSet():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f , encoding = "latin1")
    f.close()
    images = train_set[0]
    labels = train_set[1]
    return images, labels

def getTestSet():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f , encoding = "latin1")
    images = test_set[0]
    labels = test_set[1]
    f.close()
    return test_set


def sigmoid(x):
    y = np.exp(-x)
    return np.divide(1.0, 1 + y)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    y = np.exp(x)
    return y / y.sum()
def cross_entropy_derivative(output, target):
		return output - target

def  backpropagation(out1,out2,target,L_1,L_2):
        err_23 = cross_entropy_derivative(out2 ,target)
        err_23=err_23.reshape((10, 1)) 
        bias_gradient23 =  err_23
        weight_gradient23 = np.dot(err_23, out1)
        # eroare pentru layer 2
        err_12 = np.dot(L_1.transpose(), err_23) 
        weight_gradient12 = np.dot(err_12, L_2)
        bias_gradient12 =  err_12
        err_12=err_12.reshape((100, 1))
        return bias_gradient23, weight_gradient23,weight_gradient12,bias_gradient12
       
    
def train():
    nrIterations=1
    train_set = getTrainSet()
    t_set=np.array(train_set[0]) # traing set 
    t=np.array(train_set[1]) ## target
    L_1=np.random.normal(0,np.power(np.sqrt(784),(-1)) ,(100,784)) ## weights initilization
    L_2=np.random.normal(0, np.power(np.sqrt(100), (-1)), (10, 100))

    b1=np.random.normal(0,1,100) ## bias layer 1 initializare
    b2=np.random.normal(0,1,10) ## bias last layer initializare
    while (nrIterations >0):
    
     for  index in range(len(t_set)):
        x= np.dot(L_1,t_set[index])+b1
        out1=sigmoid(x)
        y= L_2.dot(out1)+b2
        out2=softmax(y)
        ## eroare pentru layer 3 (cross entropy)
        target = np.zeros(10)
        target[t[index]]=1
        bias_gradient23, weight_gradient23,weight_gradient12,bias_gradient12 = backpropagation(out1,out2,target,L_1,L_2)
       
        learningRate=0.7
        L_1 += -learningRate *weight_gradient23
        L_2 += -learningRate * weight_gradient12
        b2 += -learningRate * bias_gradient23
        b1 += -learningRate * bias_gradient12
     nrIterations-=1
    return L_1,L_2,b1,b2
def testNeuralNetwork(firstLayer, lastLayer, firstBiases, lastBiases):
    
    success = 0
   
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
    firstLayer, firstBiases, lastLayer, lastBiases = train()
    testNeuralNetwork(firstLayer, firstBiases, lastLayer, lastBiases)

main()















