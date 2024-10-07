import numpy
def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x,alpha=0.01):
    return np.where(x>0,x,alpha*x)

def softmax(x):
    exps=np.exp(x-np.max(x))
    return exps/np.sum(exps,axis=0)