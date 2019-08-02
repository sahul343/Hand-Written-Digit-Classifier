#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
import os
def load_dataset():
    def download(filename,source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading",filename)
        import urllib
        urllib.request.urlretrieve(source+filename,filename)
    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            data=np.frombuffer(f.read(),np.uint8,offset=16)
            data=data.reshape(-1,784,1)
        return data/np.float32(256)
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            data=np.frombuffer(f.read(),np.uint8,offset=8)
            #data=data.reshape(-1,1,1)
            d=np.zeros((len(data),10,1))
            for i in range(len(data)):
                d[i][data[i]][0]=1
        return d
    x_train=load_mnist_images('train-images-idx3-ubyte.gz')
    y_train=load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test=load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test=load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    train_data=[(x,y) for (x,y) in zip(x_train,y_train)]
    test_data=[(x,y) for (x,y) in zip(x_test,y_test)]
    return train_data,test_data
train_data,test_data=load_dataset()         
     
          
class network():
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for (x,y) in zip(sizes[:-1],sizes[1:])]
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        n=len(training_data)
        if test_data:
            n_test=len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
        if test_data:
            print(self.evaluate(test_data))
    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.back_prop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] 
        self.weights=[w-(eta/len(mini_batch))*nw for (w,nw) in zip(self.weights,nabla_w)]
        self.biasses=[b-(eta/len(mini_batch))*nb for (b,nb) in zip(self.biases,nabla_b)]
    def back_prop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            #print(w.shape,activation.shape)
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        delta=self.cost_derivative(activations[-1],y)*sigmoid_derivative(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z=zs[-l]
            sp=sigmoid_derivative(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)
            
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        return sum([int(x==y) for (x,y) in test_results])
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
    


# In[4]:


net=network([784,16,10])
net.SGD(train_data,30,10,3.0,test_data=test_data)



def evaluate(test_data):
    test_results=[(np.argmax(net.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
    return sum([int(x==y) for (x,y) in test_results])
evaluate(test_data)





# print(plt.imshow(train_data[0][0].reshape(28,28)))





