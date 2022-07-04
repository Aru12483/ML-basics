import random
import re
from pyrsistent import l
import tensorflow as tf 
from operator import matmul

#generating synthetic data :
def gen_data(w,b,num):
    x = tf.ones((num,w.shape[0])) # no of rows in x should be identical to no of rows in w
    x += tf.random.normal(shape=x.shape)
    y = tf.matmul(x,tf.reshape(w,(-1,1)))+b # -1 infers the length of the tensor 
    y += tf.random.normal(shape=y.shape,stddev=0.02)
    y = tf.reshape(y,(-1,1))
    return x,y

true_b = 2.5
true_w = tf.constant([5,-6])
features,labels = gen_data(true_w,true_b,500)

# function to read data :
def read_data(batch_size,features,labels):
    num_ex = len(features)
    indices = list(len(features))
    random.shuffle(indices)
    for i in range(0,batch_size,num_ex):
        j = tf.constant(indices[i:min(batch_size+i,num_ex)]) # make minibatches of the data 
        yield(tf.gather(features,j),tf.gather(labels,j))
    return 
batch_size = 20
for x,y in read_data(batch_size,features,labels):
    print(x,"\n",y)
    break

# initialise model params:
w = tf.Variable(tf.random.normal(2,1),stddev = 0.2,mean = 2,trainable=True)
b = tf.Variable(tf.random.ones(2,1),stddev = 0.2,mean = 2,trainable=True)

# define model:
def linreg(x,w,b):
    matmul(x,w)+b

# defining loss function : MSE
def squared_loss(y_hat,y):
    return (y_hat-tf.reshape(y,shape=y_hat))**2/2

# defining optimising algo : SGD
def sgd(params,grads,lr,bathc_size):
    for params,grads in zip(params,grads):
        params.assign_sub(lr*grads/batch_size)

# training model :
lr = 0.005
num_epochs = 100
reg = linreg
loss = squared_loss

#training loop :
for epoch in range (num_epochs):
    for x,y in read_data(features,labels,batch_size):
        with tf.GradientTape() as g:
            l = loss()
        
