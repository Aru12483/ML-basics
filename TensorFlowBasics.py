import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


#<tf.Tensor: shape=(5,), dtype="float32", np= array([1,2,3,4,5], dtype="float32")> 
x = tf.range(12, dtype=tf.float32)
p = tf.constant([[3,4,5,1],[4,5,6,2],[7,8,9,10]],dtype="float32")
print(x)    
x = tf.reshape(x,(3,4),) # reshapes vector to matrix 
print(x)
#p = tf.reshape(p,(3,4),)
# initialising tensor with either one or zero
#y = tf.range(12, dtype=tf.float32)
y=tf.zeros((3,4))
print(y)
z=tf.ones((1,2,3))
print(z)
t=tf.random.normal((3,4,2))
print(t)
#concat two tensors together
c=tf.concat([x,y],axis= 0),tf.concat([x,y],axis =1)
print(c)
#indexing and slicing 
x[-1], x[1:3]
#variables in tensorflow -> tensors are immutable but tensorflow variable supports change of state
#here in a variable type tensor x tensor is passesd in which we are changing the state of the first index to 11s
x_var=tf.Variable(x)
#reassigns value using the same tensor does not allocate a diff tensor to main memory
#most operations will work on variables but they cannot be reshaped. reshaping will create another tensor of desired size
x_var[0:2, :].assign(tf.ones(x_var[0:2,:].shape, dtype = tf.float32) * 12)
print(x_var)
#two variables never share memory


# if we assign a new value to a tensor we dereference it and the old value is lost 
# every time a new value is assigned a new memory location is also assigned to the tensor
before = id(y)
print(before)
y == x
print(id(x))
#unused values can be pruned by using the function decorator
tf.function
def computation(x,p):
    z = tf.zeros_like(y)
    a = x+p
    b = a+p
    c = p+b
    return c + p
o=computation(x,p)
print(o)

#converting to numpy arrays does not share memory therefore if we want to see what a certain package of numpy does we do not need to halt computation
A = x.numpy()
B = tf.constant(A)
type(A), type(B)


# creating and reading from a csv file 
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# opening the csv file
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

#to handle missing data we use Imputation or Deletion 

#1. Imputation
#select rows and columns to be used 
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
# replace missing values in NumRooms with mean
inputs = inputs.fillna(inputs.mean())
print(inputs)
# Replacing string values present with o or 1 for Pave or Nan
inputs = pd.get_dummies(inputs,dummy_na= True)
print(inputs)
# convert input and output to Tensors 
e,d = tf.constant(inputs.values),tf.constant(outputs.values)
print(e,d)

os.makedirs(os.path.join('..', 'data1'), exist_ok=True)
data_file1 = os.path.join('..', 'data1', 'house_tiny1.csv')
with open(data_file1,'w') as f:
    f.write("Name,Class,RollNo\n")
    f.write("Aru,E,12\n")
    f.write("Sanyam,A,12\n")
    f.write("Sanyam,A,13\n")
    f.write("Aru,A,11\n")
data1 = pd.read_csv(data_file1)
print(data1)

inp,out = data1.iloc[:,0:1,],data1.iloc[:,1]
n,m = tf.constant(inp),tf.constant(out)
print(n,m)

# TENSOR ARITHMETIC : already done basics of tensor arithmetic 
# Reduction and Non reduction sums:
p = tf.reduce_sum(x)
print(p)
r = tf.reduce_sum(x, axis=1, keepdims=True)
print(r)
# Dot Product of two arrays 
u = tf.constant([[1,2,3],[3,4,5],[6,7,8]],dtype= tf.float32)
i = tf.constant([[11,12,13],[15,16,17],[19,20,21]],dtype= tf.float32)
u,i, tf.tensordot(u,i, axes=1)
fd = tf.reduce_sum(u*i)
print(fd)
# vector product or cross product of two arrays
vp = tf.linalg.matvec(u,i)
print(vp)
# matrix multiplication of two arrays 
op = tf.matmul(u,i)
print(op)
# Implementing Norms :

#1. L2 and L1 norms in tensorflow 
l2 = tf.norm(u)
print(l2)
l1 = tf.reduce_sum(tf.abs(u))
print(l1)
# CALCULUS IN DL 
# limits and derivatives : 
# function to calculate limit
def f(x1):
    return (3*x1**2)-4*x1
def numerical_lim(f, x1, h):
    return (f(x1 + h) - f(x1)) / h
# iterating 5 times  
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
# GRADIENTS 
q = tf.range(4,dtype= tf.float32)
q = tf.Variable(q)
# function to calculate tensor for w = 2q
with tf.GradientTape() as t:
    w = 2*tf.tensordot(q,q, axes = 1)
w
# calling function for backpropogation 
w_grad = t.gradient(w, q)
print(w_grad)
# final result should be 4x 
#w_grad = 4*w_grad

print(w_grad)
# sum reduction function overwrite the gradient function 

# Detatching for Computation : suppose fz is a function dependant on two functions fx an fy and fy in turn i s dependant on 
# fx and we need to take fy as constant
with tf.GradientTape(persistent = True) as t :
    q = q * q 
    r = tf.stop_gradient(u)
    z = q * w
q_grad = t.gradient(z, x)
q_grad == u
# Commputing gradient with flow of control statements 
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
# method that calculates the gradient of a 
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
d_grad == d / a

