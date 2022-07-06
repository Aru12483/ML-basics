
import sys
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import tensorflow as tf 

cifar10 = tf.keras.datasets.cifar10
(image_train, label_train), (image_test, label_test) = cifar10.load_data() # tuples 
# as the labels here are also mapped to integers from 0-9 class/label generation is manual 

print(len(image_train))
print(len(image_test))
print(len(label_train))
print(len(label_test))

print(image_test.shape)

print('Train: X=%s, y=%s' % (image_train.shape, label_train.shape))
print('Test: X=%s, y=%s' % (image_test.shape, label_test.shape))
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(image_train[i])
# show the figure
plt.show()
# coonvert test and training set to standard values 

image_train = image_train/255
image_test = image_test/255


# Training phase : Make max pooling layers , blocks of 32,64,128 then 256 and introduce padding with appropriate strides 
def model(model): 
    model = Sequential([MaxPool2D(pool_size = 2, strides = 2)])
    # example output part of the model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    return model

