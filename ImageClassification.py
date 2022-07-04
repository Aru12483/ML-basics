import tensorflow as tf
import numpy as np
import time
#from d2l import tensorflow as d2l 

# reading the dataset : Here we use the MNIST dataset (handwritten digits from 0-9)
#mnist_test,mnist_train = tf.keras.datasets.fashion_mnist.load_data()
#print(len(mnist_test[0]))
#print(len(mnist_train[0]))
#print(mnist_train[0][0].shape)

#start = time.process_time()
# define the batch size and store the data for training into train_iter
#batch_size = 512
#train_iter = tf.data.Dataset.from_tensor_slices(mnist_train).batch(batch_size).shuffle(len(mnist_train[0])) 
#print(time.process_time() - start)

def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break