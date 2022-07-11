import sys
from matplotlib import pyplot
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers 


# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
	model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.22))
	model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.22))
	model.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.22))
	#model.add(Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	#model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'),kernel_regularizer=regularizers.L1L2(l1=0.1, l2=0.1),bias_regularizer=regularizers.L2(0.01),activity_regularizer=regularizers.L2(0.01))
	model.add(Dropout(0.22))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()

#IDEAL SCENARIO
# genererally Relu as an activation function and he uniform are considered as the norm as they provide better results -> why? 
# experimentally attained value that ReLu + he_uniform is the best performing initializer and activation function combo 


# CHANGE IN INITIALIZERS
# inital accuracy with 4 stacks of comvolution layers  with he uniform is : 92.990 %
# changing kernel_initalizer from he uniform to normally initialised : resulted in accuracy of 92.030 so slight drop and it took 13mins 24 seconds instead of 12 mins approx 
# changing the kernel_initializer to GlorotNormal : accuracy -> 92.020% and same time of 13 mins and 24 seconds 
# changing initialiser to lecun_normal dropped accuracy even further to 92.1290 % time of execution remains unchanged 
# changing the initialiser to variance scaling : 92.520
# changing to lecun_uniform 92.080 % time taken is the same 
# changing to he_normal gives 92.660 accuracy 

#CHANGE IN BLOCKS FROM 4 TO 5 BLOCKS : 32-64-128-256-512 LAYERS WITH DROPOUTS MAX POOLING LAST LAYER IS FLATTENED AND CONNECTED TO A FULLY CONNECTED LAYER 
# Accuracy drops after adding another layer and computation time goes up to 21 minutes and 34 seconds 91.798
# dropping a layer and adding a 256 dense layer fully connected : drops accuracy to 92.720 still lower than initial runtime the same as 4 layers previous

#CHANGE IN REGULARISER 
# adding activity kernel and bias regularizer with l1 and l2 values as 0.1 
