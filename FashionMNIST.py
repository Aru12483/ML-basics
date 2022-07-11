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
#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Nadam
#from tensorflow.keras.optimizers import Adagrad
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

# IDEALLY we should use a 3X3 or a 5X5 kernel size EVEN size kernels not ideal as previous layer pixels gets divided around the output layer 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(32, (3,3),))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((3, 3)))
	model.add(Dropout(0.2)) # probability to droput a layer 
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((3, 3)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	#model.add(Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	#model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(512, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform',))
    #kernel_regularizer=regularizers.l1(l1=1e-5),  # removing l2 
		#bias_regularizer=regularizers.L2(1e-4),
    #activity_regularizer=regularizers.l1(1e-5)))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	#opt = SGD(learning_rate=0.001, momentum=0.9)
	opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam",)
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





#initial accuracy is 75%

# genererally Relu as an activation function and he uniform are considered as the norm as they provide better results -> why? 
# experimentally attained value that ReLu + he_uniform is the best performing initializer and activation function combo 


#CHANGING KERNEL SIZE :
# 1X1 - 89.89
# 2X2 - 93 - exception here since images are 28x28 
# 3X3 - last pooling layer should be 2X2 93
# small kernel size for images smaller than 128X128 in size ideal, most popular used in all broad image classification
# 5X5 
# 7X7 reserved for larger images
# 1X3 3X1 - Google ImageNet 
# 1X1 g4 relu then 3X3 128 relu -> ResNet relies on strided convolution than using pooling layers 

# CHANGING OPTIMZERS (try nesterov momentum/momentum in all applicable )
# SGD - base result is 93%
# SGD + nesterov
# Adam - 94% 10 minute runtime
# (Nadam)Adam + nesterov momentum - longer runtime same results 
# RMSProp -
# AdaGrad - runtime increased accuracy down 89.4
# AdaMax - 
# AdaDelta -

# CHANGE IN INITIALIZERS
# inital accuracy with 4 stacks of comvolution layers  with he uniform is : 92.990 %
# changing kernel_initalizer from he uniform to normally initialised : resulted in accuracy of 92.030 so slight drop and it took 13mins 24 seconds instead of 12 mins approx 
# changing the kernel_initializer to GlorotNormal : accuracy -> 92.020% and same time of 13 mins and 24 seconds 
# changing initialiser to lecun_normal dropped accuracy even further to 92.1290 % time of execution remains unchanged 
# changing the initialiser to variance scaling : 92.520
# changing to lecun_uniform 92.080 % time taken is the same 
# changing to he_normal gives 92.660 accuracy 
# 4 layers and batch normalisation in that also decreases accuracy therefore 3 LAYERS OPTIMAL 


#CHANGE IN BLOCKS FROM 4 TO 5 BLOCKS : 32-64-128-256-512 LAYERS WITH DROPOUTS MAX POOLING LAST LAYER IS FLATTENED AND CONNECTED TO A FULLY CONNECTED LAYER 
# Accuracy drops after adding another layer and computation time goes up to 21 minutes and 34 seconds 91.798
# dropping a layer and adding a 256 dense layer fully connected : drops accuracy to 92.720 still lower than initial runtime the same as 4 layers previous
# slight increase in dropout layer yields 92.870 within randomness though also runtime was the same at around 10 minutes 
# adding batch normalisation to the layers 
# batch normalisation increases accuracy to 93.270% time has also increased but within stochastic 
# changing blocks from 4 to 3 



# CHANGE IN DEPTH BUT NOT IN WIDTH : 
# increase the width change blocks from 2 layers of 32 to 4 layers in each block -> significant increase in runtime 20  mins twice as much with minor increase in accuracy around the same as 256 layers 
# 12 passes highest accuracy is 93.27 




# CHANGE IN REGULARISER 
# adding activity kernel and bias regularizer with l1 and l2 values as 0.1 taking more time than usual gave same amount of accuracy but double the time in 21 minutes converse to around 10 
# increasing the l1 and l2 values to 0.1 reduction in accuracy of 90%
# changing norms to l1 with 1e-5 as values display no collinarity and norms perform better with smaller sizes-> gave same accuracy results (~93) but since there is no collinearity between data -> we do not need regularisation here
# removing the bias regulariser time complexity is still >10 mins accuracy has marginal improvements to above 93.210 same as before 
# removing kernel - same results 
# removing activation also same results also runtime increase is mostly due to the size of an individual block increasing 
