

# Test different window length
from __future__ import print_function
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Add, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from keras.engine.topology import Layer
import tensorflow as tf
# from d2l import tensorflow as d2l
from sklearn import tree, metrics
from sklearn.metrics import precision_score, recall_score, classification_report
 
Length = 400  # length of window
dimensions = 1




def load_data():

	labels = np.loadtxt('lbl_CE_DonorData.dat')
	encoded_seq = np.loadtxt('CE_DonorData.dat')

	if dimensions == 1:
		encoded_seq = encoded_seq.reshape(-1, 4, 400)
	elif dimensions == 2:
		encoded_seq = encoded_seq.reshape(-1, 4, 20, 20)
	else:
		assert False

	x_train,x_test,y_train,y_test = train_test_split(encoded_seq, labels, test_size=0.3)

	print(labels.shape)
	print(encoded_seq.shape)


	return x_train,y_train,x_test,y_test


def deep_cnn_classifier():
	# build the model
	model = Sequential()

	
	if dimensions==1:
		layer = Conv1D(filters=50, 
			kernel_size=9, 
			strides=1, 
			padding='same', 
			batch_input_shape=(None, 4, Length), 
			activation='relu',
			)
		model.add(layer)

		layer = MaxPooling1D(pool_size=2, strides=1)
		model.add(layer)

	

		layer = Conv1D(filters=50, 
			kernel_size=9, 
			strides=1, 
			padding='same', 
			batch_input_shape=(None, 4, Length), 
			activation='relu',
			)
		model.add(layer)

		layer = MaxPooling1D(pool_size=2, strides=1)
		model.add(layer)



		layer = Conv1D(filters=50, 
		  kernel_size=10, 
		  strides=5, 
		  padding='same', 
		  batch_input_shape=(None, 4, Length), 
			activation='relu',
			)
		model.add(layer)

		layer = MaxPooling1D(pool_size=2, strides=1)
		model.add(layer)

	elif dimensions==2:

		layer = Conv2D(filters=16, 
			kernel_size=16, 
			strides=(1,1), 
			padding='same', 
			batch_input_shape=(None, 4, 20, 20), 
			activation='relu',
			)
		model.add(layer)

		layer = MaxPooling2D(pool_size=2, strides=(1,1))
		model.add(layer)

		layer = Conv2D(filters=32, 
			kernel_size=16, 
			strides=(1,1), 
			padding='same', 
			batch_input_shape=(None, 4, 20, 20), 
			activation='relu',
			)
		model.add(layer)

		layer = MaxPooling2D(pool_size=2, strides=(1,1))
		model.add(layer)

		
	else:
		assert False


	model.add(Flatten())
	model.add(Dense(100, activation ='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(3, activation ='softmax'))

	# training the model
	model.compile(loss = keras.losses.categorical_crossentropy,
				optimizer = keras.optimizers.SGD(learning_rate = 0.01),
				metrics =['accuracy'])

	return model






kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
cvscores = []
cvloss = []

def deep_computation(x_train,y_train,x_test,y_test, datatype=''):

	y_train = to_categorical(y_train, num_classes=3)
	y_test = to_categorical(y_test, num_classes=3)

	X = x_train
	Y = y_train
	Y = np.argmax(Y, axis=1)
	start_time = time.time()
	history = None
	

	model = deep_cnn_classifier()

	for train, test in kfold.split(X, Y):

		epoch = 40
		print("======================")
		print('Convolution Neural Network')
		x_plot = list(range(1,epoch+1))
		
		
		history = model.fit(x_train, y_train, epochs=epoch, batch_size=64, shuffle=True)


		score = model.evaluate(x_test,y_test)
		model.save_weights(f"./models/weight{datatype}.h5")
		


		print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
		print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
		cvscores.append(score[1] * 100)
		cvloss.append(score[0]*100)
	accuracy = np.mean(cvscores)
	loss = np.mean(cvloss)
	print("%.2f%% (+/- %.2f%%)" % (accuracy, np.std(cvscores)))
	print("%.2f%% (+/- %.2f%%)" % (loss, np.std(cvloss)))


	model_data = model.save(f'./models/CNN_{datatype}.h5')
	

	print(model.summary())
	keras.utils.plot_model(model, f"./plots/model_summary_{datatype}.png", show_shapes=True)


	print('testing accuracy: {}'.format(accuracy))
	print('testing loss: {}'.format(loss))
	print('training took %fs'%(time.time()-start_time))


	plt.plot(history.history['accuracy'])
	plt.title('model accuracy')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	# plt.show()
	plt.savefig(f"./plots/accuracy_{datatype}.png")



	prob = model.predict(x_test)
	predict = model.predict(x_test)
	predict = to_categorical(predict, num_classes=3)
	y_true = y_test



	predicted = np.argmax(prob, axis=1)
	report = classification_report(np.argmax(y_true, axis=1), predicted, output_dict=True )
	print(report)

	macro_precision =  report['macro avg']['precision'] 
	macro_recall = report['macro avg']['recall']    
	macro_f1 = report['macro avg']['f1-score']
	class_accuracy = report['accuracy']

	data_metrics = dict({"precision: ", macro_precision, "recall: ", macro_recall, "f1: ", macro_f1, "accuracy: ", class_accuracy})
	print(data_metrics)

	with open(f'./metrics/file_metrics_{datatype}', 'w') as fl:
		fl.write(str(data_metrics))
		


def main(dataname):
	x_train,y_train,x_test,y_test = load_data()
	deep_computation(x_train,y_train,x_test,y_test,datatype=dataname)



if __name__ == '__main__':
	dataname = c_elegans
	main(dataname) 


