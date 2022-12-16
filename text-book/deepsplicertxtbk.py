# +-------------------------+-----------------------------+
# Written By   : Akpokiro Victor.
# +-------------------------+-----------------------------+
# Filename     : deepsplicertxtbk.py
# +-------------------------+-----------------------------+
# Description  : DeepSplicer: An Improved Method of 
#			   Splice Sites Prediction using Deep Learning.
# +-------------------------+-----------------------------+
# Reserach Lab :  OluwadareLab, 2021
# +-------------------------+-----------------------------+
# This also contain piece of code from:
# Wang, R. (2013) SpliceFinder source code [Source code]. 
# https://github.com/deepomicslab/SpliceFinder/blob/master/SpliceFinder_sourcecode/CNN_model.py
# +-------------------------+-----------------------------+



from __future__ import print_function
import numpy as np
import time

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf

from sklearn import tree, metrics
from sklearn.metrics import precision_score, recall_score, classification_report, roc_auc_score

import datetime
import os
import argparse

Length = 400  # length of window
dimensions = 1


def load_data(dataset, label):

	labels = np.loadtxt(label)
	encoded_seq = np.loadtxt(dataset)
	
	encoded_seq_choose = encoded_seq[:, ((400-Length)*2):(1600-(400-Length)*2)]

	x_train,x_test,y_train,y_test = train_test_split(encoded_seq_choose,labels,test_size=0.3)

	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)




def deep_cnn_classifier():
	# build the model
	model = Sequential()

	
	if dimensions==1:
		layer = Conv1D(filters=50, 
			kernel_size=9, 
			strides=1, 
			padding='same', 
			batch_input_shape=(None, Length, 4), 
			activation='relu',
			)
		model.add(layer)

		layer = AveragePooling1D(pool_size=2, strides=1, padding='same')
		model.add(layer)


		layer = Conv1D(filters=50, 
			kernel_size=9, 
			strides=1, 
			padding='same', 
			batch_input_shape=(None, Length, 4), 
			activation='relu',
			)
		model.add(layer)

		layer = AveragePooling1D(pool_size=2, strides=1, padding='same')
		model.add(layer)
		
		layer = Conv1D(filters=50, 
			kernel_size=9, 
			strides=1, 
			padding='same', 
			batch_input_shape=(None, Length, 4), 
			activation='relu',
			)
		model.add(layer)

		layer = AveragePooling1D(pool_size=2, strides=1, padding='same')
		model.add(layer)


	else:
		assert False
	model.add(Flatten())
	model.add(Dense(100, activation ='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2, activation ='softmax'))

	# training the model
	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	return model



def training_process(x_train,y_train,x_test,y_test, datatype=''):
	x_train = x_train.reshape(-1, Length, 4)
	y_train = to_categorical(y_train, num_classes=2)
	x_test = x_test.reshape(-1, Length, 4)
	y_test = to_categorical(y_test, num_classes=2)

	epoch = 40
	print("======================")
	print('Convolution Neural Network')

	start_time = time.time()
	model = deep_cnn_classifier()

	history = model.fit(x_train, y_train, epochs=epoch, batch_size=50, shuffle=True)




	loss,accuracy = model.evaluate(x_test,y_test)
	model_data = model.save(f'./models/CNN_{accuracy}_{datatype}.h5')
	print(model.summary())
	keras.utils.plot_model(model, f"./plots/model_summary_{datatype}.png", show_shapes=True)


	print('testing accuracy_{}: {}'.format(datatype, accuracy))
	print('testing loss_{}: {}'.format(datatype, loss))
	print('training took %fs'%(time.time()-start_time))




	plt.plot(history.history['accuracy'])
	plt.title('model accuracy')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['test'], loc='upper left')
	plt.savefig(f"./plots/accuracy_{datatype}.png")


	prob = model.predict(x_test)
	predict = model.predict(x_test)
	predict = to_categorical(predict, num_classes=2)
	y_true = y_test


	auc = roc_auc_score(y_true, model.predict_proba(x_test), multi_class='ovr')

	predicted = np.argmax(prob, axis=1)
	report = classification_report(np.argmax(y_true, axis=1), predicted, output_dict=True )
	print(report)

	macro_precision =  report['macro avg']['precision'] 
	macro_recall = report['macro avg']['recall']    
	macro_f1 = report['macro avg']['f1-score']
	class_accuracy = report['accuracy']

	data_metrics = {"auc_score": auc, "precision": macro_precision, "recall": macro_recall, "f1": macro_f1, "class_accuracy": class_accuracy, "accuracy": accuracy}
	print(data_metrics)

	with open(f'./logs/file_metrics_{datatype}.txt', 'w') as fl:
		fl.write(str(data_metrics))

	return





def app_init():

	parser = argparse.ArgumentParser()

	parser.add_argument("-n", "--name", type=str, required=True, help="name of convolutional model")
	parser.add_argument("-s", "--sequence", type=str, required=True, help="acceptor or donor sequence data")
	parser.add_argument("-o", "--organism", type=str, required=True, help="dataset organism")
	parser.add_argument("-g", "--encoded_seq", str=str, metavar='FILE', required=True, help="one-hot encoded genome sequence data file")
	parser.add_argument("-l", "--label", str=str, metavar='FILE', required=True, help="encoded label data")

	args = parser.parse_args()
	name = args.name
	seq = args.sequence
	org = args.organism
	file_encoded_seq = args.encoded_seq 
	file_label = args.label

	x_train,y_train,x_test,y_test = load_data(file_encoded_seq, file_label)
	training_process(x_train,y_train,x_test,y_test, datatype=seq+name+org)




if __name__ == '__main__':
	app_init()
