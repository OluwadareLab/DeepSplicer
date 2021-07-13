import numpy as np
import time
# import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel

Length = 400  # length of window

def load_data():

	labels = np.loadtxt('lbl_CE_DonorData.dat')
	encoded_seq = np.loadtxt('CE_DonorData.dat')

	encoded_seq = encoded_seq.reshape(-1,4,400)
	
	x_train,x_test,y_train,y_test = train_test_split(encoded_seq,labels,test_size=0.3)

	y_train = to_categorical(y_train, num_classes=3)
	y_test = to_categorical(y_test, num_classes=3)

	return x_train,y_train,x_test,y_test


class MyHyperModel(HyperModel):

	def __init__(self):
		pass

	def build(self, hp):
		# build the model
		model = Sequential()


		model.add(Conv1D(filters=hp.Int('filters0', min_value=50, max_value=250, step=50), kernel_size=hp.Int('kernel0', min_value=3, max_value=16, step=1), 
			strides=hp.Int('strides0', min_value=1, max_value=7, step=2), padding='same', 
			batch_input_shape=(None, 4, Length), activation=hp.Choice('activation0', values=['sigmoid', 'relu', 'softmax', 'tanh'])))

		model.add(Conv1D(filters=hp.Int('filters1', min_value=50, max_value=250, step=50), kernel_size=hp.Int('kernel1', min_value=3, max_value=16, step=1), 
			strides=hp.Int('strides1', min_value=1, max_value=7, step=2), padding='same', 
			batch_input_shape=(None, 4, Length), activation=hp.Choice('activation1', values=['sigmoid', 'relu', 'softmax', 'tanh'])))

		model.add(Conv1D(filters=hp.Int('filters2', min_value=50, max_value=250, step=50), kernel_size=hp.Int('kernel2', min_value=3, max_value=16, step=1), 
			strides=hp.Int('strides2', min_value=1, max_value=7, step=2), padding='same', 
			batch_input_shape=(None, 4, Length), activation=hp.Choice('activation2', values=['sigmoid', 'relu', 'softmax', 'tanh'])))

		

		model.add(Flatten())
		model.add(Dense(100, activation ='relu'))
		model.add(Dropout(hp.Float('dropout',min_value=.01, max_value=.05)))
		model.add(Dense(3, activation ='softmax'))

		# training the model
		model.compile(loss = categorical_crossentropy,
					optimizer = SGD(lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
					metrics =['accuracy'],
					)

		return model
	
HYPERBAND_MAX_EPOCHS = 1 # 40 # small for testing
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 3 # 3
N_EPOCH_SEARCH = 40 # 40 # small for testing

def main():
	x_train,y_train,x_test,y_test = load_data()

	# build the model once
	hypermodel = MyHyperModel()

	tuner = RandomSearch(
		hypermodel,
		objective='val_accuracy',
		max_trials=MAX_TRIALS,
		executions_per_trial=EXECUTION_PER_TRIAL,
		directory='./logs',
		project_name='splicesites50filters',
	)

	print('model built...')
	tuner.search(x_train, y_train, 
		batch_size = 50, 
		epochs=N_EPOCH_SEARCH, 
		validation_split=0.1,
		verbose=0,
		)
	print('model trained...')

	summary = tuner.results_summary()
	best_model = tuner.get_best_models()[0].summary
	best_model1 = tuner.get_best_models(num_models=1)
	best_hyperparameters = tuner.get_best_hyperparameters()[0].values

	print(f"summary: {summary}")
	print(f"\n\n best_model: {best_model}")
	print(f"\n\n best_model: {best_model1}")
	print(f"\n\n best_hyperparameter: {best_hyperparameters}")



if __name__ == '__main__':
	main()

