import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import metrics
import matplotlib.pyplot as plt




def MLP_redshift(dim):

	model = Sequential()
	model.add(Dense(512, input_dim=dim, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(5, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
              optimizer='Adamax',
              metrics=['acc'])

	return model