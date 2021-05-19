from dataset_redshift import dataset_redshift
from MLP_redshift import MLP_redshift
from CNN1D_redshift import CNN1D_redshift

import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import metrics
import matplotlib.pyplot as plt
from keras.utils import np_utils

X_train, X_test, target_train, target_test , Y_train , Y_test , nb_class = dataset_redshift('Flag')

model = CNN1D_redshift(np.shape(X_train))

h = model.fit(X_train, Y_train,
              epochs=50,
              batch_size=256,
              verbose =1,
              validation_data=(X_test, Y_test))