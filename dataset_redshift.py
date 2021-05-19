import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def dataset_redshift(target):

  ##### Target #######
  # Les cibles peuvent être : 'Flag' , 'Success' , 'DeltaZ'

	pdf_zgrid=np.load('vipers_ml_export.np_pdf_zgrid.npy')
	attributes=np.load('vipers_ml_export.np_attributes.npy',allow_pickle=True)
	pdfs=np.load('vipers_ml_export.np_pdfs.npy',allow_pickle=True)


	if target == 'Flag':
		y=np.floor(attributes[:,-1])
		y=np.where(y==9, 0, y) 


	elif target == 'Success':
		y=attributes[:,5]

	elif target == 'DeltaZ':
		y=attributes[:,3]

	else:
		raise ValueError("Choisis la target = 'Flag' , 'Success' , 'DeltaZ' ")


	nb_classes=len(np.unique(y))

	X_train, X_test, target_train, target_test = train_test_split(pdfs, y, test_size=0.33)


	Y_train =  np_utils.to_categorical(target_train, nb_classes)   #convertir en one-hot-code
	Y_test = np_utils.to_categorical(target_test, nb_classes)


	return X_train, X_test, target_train, target_test , Y_train , Y_test , nb_classes

