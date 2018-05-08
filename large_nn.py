# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import cv2
from keras.models import model_from_json
import os

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

json_file = open('large_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("large_model.h5")
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


num_correct = 0
total_count = 0

for filename in os.listdir('digits'):
	if filename.endswith(".png"):
		total_count += 1
		img_pred = cv2.imread("digits/" + filename, 0);
		img_pred = cv2.bitwise_not(img_pred)
		img_pred = cv2.resize(img_pred, (28, 28))

		cv2.imshow(filename, img_pred)
		cv2.waitKey(200)
		cv2.destroyAllWindows()

		img_pred = img_pred.reshape(1, 784).astype('float32')
		img_pred = img_pred / 255
		pred = model.predict_classes(img_pred)

		pred_proba = model.predict_proba(img_pred)
		pred_proba = "% .2f %%" % (pred_proba[0][pred] * 100)

		result = "incorrect"
		if (str(pred[0]) == filename[:1]):
			result = "CORRECT"
			num_correct += 1
		print ("Filename: " + filename + " is a " + str(pred[0]) + " with probability" + str(pred_proba) + " " + result)

print (str(num_correct) + " out of " + str(total_count) + " correct.")
