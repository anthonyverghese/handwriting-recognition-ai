import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import cv2
import os


json_file = open('small_model_char.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("small_model_char.h5")
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


num_correct = 0
total_count = 0

for filename in os.listdir('chars'):
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", \
    "s", "t", "u", "v", "w", "x", "y", "z"]
    if filename.endswith(".png"):
        total_count += 1
        img_pred = cv2.imread("chars/" + filename, 0);
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
        if (letters[pred[0] - 1]) == filename[0:1]:
            result = "CORRECT"
            num_correct += 1
        print ("Filename: " + filename + " is a " + letters[pred[0] - 1] + " " + result)

print (str(num_correct) + " out of " + str(total_count) + " correct.")
