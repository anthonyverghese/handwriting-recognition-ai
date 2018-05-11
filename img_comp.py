"""
An approach that takes the Mean Squared Error of different images and compares them to the target image. Take the
highest average as the guess. MSE model inspired https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/.
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import os


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


test_file = "digits/0.png"
# for filename in os.listdir("digits"):
#     if filename.endswith(".png"):
#         print(file)
#         print(mse())

# Compare with mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()