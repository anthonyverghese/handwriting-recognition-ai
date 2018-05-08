#knnimport numpy
#from keras.datasets import mnist
#from keras.models import Sequential
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from keras.layers import Dense
#from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time
from matplotlib import pyplot as plt


def svd_pca(data, k):
    """Reduce DATA using its K principal components."""
    data = data.astype("float64")
    data -= np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data, full_matrices=False)
    return U[:,:k].dot(np.diag(S)[:k,:k])

# define baseline model
def knn_model(x_train,y_train,x_test,y_test,n_neighbors,pca_comp):
    '''
    KNN with PCA
    '''
    print("start model, ",str(n_neighbors), " neighbors, ", str(pca_comp), " components")
    #print("start baseline")
    # create model
    pca = PCA(n_components=pca_comp)
    x_train = pca.fit_transform(x_train)
    x_test = pca.fit_transform(x_test)
    model = KNeighborsClassifier(n_neighbors = n_neighbors)
    #print("start fit")
    #model = RandomForestClassifier()
    model.fit(x_train,y_train)
    #print("end fit")
    start = time.time()
    pred = model.predict(x_test)
    end = time.time()
    seconds = end-start
    minutes = seconds/60.
    print(str(minutes)+" minutes to predict")
    print("end model")
    return accuracy_score(y_test,pred)

def load_data(train_path,test_path):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    print("before reading csv")
    train_db = pd.read_csv(train_path)
    test_db  = pd.read_csv(test_path)
    print("after reading csv")

    y_train = train_db.iloc[:,0]
    num_pixels = 784
    num_classes = 47
    y_train = np_utils.to_categorical(y_train, num_classes)

    x_train = train_db.iloc[:,1:]
    x_train = x_train.astype('float32')
    x_train /= 255

    x_train.head()

    y_test = test_db.iloc[:,0]
    y_test = np_utils.to_categorical(y_test, num_classes)

    x_test = test_db.iloc[:,1:]
    x_test = x_test.astype('float32')
    x_test /= 255
    return (x_train,y_train,x_test,y_test)

def main():
    train_path = "emnist-letters-train.csv"
    test_path = "emnist-letters-test.csv"
    (x_train,y_train,x_test,y_test)=load_data(train_path,test_path)

    components = [50, 75, 100, 125, 150, 175]
    neighbors = [5, 6, 7, 8, 9, 10, 11]

    scores = np.zeros( (components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ) )
    for component in components:
        for n in neighbors:
            score = knn_model(x_train,y_train,x_test,y_test,n,component)
            scores[component][n] = score
            print('Components = ', component, ', neighbors = ', n,', Score = ', score)

'''
    pca = PCA(200)
    print(x_train.shape)
    pca_full = pca.fit(x_train)

    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
'''

if __name__ == '__main__':
    main()


'''
# build the model
model = baseline_model()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=200, verbose=2)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model_json = model.to_json()
with open("small_model_char.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("small_model_char.h5")
'''
