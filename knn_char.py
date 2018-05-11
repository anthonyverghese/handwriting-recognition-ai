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
from sklearn.model_selection import train_test_split
from sklearn import svm



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

    x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_train, y_train, test_size=0.2, random_state=13)

    model = KNeighborsClassifier(n_neighbors = n_neighbors)
    #print("start fit")
    #model = RandomForestClassifier()
    model.fit(x_train_pca,y_train_pca)
    #print("end fit")
    start = time.time()
    pred = model.predict(x_test_pca)
    end = time.time()
    seconds = end-start
    minutes = seconds/60.
    print(str(minutes)+" minutes to predict")
    print("end model")
    return accuracy_score(y_test_pca,pred)

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

def test_neigh_and_comp(x_train,y_train,x_test,y_test):
    '''
    '''
    components = [30]
    neighbors = [5, 10, 15, 20, 25, 30, 35,40,45,50,55,60,65,70,75,80,85]
    scores = np.zeros( (components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ) )
    for n in neighbors:
        for component in components:
            score = knn_model(x_train,y_train,x_test,y_test,n,component)
            scores[component][n] = score
            print('Components = ', component, ', neighbors = ', n,', Score = ', score)

def my_svm(x_train,y_train,x_test,y_test,pca_comp):
    '''
    '''
    start = time.time()
    pca = PCA(n_components=pca_comp)

    #x_train = pca.fit_transform(x_train)
    #x_test = pca.fit_transform(x_test)

    classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print("fit classifier")
    classifier.fit(x_train, y_train)
    print("make predictions")
    pred = classifier.predict(x_test)
    end = time.time()
    seconds = end-start
    minutes = seconds/60.
    print(str(minutes), " minutes to run svm")
    return accuracy_score(y_test,pred)



def main():
    train_path = "emnist-digits-train.csv"
    test_path = "emnist-digits-test.csv"
    (x_train,y_train,x_test,y_test)=load_data(train_path,test_path)
    #test_neigh_and_comp(x_train,y_train,x_test,y_test)
    print(my_svm(x_train,y_train,x_test,y_test,5))
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

