from sklearn import datasets
import pandas as pd
import keras

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from sklearn.preprocessing import normalize
import numpy as np


def load_data():
    return datasets.load_iris()


def split_data(data):
    pass


def normalize(X_data):
    return normalize(data, axis=0)


def hot_encode(Y_data):
    return np_utils.to_categorical(Y_data)


def model():
    model = Sequential()
    model.add(Dense(1000,input_dim=4,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(300,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def train(data):
    pass

def predict():
    pass


if __name__ == '__main__':
    print("Hello World")

    iris_data = load_data()
    print(iris_data.describe())
    print("********************************************************")
    print(iris_data.columns)
    print("********************************************************")

    X_data = iris_data.iloc[:, 1:5].values
    Y_data = iris_data.iloc[:, 5].values



    transformed_X_data = normalize(X_data)
    transofrmed_Y_data = hot_encode(Y_data)

    model = model()
    model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)

    prediction=model.predict(X_test)

