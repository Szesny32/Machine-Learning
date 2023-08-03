import math
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, datasets, losses


fashion_mnist = datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data normalization - pixel values ranging from 0 to 255
train_images = train_images / 255.0
test_images = test_images / 255.0



def train(index, data):
    index-=1
    if index < 10:
        train_L1(index, data)
        index = 10

    if index < 110:
        train_L2(index, data)
        index = 110

    train_L3(index, data)
   


def train_L1(index, data):
    for k in range(index, 10):
                avg_loss = [0] * 10
                avg_accuracy = [0] * 10
                L1 = pow(2, k+1)
                layers = [L1]
                print(f"----------------------Traing mk.{index+1} = {layers}----------------------")
                for i in range(10):
                    model = build_L1_model(L1)
                    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                    for e in range(10):
                        print(f"mk.{index+1} | iteration = {i+1} | epoch = {e+1}")
                        model.fit(train_images, train_labels, epochs=1)
                        loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
                        print(f"results:\nloss = {loss}\naccuracy = {accuracy}\n")
                        avg_loss[e] += loss
                        avg_accuracy[e] += accuracy
                for e in range(10):    
                    avg_loss[e] /= 10.0
                    avg_accuracy[e] /= 10.0
        
                update_data(data, index+1, avg_accuracy, avg_loss, layers)
                index+=1   

def build_L1_model(L1):
    model = models.Sequential([
        layers.Flatten(input_shape=(train_images.shape[1], train_images.shape[2])),
        layers.Dense(L1, activation='relu'),
        layers.Dense(10)
    ])
    return model




def train_L2(index, data):
    index-=10
    L1_start = index % 10
    L2_start = math.floor(index / 10) 
    for k in range(L1_start , 10):
        avg_loss = [0] * 10
        avg_accuracy = [0] * 10
        L1 = pow(2, k+1)
        L2 = pow(2, L2_start+1)

        layers = [L1, L2]
        print(f"----------------------Traing mk.{index+11} = {layers}----------------------")
        for i in range(10):
            model = build_L2_model(L1, L2)
            model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            for e in range(10):
                print(f"mk.{index+11} | iteration = {i+1} | epoch = {e+1}")
                model.fit(train_images, train_labels, epochs=1)
                loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
                print(f"results:\nloss = {loss}\naccuracy = {accuracy}\n")
                avg_loss[e] += loss
                avg_accuracy[e] += accuracy
        for e in range(10):    
            avg_loss[e] /= 10.0
            avg_accuracy[e] /= 10.0

        update_data(data, index+11, avg_accuracy, avg_loss, layers)
        index += 1

    L2_start  += 1      
    for j in range(L2_start , 10):
        for k in range(0, 10):
            avg_loss = [0] * 10
            avg_accuracy = [0] * 10
            L1 = pow(2, k+1)
            L2 = pow(2, j+1)

            layers = [L1, L2]
            print(f"----------------------Traing mk.{index+11} = {layers}----------------------")
            for i in range(10):
                model = build_L2_model(L1, L2)
                model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                for e in range(10):
                    print(f"mk.{index+11} | iteration = {i+1} | epoch = {e+1}")
                    model.fit(train_images, train_labels, epochs=1)
                    loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
                    print(f"results:\nloss = {loss}\naccuracy = {accuracy}\n")
                    avg_loss[e] += loss
                    avg_accuracy[e] += accuracy
            for e in range(10):    
                avg_loss[e] /= 10.0
                avg_accuracy[e] /= 10.0

            update_data(data, index+11, avg_accuracy, avg_loss, layers)
            index+=1

def build_L2_model(L1, L2):
    model = models.Sequential([
        layers.Flatten(input_shape=(train_images.shape[1], train_images.shape[2])),
        layers.Dense(L1, activation='relu'),
        layers.Dense(L2, activation='relu'),
        layers.Dense(10)
    ])
    return model



def train_L3(index, data):
    index-=100
    L1_start  = index % 10
    L2_start  = math.floor(index / 10) 
    L3_start  = math.floor(index / 100) 
    for k in range(L1_start, 10):
        avg_loss = [0] * 10
        avg_accuracy = [0] * 10
        L1 = pow(2, k+1)
        L2 = pow(2, L2+1)
        L3 = pow(2, L3+1)

        layers = [L1, L2, L3]
        print(f"----------------------Traing mk.{index+111} = {layers}----------------------")
        for i in range(10):
            model = build_L3_model(L1, L2, L3)
            model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            for e in range(10):
                print(f"mk.{index+111} | iteration = {i+1} | epoch = {e+1}")
                model.fit(train_images, train_labels, epochs=1)
                loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
                print(f"results:\nloss = {loss}\naccuracy = {accuracy}\n")
                avg_loss[e] += loss
                avg_accuracy[e] += accuracy
        for e in range(10):    
            avg_loss[e] /= 10.0
            avg_accuracy[e] /= 10.0

        update_data(data, index+111, avg_accuracy, avg_loss, layers)
        index += 1

    L2_start  += 1      
    for j in range(L2_start , 10):
        for k in range(0, 10):
            avg_loss = [0] * 10
            avg_accuracy = [0] * 10
            L1 = pow(2, k+1)
            L2 = pow(2, j+1)
            L3 = pow(2, L3_start +1)

            layers = [L1, L2, L3]
            print(f"----------------------Traing mk.{index+111} = {layers}----------------------")
            for i in range(10):
                model = build_L3_model(L1, L2, L3)
                model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                for e in range(10):
                    print(f"mk.{index+111} | iteration = {i+1} | epoch = {e+1}")
                    model.fit(train_images, train_labels, epochs=1)
                    loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
                    print(f"results:\nloss = {loss}\naccuracy = {accuracy}\n")
                    avg_loss[e] += loss
                    avg_accuracy[e] += accuracy
            for e in range(10):    
                avg_loss[e] /= 10.0
                avg_accuracy[e] /= 10.0

            update_data(data, index+111, avg_accuracy, avg_loss, layers)
            index+=1
    L3_start  += 1      
    for i in range(L3_start , 10):
        for j in range(0, 10):
            for k in range(0, 10):
                avg_loss = [0] * 10
                avg_accuracy = [0] * 10
                L1 = pow(2, k+1)
                L2 = pow(2, j+1)
                L3 = pow(2, i+1)

                layers = [L1, L2, L3]
                print(f"----------------------Traing mk.{index+111} = {layers}----------------------")
                for i in range(10):
                    model = build_L3_model(L1, L2, L3)
                    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                    for e in range(10):
                        print(f"mk.{index+111} | iteration = {i+1} | epoch = {e+1}")
                        model.fit(train_images, train_labels, epochs=1)
                        loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
                        print(f"results:\nloss = {loss}\naccuracy = {accuracy}\n")
                        avg_loss[e] += loss
                        avg_accuracy[e] += accuracy
                for e in range(10):    
                    avg_loss[e] /= 10.0
                    avg_accuracy[e] /= 10.0

                update_data(data, index+111, avg_accuracy, avg_loss, layers)
                index+=1

def build_L3_model(L1, L2, L3):
    model = models.Sequential([
        layers.Flatten(input_shape=(train_images.shape[1], train_images.shape[2])),
        layers.Dense(L1, activation='relu'),
        layers.Dense(L2, activation='relu'),
        layers.Dense(L3, activation='relu'),
        layers.Dense(10)
    ])
    return model


def init_data():
    data = [{   "id": i, 
                "layers": [],
                "accuracy": [], 
                "loss": [] ,
                "date": None
            } for i in range(1, 1111)]
    return data

   
def save_to_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def load_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def update_data(data, id, accuracy, loss, layers):
    data[id - 1]['id'] = id
    data[id - 1]["accuracy"] = accuracy
    data[id - 1]["loss"] = loss
    data[id - 1]["layers"] = layers
    data[id - 1]['date'] = str(datetime.now())
    save_to_json("dane.json", data)



# data = init_data()
# save_to_json("dane.json", data)

data = load_from_json("dane.json")
train(4, data)