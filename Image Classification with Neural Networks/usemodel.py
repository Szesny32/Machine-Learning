import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

class_names=['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']



model = models.load_model('image_classifier.model')


img = cv.imread('plane.jpg')

# load img with opencv they're in BGR
# we worked with RGB color scheme
# convert color scheme

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

#probability distribution of all neruons (all options)
predictions = model.predict(np.array([img]) / 255)

#argmax give index of maximum value
index = np.argmax(predictions)
sorted_indexes = np.argsort(predictions[0])[::-1]

plt.xticks([]) # usuwa przedziały z osi x
plt.yticks([]) # usuwa przedziały z osi y
if predictions[0][index] > 0.6 :
    plt.xlabel(f'Prediction is {class_names[index]}. p = {predictions[0][index]}')
else:
    plt.xlabel(f'Prediction is {class_names[sorted_indexes[0]]}. p = {predictions[0][sorted_indexes[0]]} \n or is {class_names[sorted_indexes[1]]}. p = {predictions[0][sorted_indexes[1]]}')
plt.show()




img = cv.imread('car.jpg')

# load img with opencv they're in BGR
# we worked with RGB color scheme
# convert color scheme

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

#probability distribution of all neruons (all options)
predictions = model.predict(np.array([img]) / 255)

#argmax give index of maximum value
index = np.argmax(predictions)
sorted_indexes = np.argsort(predictions[0])[::-1]

plt.xticks([]) # usuwa przedziały z osi x
plt.yticks([]) # usuwa przedziały z osi y
if predictions[0][index] > 0.6 :
    plt.xlabel(f'Prediction is {class_names[index]}. p = {predictions[0][index]}')
else:
    plt.xlabel(f'Prediction is {class_names[sorted_indexes[0]]}. p = {predictions[0][sorted_indexes[0]]} \n or is {class_names[sorted_indexes[1]]}. p = {predictions[0][sorted_indexes[1]]}')
plt.show()





img = cv.imread('horse.jpg')

# load img with opencv they're in BGR
# we worked with RGB color scheme
# convert color scheme

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

#probability distribution of all neruons (all options)
predictions = model.predict(np.array([img]) / 255)

#argmax give index of maximum value
index = np.argmax(predictions)
sorted_indexes = np.argsort(predictions[0])[::-1]

plt.xticks([]) # usuwa przedziały z osi x
plt.yticks([]) # usuwa przedziały z osi y
if predictions[0][index] > 0.6 :
    plt.xlabel(f'Prediction is {class_names[index]}. p = {predictions[0][index]}')
else:
    plt.xlabel(f'Prediction is {class_names[sorted_indexes[0]]}. p = {predictions[0][sorted_indexes[0]]} \n or is {class_names[sorted_indexes[1]]}. p = {predictions[0][sorted_indexes[1]]}')
plt.show()




img = cv.imread('deer.jpg')

# load img with opencv they're in BGR
# we worked with RGB color scheme
# convert color scheme

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

#probability distribution of all neruons (all options)
predictions = model.predict(np.array([img]) / 255)

#argmax give index of maximum value
index = np.argmax(predictions)
sorted_indexes = np.argsort(predictions[0])[::-1]

plt.xticks([]) # usuwa przedziały z osi x
plt.yticks([]) # usuwa przedziały z osi y
if predictions[0][index] > 0.7 :
    plt.xlabel(f'Prediction is {class_names[index]}. p = {predictions[0][index]}')
else:
    plt.xlabel(f'Prediction is {class_names[sorted_indexes[0]]}. p = {predictions[0][sorted_indexes[0]]} \n or is {class_names[sorted_indexes[1]]}. p = {predictions[0][sorted_indexes[1]]}')
plt.show()
