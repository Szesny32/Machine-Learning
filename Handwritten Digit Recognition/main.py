import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # neural network 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# ordinary feed forward neural network
model = tf.keras.models.Sequential()
#Flatten layer is basically just a one-dimentional layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# dense -> all neurons are connected to the previous layer 
model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 128, activation=tf.nn.relu))

# output
# softmax -> tries to take all the outputs (10) and each neuron has certain activation
# How likely it is for example that the number is or the digit is x

model.add(tf.keras.layers.Dense(units = 10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# epochs = number how many times is the model going to see the 
# data over and over again, how many time repeat thw whole process
model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

#after we've trained the model we're going to scan in out own images
# into the neural network to classify them
model.save('digits.model')

for x in range(1,6):
    img = cv.imread(f'test_samples\{x}.png')[:,:,0]
    # we need black on white because otherwise neural network will be confused
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
