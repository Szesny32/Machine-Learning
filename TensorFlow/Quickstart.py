from tensorflow.keras import datasets, layers, models, losses
from tensorflow.nn import softmax

mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))

predictions = model(x_train[:1]).numpy()
print("Predictions: ", predictions)
print("\nSoftmax: ",softmax(predictions).numpy())

loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
print("\nInitial loss: ", loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)