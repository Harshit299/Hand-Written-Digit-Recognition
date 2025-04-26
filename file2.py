from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# load_data() returns only two tuples
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255


# reshape 2D image(28x28) into 1D vector(784 elements)
x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)

# print(x_train_flat[0]) # 1D array with total 784 elements(one for each pixel in image)
# print(x_train_flat.shape) # (60000, 784)

print(y_test)

# create layers
model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(200, activation = 'relu'),
    keras.layers.Dense(150, activation = 'relu'),
    keras.layers.Dense(180, activation = 'relu'),
    keras.layers.Dense(160, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# train the model
model.fit(x_train_flat, y_train, epochs = 5)

# calculate model performance
model.evaluate(x_test_flat, y_test)

# store model predictions
y_predicted = model.predict(x_test_flat)
print(y_predicted)

# print(np.argmax(y_predicted[0]))
plt.imshow(x_test_flat[9999].reshape(28,28))
plt.title(f"actual digit {y_test[9999]} | predicted digit  {np.argmax(y_predicted[9999])}")
plt.show()