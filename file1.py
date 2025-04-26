from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
 
# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0


# Reshape the input data to (28, 28, 1) because CNN expects a 3D input (height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Create the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # Output layer (10 digits)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
# model.fit(x_train, y_train, epochs=5)

# # Evaluate the model
# model.evaluate(x_test, y_test)


# Save the trained model
# model.save("mnist_model.h5")
# print("Model saved as mnist_model.h5")

# Load the saved model
loaded_model = keras.models.load_model("mnist_model.h5")
print("Model loaded from mnist_model.h5")

# Store model predictions
y_predicted = loaded_model.predict(x_test)

# Display an image with the predicted digit
plt.imshow(x_test[1500].reshape(28,28), cmap='gray')
plt.title(f"Actual digit: {y_test[1500]} | Predicted digit: {np.argmax(y_predicted[1500])}")
plt.show()