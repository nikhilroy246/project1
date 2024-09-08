# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the input data to match the expected input shape of the model
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert class labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)

# Define the FGSM attack function
def fgsm_attack(model, x, y, epsilon=0.1):
    x_tf = tf.constant(x)  # Convert NumPy array to TensorFlow tensor
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        predictions = model(x_tf)
        loss = tf.keras.losses.categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, x_tf)
    gradients = tf.sign(gradients)
    x_adv = x_tf + epsilon * gradients
    x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    return x_adv.numpy()  # Convert back to NumPy array

# Perform the FGSM attack on the test set
x_test_adv = fgsm_attack(model, x_test, y_test)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Evaluate the model on the adversarial test set
loss_adv, accuracy_adv = model.evaluate(x_test_adv, y_test)
print(f"Test accuracy (adversarial): {accuracy_adv:.4f}")