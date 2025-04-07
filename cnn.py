import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import streamlit as st
from PIL import Image

# Step 1: Load and Preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data: normalize and reshape
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

# Step 2: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the CNN model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Step 5: Save the trained model
model.save('mnist_cnn_model.h5')

# Step 6: Streamlit App for Real-time Prediction
def classify_image(uploaded_file):
    # Open image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    return predicted_digit, image

# Streamlit App Setup
st.title("Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

# Image upload feature
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Predict the digit
    predicted_digit, image = classify_image(uploaded_file)

    # Show the result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Digit: {predicted_digit}")
