

# **Handwritten Digit Classifier Using CNN & Streamlit**

This project uses a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. It also includes a **Streamlit web interface** for real-time predictions, where users can upload an image of a handwritten digit and receive the predicted digit in response.

---

## **Project Overview**

- **Model**: Convolutional Neural Network (CNN)
- **Dataset**: MNIST (handwritten digits 0-9)
- **Frameworks**: TensorFlow/Keras for building the CNN model, Streamlit for building the web interface.
- **Performance**: Achieved **98% accuracy** on the test set.
- **Real-time Prediction**: Users can upload an image and get the predicted digit.

---

## **Features**

- **Upload Image**: Allows the user to upload an image of a handwritten digit.
- **Real-time Prediction**: The CNN model predicts the digit (0-9) from the uploaded image.
- **Streamlit Interface**: Simple and interactive interface to use the model.

---

## **Project Structure**

```
handwritten_digit_classifier/
│
├── app.py                # Streamlit Web Interface
├── mnist_cnn_model.h5    # Pre-trained CNN model
├── requirements.txt      # Required Python libraries
└── README.md             # Project documentation
```

---

## **Installation**

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/handwritten_digit_classifier.git
   cd handwritten_digit_classifier
   ```

2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Training the Model**

Before running the Streamlit app, ensure the model is trained. You can train the model using the following code:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data: normalize and reshape
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

# Build the CNN model
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

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save the model
model.save('mnist_cnn_model.h5')
```

This will train the model and save it as `mnist_cnn_model.h5`.

---

## **Running the Web App**

1. Once the model is saved, run the Streamlit app with the following command:

   ```bash
   streamlit run app.py
   ```

2. The app will open in your browser, where you can upload a handwritten digit image for prediction.

---

## **Streamlit Interface Overview**

When you run the app, the interface will appear like this:

### **Streamlit UI:**

![Streamlit UI](https://via.placeholder.com/700x400?text=Streamlit+Web+App)

- **Upload Image**: A file uploader button allows you to choose an image (JPG format).
- **Prediction**: Once the image is uploaded, the model predicts the digit and displays the result.

---

## **How to Use the App**

1. **Upload an Image**: Click on the "Choose an image" button to upload an image of a handwritten digit (0-9).
   
   Example of an image:
   ![Example Image](https://via.placeholder.com/200x200?text=Example+Handwritten+Digit)

2. **Prediction**: After uploading, the app will display the image and show the predicted digit below it.

   Example of result:
   ![Prediction Result](https://via.placeholder.com/200x200?text=Predicted+Digit%3A+3)

---

## **Model Evaluation**

After training the model, you can evaluate its performance:

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```

- **Test Accuracy**: This model achieved around **98% accuracy** on the MNIST test dataset.

---

## **Requirements**

The following Python libraries are required to run this project:

- **TensorFlow**: Deep learning framework used to train the CNN model.
- **Streamlit**: Framework to build and deploy the web interface.
- **NumPy**: For handling numerical operations.
- **Pillow**: For image handling and preprocessing.

---

## **Deployment (Optional)**

To deploy the Streamlit app online, follow these steps:

1. Push the code to a GitHub repository.
2. Sign up for a free account on [Streamlit Cloud](https://streamlit.io/cloud).
3. Deploy your app by connecting your GitHub repository.

---

## **Conclusion**

You’ve successfully built an **Image Classifier using CNN** to recognize handwritten digits with the MNIST dataset. The project also includes a **Streamlit interface** that allows users to upload an image and get the predicted digit in real time.

Feel free to improve the model by:
- Implementing **image augmentation** to boost performance.
- Experimenting with different **CNN architectures**.
- Adding **more features** to the app.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

#### **Screenshots & Visuals**
- **Web Interface Preview:**
  ![Streamlit Preview](https://via.placeholder.com/700x400?text=Streamlit+Preview)

- **Uploaded Image Example**:
  ![Uploaded Image Example](https://via.placeholder.com/150x150?text=Handwritten+Digit)

- **Prediction Result Example**:
  ![Prediction Result Example](https://via.placeholder.com/150x150?text=Predicted+Digit%3A+5)

---

