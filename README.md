# MNIST Handwritten Digit Classification with CNN

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The application is built using **TensorFlow** and **Streamlit**, allowing users to upload or draw digits and receive predictions.

### 🌍 Live Demo
🔗 **Try it here:** [Live Streamlit App](https://digit-cnn-classification.streamlit.app/)

## 🎯 Goal
- Train a **first CNN model** to classify digits from `0` to `9`.
- Build an interactive **Streamlit** app for user testing.
- Provide a graphical visualization of **top 10 predictions** with confidence scores.

## 📚 Required Libraries
Before running the project, install the dependencies:
```bash
pip install streamlit tensorflow numpy matplotlib Pillow streamlit-drawable-canvas
```
- **TensorFlow** → Train and load the CNN model.
- **NumPy** → Handle array operations.
- **Matplotlib** → Visualize predictions and confidence scores.
- **Pillow (PIL)** → Image processing.
- **Streamlit** → Create an interactive web app.
- **Streamlit-Drawable-Canvas** → Allow users to draw digits.

## 📦 Dataset
- **MNIST Dataset** from `keras.datasets.mnist`
- Contains **60,000 training images** and **10,000 test images** (28x28 grayscale pixels).

## 🏗️ Model Architecture (First CNN)
The CNN consists of:
1. **Input Layer** → Accepts 28x28 grayscale images.
2. **8 Convolutional Filters (3x3, ReLU Activation)** → Extracts basic patterns.
3. **MaxPooling (2x2)** → Reduces spatial dimensions.
4. **Dropout (20%)** → Prevents overfitting.
5. **16 Convolutional Filters (3x3, ReLU Activation)** → Extracts deeper features.
6. **MaxPooling (2x2) + Dropout (20%)** → Reduces feature size and overfitting.
7. **Flatten Layer** → Converts to a fully connected layer.
8. **Dense Layer (100 neurons, ReLU)** → High-level abstraction.
9. **Dropout (50%)** → Helps regularization.
10. **Dense Layer (10 neurons, Softmax Activation)** → Outputs probabilities for 10 classes.

```python
import tensorflow as tf
from tensorflow import keras

# Define the CNN Model
model = keras.models.Sequential()

# First layer (Input)
model.add(keras.layers.Input((28,28,1)))  # 28x28 pixel, 1 grayscale channel

# 8 Convolutional Layer
model.add(keras.layers.Conv2D(8, (3,3), activation='relu'))  # 8 filters, 3x3 kernel

# Pooling Layer
model.add(keras.layers.MaxPooling2D((2,2)))  # 2x2 pooling

# Dropout Layer (20%)
model.add(keras.layers.Dropout(0.2))

# 16 Convolutional Layer
model.add(keras.layers.Conv2D(16, (3,3), activation='relu'))  # 16 filters, 3x3 kernel
model.add(keras.layers.MaxPooling2D((2,2)))  # 2x2 pooling
model.add(keras.layers.Dropout(0.2))

# Fully Connected Layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))  # 50% dropout for regularization

# Output Layer (10 Classes)
model.add(keras.layers.Dense(10, activation='softmax'))  # Softmax for multi-class classification

# Compile Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## 🎨 Streamlit Web Application
Once the model is trained and saved as `mnist_cnn_model.h5`, we create an interactive **Streamlit app** where users can:
✅ Upload an image to predict.  
✅ Draw a digit for real-time classification.  
✅ See a **graph of top 10 predictions** with confidence scores.  

**To launch the Streamlit app:**
```bash
streamlit run app.py
```

## 🔥 Features
- **Random MNIST image testing** 
- **Upload custom image for classification** 
- **Draw a digit on a canvas for real-time prediction** 
- **Top-10 predictions bar chart** 
- **CNN model summary display** 🏗
