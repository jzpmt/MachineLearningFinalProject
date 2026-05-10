#FinalProject.py
#Library/Tools
import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

#Model File (to stop from retraining the model every run)
#model_file = "CNN_model_MNIST_dataset.h5"

#Training with CNN

def training_model():
  #MNIST dataset/dataset handling
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train / 255.0
  x_test = x_test / 255.0


  x_train = x_train.reshape(60000, 28, 28, 1)
  x_test = x_test.reshape(10000, 28, 28, 1)

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  #CNN model
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(64,(3,3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

  #Train model
  model.fit(x_train, y_train, epochs=10, batch_size=64,validation_split=0.1)

  #Evaluating the model
  loss, accuracy = model.evaluate(x_test, y_test)
  print("Test Loss: ", loss)
  print("Test Accuracy: ", accuracy)

  #save the model
  #model.save(model_file)
  return model


#Preprocessing the Uploaded Image
def preprocessing_image(image):
  #grayscale the image
  image = image.convert("L")
  
  #resize image
  image = image.resize((28, 28))

  image_array = np.array(image)
  image_array = image_array / 255.0
  image_array = image_array.reshape(1, 28, 28, 1)
  return image_array

#Load Model or Train a New Model
model = training_model()

#Streamlit GUI
st.title("Handwritten Digit Recognition System Using CNN, MNIST, and GUI")
st.write("Upload the handwritten digit image and the CNN will predict the digit")

#uploading image
upload_file = st.file_uploader("Choose image file: ", type = ["png", "jpg", "jpeg"])

#predicting uploaded image
if upload_file is not None:
  image = Image.open(upload_file)
  st.image(image,caption="Uploaded image", width = 200)
  processed_image = preprocessing_image(image)
  prediction = model.predict(processed_image)
  predict_digit = np.argmax(prediction)

  #output
  st.subheader("Predicted Digit is: ")
  st.write(predict_digit)