import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st 
from tensorflow.keras import regularizers
from keras.layers import Dropout

@st.cache_data()
def ann() : 

  # Load and prepare the MNIST dataset
  mnist = tf.keras.datasets.mnist

  # Split dataset in data of Train and Data od Test
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # The pixel values of the images range from __0__ through __255__.
  x_train, x_test = x_train / 255.0, x_test / 255.0

  model = tf.keras.models.Sequential([

    # Input Layer
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      # Hidden Layer
      tf.keras.layers.Dense(512, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.088, seed=None)),
      tf.keras.layers.BatchNormalization(),#regularización y reducir el sobreajuste del modelo
      tf.keras.layers.Dropout(0.1),#regularización y reducir el sobreajuste del modelo

      tf.keras.layers.Dense(256, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.088, seed=None)),
      tf.keras.layers.BatchNormalization(),#regularización y reducir el sobreajuste del modelo
      tf.keras.layers.Dropout(0.1),#regularización y reducir el sobreajuste del modelo

      tf.keras.layers.Dense(128, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.088, seed=None)),
      tf.keras.layers.BatchNormalization(),#regularización y reducir el sobreajuste del modelo
      tf.keras.layers.Dropout(0.2),#regularización y reducir el sobreajuste del modelo
      
     # Outoput Layer
      tf.keras.layers.Dense(10,activation='softmax'), #Esta capa tiene 10 neuronas, clasificación de 10 categorías
  ])


  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])

  model.fit(x_train, y_train, epochs=10 )

  return model




