import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Supongamos que tienes un array llamado 'array_28x28' de dimensiones 28x28

# Convertir el array en un objeto de imagen
imagen = Image.fromarray(np.uint8(x_train[:1]))

# Guardar la imagen en un archivo
imagen.save("imagen.png")


model_l2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),  # Capa de dropout después de la capa oculta
    keras.layers.Dense(10),
    keras.layers.Dropout(0.2)  # Capa de dropout antes de la capa de salida
])


model_l2.compile(
    optimizer='adam',        
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model_l2.fit(train_images, train_labels, epochs=10)