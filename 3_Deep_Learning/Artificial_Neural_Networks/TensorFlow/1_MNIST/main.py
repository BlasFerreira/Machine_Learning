import streamlit as st
from grafic import *
from model_trained import *
import tensorflow as tf
import numpy as np


# if __name__=='__main__':

model_ann = ann()

st.title('Ingrese un numero ')
sample_img = grafic() 

# Upload dataset to try
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



if np.all( np.reshape(sample_img, (28, 28)) == 0):

	st.title("Dibuje algo en el lienzo. ")
else :
	# Show result
	predictions = model_ann(sample_img).numpy()
	arr = tf.nn.softmax(predictions).numpy()[0]
	dict_arr = dict(zip(range(len(arr)), arr.tolist()))
	d_ordenado = dict(sorted(dict_arr.items(), key=lambda x: x[1], reverse=True))

	st.title( f' { round( list(d_ordenado.values())[0]*100,2) } % de ser {list(d_ordenado.keys())[0]} '  ) 

	# for key, value in d_ordenado.items():

	# 	st.write( 'El numero que ingresaste tiene un ' , value*100 , 'de ser' ,key )

	# st.write(np.reshape(sample_img, (28, 28)) )

	# # Save in doc
	# imagen = Image.fromarray(np.uint8( np.reshape(sample_img *255,(28,28)) ))
	# imagen.save("lienzoIMG.png")

	im = Image.fromarray(np.uint8( np.reshape( x_train[17:18] *255,(28,28)) ))
	im.save("mnist_data.png")


	# xaux=x_test
	# yauc=y_test
	# for n in range(0,20)  :
	#     predictions = model_ann(xaux[n:n+1]).numpy()
	#     arr = tf.nn.softmax(predictions).numpy()[0]
	#     dict_arr = dict(zip(range(len(arr)), arr.tolist()))
	#     d_ordenado = dict(sorted(dict_arr.items(), key=lambda x: x[1], reverse=True))

	#     st.write('<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	#     for key, value in d_ordenado.items():
	#         st.write( "El valor real  [", yauc[n], '] predicho :[',key, "]on", value*100,'probabilidad','\n')
	#     st.write(np.reshape(xaux[n:n+1], (28, 28)) )


