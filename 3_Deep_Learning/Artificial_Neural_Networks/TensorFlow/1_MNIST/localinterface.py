import streamlit as st
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import io 
import numpy as np

saved_image = []
# Función para dibujar en el lienzo
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

def save():
    # Obtiene la imagen del lienzo y la convierte a blanco y negro
    drawn_image = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(drawn_image.encode('utf-8')))
    img = img.convert('L')
    # Guarda la imagen en el archivo NumeroDibujado.png
    filename = "NumeroDibujado.png"
    img.save(filename)

    # imagen mas pequeñar
    img_resized = img.resize((28, 28))


    # Invertir los valores
    img_inverted = ImageOps.invert(img_resized)

    # Guardar la imagen con el nuevo tamaño
    img_inverted.save("imagen_nueva.png")

    img_array = np.array(img_inverted)

    # Asignar el valor devuelto a la variable global
    global saved_image
    saved_image = img_array


if __name__=='__main__':
    # Crea la ventana principal
    root = Tk()

    # Crea el lienzo para dibujar
    canvas = Canvas(root, width=280, height=280, bg="white")
    canvas.pack()

    # Crea la imagen y el objeto de dibujo
    img = Image.new("RGB", (280, 280), "white")
    draw = ImageDraw.Draw(img)

    # Vincula el evento de dibujo con la función paint
    canvas.bind("<B1-Motion>", paint)


    # Crea el botón para guardar el dibujo
    button = Button(root, text="Guardar", command=save)

    button.pack()

    # Inicia el bucle de eventos
    root.mainloop()
    
    # return saved_image
    print(saved_image)

    st.write('GOOD')





