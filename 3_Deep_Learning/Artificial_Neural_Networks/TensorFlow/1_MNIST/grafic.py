from PIL import Image, ImageOps
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import base64
import io
import cv2

def grafic() : 
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.9)",  # Color de fondo del lienzo
        stroke_width=23,  # Ancho del trazo
        stroke_color="Black",  # Color del trazo
        background_color="#FFF",  # Color de fondo de la aplicación
        height=280,  # Altura del lienzo en píxeles
        width=280,  # Ancho del lienzo en píxeles
        drawing_mode="freedraw",  # Modo de dibujo
        key="canvas" 
        )

    if canvas_result.image_data is not None:
        # st.image(canvas_result.image_data/255)
        pic = Image.fromarray(canvas_result.image_data, 'RGBA').convert('L')

        img_resized = pic.resize((28, 28))

        img_inverted = ImageOps.invert(img_resized)

        img_array = np.array(img_inverted)
        img_array = np.reshape(img_array, (1, 28, 28))

        return img_array/255

