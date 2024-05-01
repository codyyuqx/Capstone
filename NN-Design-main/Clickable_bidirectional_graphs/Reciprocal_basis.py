import os
import numpy as np
import io
import base64
import streamlit as st 
import matplotlib.pyplot as plt
from src.clickable_graphs import clickable_graphs


st.set_page_config(page_title="Streamlit Image Coordinates", page_icon="🎯", layout="wide")

def load_css(filename):
    with open(filename) as css:
        return st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


css_path = os.path.join(".", "nnd.css")
load_css(css_path)

def get_image_path(filename):
    return os.path.join(".", "Logo", filename)

def load_svg(svg_file):
    with open(svg_file, "r", encoding="utf-8") as f:
        svg = f.read()
        svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        svg_html = f'''
        <div style="text-align: left; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 100px; margin: 20px;">
        </div>
        '''
    return svg_html



def main():
    
    value = clickable_graphs("grams", "onRender3Click")
    vectors = []
    if value is not None:
        
        vector1 = value[0]
        vector2 = value[1]
        
        vectors.append(vector1)
        vectors.append(vector2)

    st.write(vectors)
    

if __name__=="__main__":
    main()