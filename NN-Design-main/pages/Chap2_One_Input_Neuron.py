import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from st_pages import Page, show_pages, add_page_title, hide_pages
from constants import pages_created
import base64
import os


# Define transfer functions
def purelin(n):
    return n

def poslin(n):
    return np.maximum(0, n)

def hardlim(n):
    return np.where(n >= 0, 1, 0)

def hardlims(n):
    return np.where(n >= 0, 1, -1)

def satlin(n):
    return np.minimum(np.maximum(0, n), 1)

def satlins(n):
    return np.minimum(np.maximum(-1, n), 1)

def logsig(n):
    return 1 / (1 + np.exp(-n))

def tansig(n):
    return 2 / (1 + np.exp(-2 * n)) - 1

# Mapping of functions to their string representations
functions = {
    "purelin": purelin,
    "poslin": poslin,
    "hardlim": hardlim,
    "hardlims": hardlims,
    "satlin": satlin,
    "satlins": satlins,
    "logsig": logsig,
    "tansig": tansig
}

if __name__ == "__main__":
    def load_svg(svg_file):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg = f.read()
        svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        svg_html = f'''
        <div style="text-align: left; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 100px; margin: 10px;">
        </div>
        '''
        return svg_html

    def load_svg_2(svg_file):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg = f.read()
        svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        svg_html = f'''
        <div style="text-align: center; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 90%; height: 250px; margin: 10px;">
        </div>
        '''
        return svg_html

    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)



    image_path = 'media'

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    hide_pages(pages_created)

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('One-input Neuron')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')
    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/2.svg")), unsafe_allow_html=True)
        st.markdown("Alter the weight, bias and input by moving the sliders.")
        st.markdown("Pick the transfer function with the f menu.")
        st.markdown("Watch the change to the neuron function and its output.")

        st.markdown('Weight (w):', unsafe_allow_html=True)
        weight = st.slider("", min_value=-3.0, max_value=3.0, value=1.0, step=0.1, key="weight")

        st.markdown('Bias (b):', unsafe_allow_html=True)
        bias = st.slider("", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, key="bias")


        st.markdown('Transfer Function (f):', unsafe_allow_html=True)
        selected_function = st.selectbox("", options=list(functions.keys()))
        st.subheader('*Chapter2*')
        st.markdown('---')

    st.markdown(load_svg_2(get_image_path("Figures/SingleInputNeuron.svg")), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(3, 2))
    p = np.arange(-4, 4, 0.1)  # Input range
    func = np.vectorize(functions[selected_function])
    out = func(weight * p + bias)

    ax.plot(p, out, color="red", lw=0.5)
    ax.axhline(0, color='black', lw=0.2)
    ax.axvline(0, color='black', lw=0.2)
    # ax.grid(True, which='both')

    # Set x and y axis limits

    tick_interval = 0.5
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])
    ax.set_xticks(np.arange(-2.0, 2.5, tick_interval))  # Set custom x-ticks
    ax.set_yticks(np.arange(-2.0, 2.5, tick_interval))  # Set custom y-ticks

    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_xlabel("$p$")
    ax.set_ylabel("$a$")
    ax.set_title(f"$a = {selected_function}(w \cdot p + b)$", fontsize=10)

    st.pyplot(fig)

