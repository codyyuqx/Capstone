import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from nndesigndemos.get_package_path import PACKAGE_PATH
## front style
def load_svg(svg_file):
    with open(svg_file, "r", encoding="utf-8") as f:
        svg = f.read()
    svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
    # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
    svg_html = f'''
    <div style="text-align: center; width: 100%;">
        <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 150px; margin: 20px;">
    </div>
    '''
    return svg_html

def load_svg2(svg_file):
    with open(svg_file, "r", encoding="utf-8") as f:
        svg = f.read()
    svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
    # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
    svg_html = f'''
    <div style="text-align: center; width: 100%;">
        <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 300px; margin: 20px;">
    </div>
    '''
    return svg_html

# Set the layout to "wide"
st.set_page_config(layout="wide")

# Define the relative path for the images using a raw string
image_path = r"D:\nndesign-demo-master\nndesign-demo-master\nndesigndemos"

st.markdown("""
<style>
.streamlit-container {
    max-width: 95%;
}
.font {
    font-size: 28px !important;
    font-family: 'Times New Roman', Times, serif !important;
}
.header {
    text-align: right;
    font-size: 28px;
    font-family: 'Times New Roman', Times, serif;
}
.title-line {
    display: inline-block; /* or 'inline-block' depending on your layout needs */
    margin-bottom: 5px; /* Adjust the bottom margin to control line spacing */
}
.content-font {
    font-size: 18px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

.blue-line {
    height: 4px;
    background-color: darkblue;
    margin: 0px 0;
}
.vertical-line {
            border-left: 4px solid darkblue;
            height: 800px;  # Adjust the height as needed
        }
.footer {
    text-align: right;
    font-size: 18px;
    font-family: 'Times New Roman', Times, serif;
}

.space-top {
    margin-top: 50px;
}
.selectbox-option {
    font-size: 18px !important;
    font-family: 'Times New Roman', Times, serif !important;
}
.stButton>button {
    font-size: 18px !important;
    font-family: 'Times New Roman', Times, serif !important;
    color: black !important; /* Text color */
    background-color: white !important; /* Background color */
    border: 1px solid black !important; /* Black border color and width */
    border-radius: 0.3rem !important;
    line-height: 1.5 !important;
    width: 100% !important; /* Make buttons use the full width of the column */
    transition: background-color 0.3s !important; /* Smooth transition for background color */
}
.stSlider .rc-slider-handle {
            border: 2px solid darkblue !important;  /* Handle border color */
            background-color: darkblue !important;  /* Handle background color */
        }
        /* Additional styling for the handle on hover and focus, if needed */
        .stSlider .rc-slider-handle:hover,
        .stSlider .rc-slider-handle:focus {
            border-color: darkblue !important;
            box-shadow: none !important;
        }

</style>
""", unsafe_allow_html=True)
def get_image_path(filename):
    # Use a raw string for the path
    return os.path.join(image_path, filename)

# Define transfer functions
def purelin(n): return n
def poslin(n): return np.maximum(0, n)
def hardlim(n): return np.where(n >= 0, 1, 0)
def hardlims(n): return np.where(n >= 0, 1, -1)
def satlin(n): return np.minimum(np.maximum(0, n), 1)
def satlins(n): return np.minimum(np.maximum(-1, n), 1)
def logsig(n): return 1 / (1 + np.exp(-n))
def tansig(n): return np.tanh(n)

# Mapping transfer function names to their implementations
transfer_functions = {
    "purelin": purelin,
    "poslin": poslin,
    "hardlim": hardlim,
    "hardlims": hardlims,
    "satlin": satlin,
    "satlins": satlins,
    "logsig": logsig,
    "tansig": tansig
}





col1, col2,col3= st.columns([10,0.1,3])

with col3:
    # Sliders for weight and bias
    st.markdown(load_svg(get_image_path("Chapters/2/Logo_Ch_2.svg")), unsafe_allow_html=True)
    st.markdown('<p class="content-font">Alter the weight, bias and input by moving the sliders.'
                '<br>'
                'Pick the transfer function with the f menu.'
                '<br>'
                'Watch the change to the neuron function and its output.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content-font">Weight (w):</p>', unsafe_allow_html=True)
    weight = st.slider("", min_value=-3.0, max_value=3.0, value=1.0, step=0.1, key="weight")

    st.markdown('<p class="content-font">Bias (b):</p>', unsafe_allow_html=True)
    bias = st.slider("", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, key="bias")

    st.markdown('<p class="content-font">Transfer Function (f):</p>', unsafe_allow_html=True)
    selected_function = st.selectbox("", options=list(transfer_functions.keys()))


with col2:
    st.markdown('<p class="content-font">'
                '<br>'
                '<br>'
                '<br>'
                '<br>'
                '</p>', unsafe_allow_html=True)
    st.markdown('<div class="vertical-line" style="height: 800px;"></div>', unsafe_allow_html=True)


with col1:
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="font" style="float: left;">
                <span class="title-line"><em>Neural Network</em></span>
                <span class="title-line">DESIGN</span>
            </div>
            <div class="header" style="float: right;">Chapter 2: One-Input Neuron</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
    st.markdown(load_svg2(get_image_path("Chapters/2/SingleInputNeuron.svg")), unsafe_allow_html=True)
    fig, ax = plt.subplots()
    p = np.arange(-4, 4, 0.1)  # Input range
    func = np.vectorize(transfer_functions[selected_function])
    out = func(weight * p + bias)

    ax.plot(p, out, color="red")
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True, which='both')

    # Set x and y axis limits

    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])

    ax.set_xlabel("$p$")
    ax.set_ylabel("$a$")
    ax.set_title("$a = f(w \cdot p + b)$")

    st.pyplot(fig)




