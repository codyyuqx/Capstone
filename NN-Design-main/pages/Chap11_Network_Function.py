import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
import math
from st_pages import Page, show_pages, add_page_title, hide_pages
from constants import pages_created


def get_graph(weight1_1, bias1_1, weight2_1, bias2, weight1_2, bias1_2, weight2_2):
    fig, ax = plt.subplots(figsize=(15, 5))

    weight_1, bias_1 = np.array([[weight1_1, weight1_2]]), np.array(
        [[bias1_1, bias1_2]]
    )
    weight_2, bias_2 = np.array([[weight2_1], [weight2_2]]), np.array([[bias2]])
    # Plot the neuron
    p = np.arange(-4, 4, 0.01)
    func = np.vectorize(transfer_functions[selected_function])
    out = func(
        np.dot(tansig(np.dot(p.reshape(-1, 1), weight_1) + bias_1), weight_2)
        + bias_2
    )

    ax.plot(p, out, color="red")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(True, which="both")

    # Set x and y axis limits

    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])

    ax.set_xlabel("$p$")
    ax.set_ylabel("$a$")
    ax.set_title("a = {}(w2 * tansig(w1 * p + b1) + b2))".format(selected_function))

    ax.legend()
    st.pyplot(fig)

def on_random():
    st.session_state["weight1_1"] = round(np.random.uniform(-10, 10))
    st.session_state["weight1_2"] = round(np.random.uniform(-10, 10))
    st.session_state["bias1_1"] = round(np.random.uniform(-10, 10))
    st.session_state["bias1_2"] = round(np.random.uniform(-10, 10))
    st.session_state["weight2_1"] = round(np.random.uniform(-2, 2))
    st.session_state["weight2_2"] = round(np.random.uniform(-2, 2))
    st.session_state["bias2"] = round(np.random.uniform(-2, 2))

def purelin(n):
    return n

def logsig(x):
    return 1 / (1 + math.e ** (-x))

def tansig(x):
    return 2 / (1 + math.e ** (-2 * x)) - 1

transfer_functions = {"purelin": purelin, "logsig": logsig, "tansig": tansig}


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')
    def load_svg(svg_file, graph=False):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg = f.read()
        svg_base64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        if graph:
            svg_html = f"""
            <div style="text-align: center; width: 100%;">
                <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 100%; height: 300px; margin: 0px;">
            </div>
            """
        else:
            svg_html = f"""
        <div style="text-align: left; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 100px; margin: 10px;">
        </div>
        """
        return svg_html


    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)


    hide_pages(pages_created)


    image_path = 'media'


    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('Network Function')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')
    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/17.svg")), unsafe_allow_html=True)
        st.markdown("Alter the network's parameters by dragging Choose the output transfer function f below.")
        st.markdown("Click on [Random] to set each parameter set each parameter Watch the change to the neuron function and its output.")

        selected_function = st.selectbox("", options=list(transfer_functions.keys()))
        rnd_btn = st.button("Random")
        if rnd_btn:
            on_random()
        st.subheader('*Chapter11*')
        st.markdown('---')


    c1, c4, c2, c5, c3, c6 = st.columns(6)

    if "weight1_1" not in st.session_state:
        st.session_state["weight1_1"] = None
    if "bias1_1" not in st.session_state:
        st.session_state["bias1_1"] = None
    if "weight2_1" not in st.session_state:
        st.session_state["weight2_1"] = None
    if "bias2" not in st.session_state:
        st.session_state["bias2"] = None
    if "weight1_2" not in st.session_state:
        st.session_state["weight1_2"] = None
    if "bias1_2" not in st.session_state:
        st.session_state["bias1_2"] = None
    if "weight2_2" not in st.session_state:
        st.session_state["weight2_2"] = None

    c1, c4, c2, c5, c3, c6 = st.columns(6)

    with c1:
        st.session_state["weight1_1"] = st.slider(
            "W1(1,1):", -10, 10, st.session_state["weight1_1"]
        )
    with c2:
        st.session_state["bias1_1"] = st.slider(
            "b1(1):", -10, 10, st.session_state["bias1_1"]
        )
    with c3:
        st.session_state["weight2_1"] = st.slider(
            "W2(1,1)", -2, 2, st.session_state["weight2_1"]
        )
    st.markdown(
        load_svg(get_image_path("Figures/nnd11_1.svg"), True), unsafe_allow_html=True
    )
    c1, c4, c2, c5, c3, c6 = st.columns(6)
    with c3:
        st.session_state["bias2"] = st.slider(
            "b2:", -2, 2, st.session_state["bias2"]
        )
    c1, c4, c2, c5, c3, c6 = st.columns(6)
    with c1:
        st.session_state["weight1_2"] = st.slider(
            "W1(2,1):", -10, 10, st.session_state["weight1_2"]
        )
    with c2:
        st.session_state["bias1_2"] = st.slider(
            "b1(2):", -10, 10, st.session_state["bias1_2"]
        )
    with c3:
        st.session_state["weight2_2"] = st.slider(
            "W2(1,2)", -2, 2, st.session_state["weight2_2"]
        )
    get_graph(
        st.session_state["weight1_1"],
        st.session_state["bias1_1"],
        st.session_state["weight2_1"],
        st.session_state["bias2"],
        st.session_state["weight1_2"],
        st.session_state["bias1_2"],
        st.session_state["weight2_2"],
    )


