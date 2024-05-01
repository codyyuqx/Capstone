import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
import math

st.set_page_config(layout="wide")

if "weight1_1" not in st.session_state:
    st.session_state["weight1_1"] = -1.0
if "bias1_1" not in st.session_state:
    st.session_state["bias1_1"] = 2.0
if "weight2_1" not in st.session_state:
    st.session_state["weight2_1"] = 1.0
if "bias2" not in st.session_state:
    st.session_state["bias2"] = 0.0
if "weight1_2" not in st.session_state:
    st.session_state["weight1_2"] = 1.0
if "bias1_2" not in st.session_state:
    st.session_state["bias1_2"] = 2.0
if "weight2_2" not in st.session_state:
    st.session_state["weight2_2"] = 1.0


def run():
    def get_graph():
        weight1_1 = st.session_state["weight1_1"]
        weight1_2 = st.session_state["weight1_2"]
        bias1_1 = st.session_state["bias1_1"]
        bias1_2 = st.session_state["bias1_2"]
        weight2_1 = st.session_state["weight1_2"]
        weight2_2 = st.session_state["weight2_2"]
        bias2 = st.session_state["bias2"]

        figure, axis = plt.subplots(figsize=(15, 5))
        axis = figure.add_subplot(1, 1, 1)
        axis.set_xlim(-5, 5)
        axis.set_ylim(0, 1)
        axis.plot(
            [0] * 50,
            np.linspace(-5, 5, 50),
            color="black",
            linestyle="--",
            linewidth=0.2,
        )
        axis.plot(
            np.linspace(0, 1, 10),
            [0] * 10,
            color="black",
            linestyle="--",
            linewidth=0.2,
        )
        (axis_output,) = axis.plot([], [], markersize=3, color="red")
        weight_1, bias_1 = np.array([[weight1_1, weight1_2]]), np.array(
            [[bias1_1, bias1_2]]
        )
        weight_2, bias_2 = np.array([[weight2_1], [weight2_2]]), np.array([[bias2]])

        p = np.arange(-5, 5, 0.01)

        out = weight_2[0, 0] * np.exp(-(((p - weight_1[0, 0]) * bias_1[0, 0]) ** 2))
        out += (
            weight_2[1, 0] * np.exp(-(((p - weight_1[0, 1]) * bias_1[0, 1]) ** 2))
            + bias_2[0, 0]
        )
        axis_output.set_data(p, out.reshape(-1))
        axis.legend()
        st.pyplot(figure, use_container_width=True)

    def on_random():
        st.session_state["weight1_1"] = round(np.random.uniform(-10, 10))
        st.session_state["weight1_2"] = round(np.random.uniform(-10, 10))
        st.session_state["bias1_1"] = round(np.random.uniform(-10, 10))
        st.session_state["bias1_2"] = round(np.random.uniform(-10, 10))
        st.session_state["weight2_1"] = round(np.random.uniform(-2, 2))
        st.session_state["weight2_2"] = round(np.random.uniform(-2, 2))
        st.session_state["bias2"] = round(np.random.uniform(-2.0, 2.0))

    def load_svg(svg_file, graph=False):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg = f.read()
        svg_base64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        if graph:
            svg_html = f"""
            <div style="text-align: center; width: 100%;">
                <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 300px; margin: 20px;">
            </div>
            """
        else:
            svg_html = f"""
        <div style="text-align: center; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 150px; margin: 20px;">
        </div>
        """
        return svg_html

    # Set the layout to "wide"

    st.markdown(
        """
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
        .subheader {
        text-align: middle;
        font-size: 20px;
        font-family: 'Times New Roman', Times, serif;
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
        """,
        unsafe_allow_html=True,
    )

    # Set Image path to load logo and chpater image

    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join("..", "Figures", filename)

    col1, col2, col3 = st.columns([10, 0.1, 3])

    with col3:
        # Sliders for weight and bias
        st.markdown(load_svg(get_image_path("Logo_Ch_17.svg")), unsafe_allow_html=True)
        st.markdown(
            """<p class="content-font">Alter the network's parameters by dragging the slide bars."""
            "<br>"
            """<p class="content-font">Click on [Random] to set each parameter to a random value.</p>""",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="content-font"></p>', unsafe_allow_html=True)
        st.session_state["bias2"] = st.slider(
            "b2: ", -2.0, 2.0, step=0.1, value=st.session_state["bias2"]
        )
        st.button("Random", on_click=on_random)

    with col2:
        st.markdown(
            '<p class="content-font">' "<br>" "<br>" "<br>" "<br>" "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vertical-line" style="height: 800px;"></div>',
            unsafe_allow_html=True,
        )

    with col1:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="font" style="float: left;">
                    <span class="title-line"><em>Neural Network</em></span>
                    <span class="title-line">DESIGN</span>
                </div>
                <div class="header" style="float: right;">Chapter 17:Linear Least Squares</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)

        c1, c4, c2, c5, c3, c6 = st.columns(6)

        with c1:
            st.session_state["weight1_1"] = st.slider(
                "W1(1,1):", -4.0, 4.0, step=0.1, value=st.session_state["weight1_1"]
            )
        with c2:
            st.session_state["bias1_1"] = st.slider(
                "b1(1):", -4.0, 4.0, step=0.1, value=st.session_state["bias1_1"]
            )
        with c3:
            st.session_state["weight2_1"] = st.slider(
                "W2(1,1)", -2.0, 2.0, step=0.1, value=st.session_state["weight2_1"]
            )
        st.markdown(
            load_svg(get_image_path("nnd17_1.svg"), True),
            unsafe_allow_html=True,
        )

        c1, c4, c2, c5, c3, c6 = st.columns(6)
        with c1:
            st.session_state["weight1_2"] = st.slider(
                "W1(2,1):", -4.0, 4.0, step=0.1, value=st.session_state["weight1_2"]
            )
        with c2:
            st.session_state["bias1_2"] = st.slider(
                "b1(2):", -4.0, 4.0, step=0.1, value=st.session_state["bias1_2"]
            )
        with c3:
            st.session_state["weight2_2"] = st.slider(
                "W2(1,2)", -2.0, 2.0, step=0.1, value=st.session_state["weight2_2"]
            )
        get_graph()


if __name__ == "__main__":
    run()
