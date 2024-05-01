import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
import math

st.set_page_config(layout="wide")

if "S1" not in st.session_state:
    st.session_state["S1"] = 5
if "n_points" not in st.session_state:
    st.session_state["n_points"] = 10
if "ro" not in st.session_state:
    st.session_state["ro"] = 0.0
if "sigma" not in st.session_state:
    st.session_state["sigma"] = 0.0
if "freq" not in st.session_state:
    st.session_state["freq"] = 0.5
if "phase" not in st.session_state:
    st.session_state["phase"] = 90
if "w1_1" not in st.session_state:
    st.session_state["w1_1"] = -2.0
if "bias" not in st.session_state:
    st.session_state["bias"] = 1.67


def run():

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

    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)

    image_path ='/Users/yuqixiong/Desktop/NN-Design-main/media'

    col1, col2, col3 = st.columns([10, 0.1, 3])

    with col3:
        # Sliders for weight and bias
        st.markdown(load_svg(get_image_path("Logo/Logo_Ch_11.svg")), unsafe_allow_html=True)
        st.markdown(
            """<p class="content-font">Basis functions are spaced evenly. You can change the first 
            center location and the bias. The automatic bias will produce 
            overlap at 0.5. """
            "<br>"
            """<p class="content-font">The function is shown in blue and the network 
            response in red.</p>""",
            unsafe_allow_html=True,
        )

        st.session_state["w1_1"] = st.slider(
            "W1(1,1): ", -2.0, 2.0, step=0.1, value=st.session_state["w1_1"]
        )
        st.session_state["bias"] = st.slider(
            "b: ", 0.0, 10.0, step=0.01, value=st.session_state["bias"]
        )
        auto_bias = st.selectbox("Auto Bias", ("Yes", "No"))
        if auto_bias == "Yes":
            auto_bias = True
        else:
            auto_bias = False

    with col2:
        st.markdown(
            '<p class="content-font">' "<br>" "<br>" "<br>" "<br>" "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vertical-line" style="height: 650px;"></div>',
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
        randseq = [
            -0.7616,
            -1.0287,
            0.5348,
            -0.8102,
            -1.1690,
            0.0419,
            0.8944,
            0.5460,
            -0.9345,
            0.0754,
            -0.7616,
            -1.0287,
            0.5348,
            -0.8102,
            -1.1690,
            0.0419,
            0.8944,
            0.5460,
            -0.9345,
            0.0754,
        ]

        figure, axis = plt.subplots(figsize=(15, 5))
        figure2, axis2 = plt.subplots(figsize=(15, 2))
        # figure2.set_tight_layout(True)

        axis.set_xlim(-2, 2)
        axis.set_ylim(-2, 4)
        axis.set_xticks([-2, -1, 0, 1])
        axis.set_yticks([-2, -1, 0, 1, 2, 3])
        axis.plot(
            np.linspace(-2, 2, 10),
            [0] * 10,
            color="black",
            linestyle="--",
            linewidth=0.2,
        )
        axis.set_xlabel("p")
        axis.xaxis.set_label_coords(1, -0.025)
        axis.set_ylabel("$a^2$")
        axis.yaxis.set_label_coords(-0.025, 1)

        # axis2 = figure2.add_subplot(1, 1, 1)
        axis2.set_xlim(-2, 2)
        axis2.set_ylim(0, 1)
        axis2.set_xticks([-2, -1, 0, 1])
        axis2.set_yticks([0, 0.5])
        axis2.set_xlabel("p")
        axis2.xaxis.set_label_coords(1, -0.025)
        axis2.set_ylabel("$a^1")
        axis2.yaxis.set_label_coords(-0.025, 1)

        auto_bias = True

        d1 = (2 - -2) / (st.session_state["n_points"] - 1)
        p = np.arange(-2, 2 + 0.0001, d1)
        t = (
            np.sin(
                2
                * np.pi
                * (st.session_state["freq"] * p + st.session_state["phase"] / 360)
            )
            + 1
            + st.session_state["sigma"] * np.array(randseq[: len(p)])
        )
        delta = (2 - -2) / (st.session_state["S1"] - 1)
        if auto_bias:
            st.session_state["bias"] = 1.6652 / delta

        total = 2 - -2
        W1 = (
            np.arange(-2, 2 + 0.0001, delta) + st.session_state["w1_1"] - -2
        ).T.reshape(-1, 1)
        b1 = st.session_state["bias"] * np.ones(W1.shape)
        Q = len(p)
        pp = np.repeat(p.reshape(1, -1), st.session_state["S1"], 0)
        n1 = np.abs(pp - np.dot(W1, np.ones((1, Q)))) * np.dot(b1, np.ones((1, Q)))
        a1 = np.exp(-(n1**2))
        Z = np.vstack((a1, np.ones((1, Q))))
        x = np.dot(
            np.linalg.pinv(
                np.dot(Z, Z.T)
                + st.session_state["ro"] * np.eye(st.session_state["S1"] + 1)
            ),
            np.dot(Z, t.T),
        )
        W2, b2 = x[:-1].T, x[-1]
        a2 = np.dot(W2, a1) + b2
        p2 = np.arange(-2, 2 + total / 100, total / 100)
        Q2 = len(p2)
        pp2 = np.repeat(p2.reshape(1, -1), st.session_state["S1"], 0)
        n12 = np.abs(pp2 - np.dot(W1, np.ones((1, Q2)))) * np.dot(b1, np.ones((1, Q2)))
        a12 = np.exp(-(n12**2))
        a22 = np.dot(W2, a12) + b2
        t_exact = (
            np.sin(
                2
                * np.pi
                * (st.session_state["freq"] * p2 + st.session_state["phase"] / 360)
            )
            + 1
        )
        temp = np.vstack((np.dot(W2.T, np.ones((1, Q2)) * a12), b2 * np.ones((1, Q2))))

        while len(axis.lines) > 1:
            axis.lines[-1].remove()
        if axis.collections:
            axis.collections[0].remove()
        axis.scatter(p, t, color="white", edgecolor="black")
        for i in range(len(temp)):
            axis.plot(p2, temp[i], linestyle="--", color="black", linewidth=0.5)

        axis.plot(p2, t_exact, color="blue", linewidth=2)
        axis.plot(p2, a22, color="red", linewidth=1)

        while axis2.lines:
            axis2.lines[-1].remove()
        for i in range(len(a12)):
            axis2.plot(p2, a12[i], color="black")

        st.pyplot(figure)
        st.pyplot(figure2)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.session_state["S1"] = st.slider(
                "Hidden Neurons: ", 2, 9, value=st.session_state["S1"]
            )
        with c3:
            st.session_state["n_points"] = st.slider(
                "Number of Points: ", 2, 20, value=st.session_state["n_points"]
            )
        with c5:
            st.session_state["ro"] = st.slider(
                "Regularization: ", 0.0, 1.0, step=0.1, value=st.session_state["ro"]
            )
        with c1:
            st.session_state["sigma"] = st.slider(
                "Stdev Noise: ", 0.0, 1.0, step=0.1, value=st.session_state["sigma"]
            )
        with c3:
            st.session_state["freq"] = st.slider(
                "Function Frequency: ",
                0.0,
                1.0,
                step=0.01,
                value=st.session_state["freq"],
            )
        with c5:
            st.session_state["phase"] = st.slider(
                "Function Phase: ", 0, 360, value=st.session_state["phase"]
            )


if __name__ == "__main__":
    run()
