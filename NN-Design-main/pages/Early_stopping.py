import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import base64
import os



def run():
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

    # Set the layout to "wide"
    st.set_page_config(layout="wide")

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
        """, unsafe_allow_html=True)
    #Redirect back to the Chapters page when clicked
    if st.button('Back to NND Page'):
        st.session_state.page = 'nnd'
        st.rerun()

    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join("..", "Logo", filename)

    def pause_train():
        # TODO: Add logic
        return True
    
    def on_run():
        # TODO: Add logic
        pause_button.setText("Pause")
        pause = True
        init_params()
        ani_stop()
        net_approx.set_data([], [])
        train_error, test_error = [], []
        #canvas.draw()
        #canvas2.draw()
        #run_animation()
    
    def init_params():
        W1 = np.random.uniform(-0.5, 0.5, (S1, 1))
        b1 = np.random.uniform(-0.5, 0.5, (S1, 1))
        W2 = np.random.uniform(-0.5, 0.5, (1, S1))
        b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def ani_stop(self):
        if ani_1 and ani_1.event_source:
            ani_1.event_source.stop()
        if ani_2 and ani_2.event_source:
            ani_2.event_source.stop()

    def ani_start():
        if ani_1 and ani_1.event_source:
            ani_1.event_source.start()
        if ani_2 and ani_2.event_source:
            ani_2.event_source.start()

    def on_stop():
        if pause:
            ani_stop()
            pause_button.setText("Unpause")
            pause = False
        else:
            ani_start()
            pause_button.setText("Pause")
            pause = True
        

    col1, col2, col3 = st.columns([10, 0.1, 3])

    with col3:
        # Sliders for weight and bias
        st.markdown(load_svg(get_image_path("Logo_Ch_13.svg")), unsafe_allow_html=True)
        st.markdown('<p class="content-font">Use the slider to change the\nNoise Standard Deviation of\nthe training points.\n\n'
                    "Click [Train] to train\non the training points.\n\nThe training and validation\n"
                    "The training and validation\n"
                    "performance indexes will be\npresented on the right.\n\nYou will notice that\n"
                    "without early stopping\nthe validation error\nwill increase.</p>", unsafe_allow_html=True)

        nsd_value = st.empty()
        noise_sd = st.slider("Noise Standard Deviation", min_value=0.0, max_value=3.0, value=1.0, step=0.1, key="noise_sd", label_visibility="collapsed")
        nsd_value.markdown("""<p class="content-font">Noise standard deviation: {}</p>""".format(noise_sd), unsafe_allow_html=True)
        nsd = float(noise_sd / 10)
        

        # TODO: add button press logic
        train = st.button("Train", on_click=on_run)
        pause_button = st.button("Pause", on_click=pause_train, key="pause_button", disabled=False)
        unpause_button = st.button("UnPause", on_click=pause_train, key="unpause_button", disabled=True)

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
                <div class="header" style="float: right;">Chapter 13: Early Stopping</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        #Plot the neuron
        figure1, axes_1 = plt.subplots()
        figure2, axes_2 = plt.subplots()

        max_epoch = 120
        T = 2
        pp0 = np.linspace(-1, 1, 201)
        tt0 = np.sin(2 * np.pi * pp0 / T)

        pp = np.linspace(-0.95, 0.95, 20)
        p = np.linspace(-1, 1, 21)

        #make_plot(1, (100, 90, 300, 300))
        #make_plot(2, (100, 380, 300, 300))

        train_error, error_train = [], None
        test_error, error_test = [], None
        ani_1, ani_2 = None, None
        W1, b1, W2, b2 = None, None, None, None
        S1, random_state = 20, 42
        tt, t = None, None

        axes_1.set_title("Function", fontdict={'fontsize': 10})
        axes_1.set_xlim(-1, 1)
        axes_1.set_ylim(-1.5, 1.5)
        axes_1.plot(pp0, np.sin(2 * np.pi * pp0 / T))
        net_approx, = axes_1.plot([], linestyle="--")
        train_points, = axes_1.plot([], marker='*', label="Train", linestyle="")
        test_points, = axes_1.plot([], marker='.', label="Test", linestyle="")
        axes_1.legend()
        c2.pyplot(figure1)

        axes_2.set_title("Performance Indexes", fontdict={'fontsize': 10})
        train_e, = axes_2.plot([], [], linestyle='-', color="b", label="train error")
        test_e, = axes_2.plot([], [], linestyle='-', color="r", label="test error")
        axes_2.legend()
        axes_2.plot(1, 1000, marker="*")
        axes_2.plot(100, 1000, marker="*")
        axes_2.plot(1, 0.01, marker="*")
        axes_2.plot(100, 0.01, marker="*")
        axes_2.set_xscale("log")
        axes_2.set_yscale("log")
        c2.pyplot(figure2)

if __name__ == "__main__":
    run()