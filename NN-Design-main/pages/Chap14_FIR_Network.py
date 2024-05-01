import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
from st_pages import hide_pages
from constants import pages_created
import os
import base64

from scipy.signal import lfilter


class FIRNetwork():
    def __init__(self, func1, freq, weight_0, weight_1, weight_2, autoscale):
        w_ratio, h_ratio, dpi = 1, 1, 100

        self.autoscale = autoscale
        # self.make_combobox(3, ["No", "Yes"], (self.x_chapter_usual, 445, self.w_chapter_slider, 100),
        #                    self.change_autoscale,
        #                    "label_autoscale", "Autoscale", (self.x_chapter_slider_label, 420, 150, 100))

        # self.make_plot(1, (15, 300, 500, 370))
        # self.axis = self.figure.add_subplot(1, 1, 1)

        self.figure = plt.figure()
        self.axis = self.figure.add_subplot(1, 1, 1)

        if not self.autoscale:
            self.axis.set_xlim(0, 25)
            self.axis.set_ylim(-6, 6)
        self.axis.plot(np.linspace(0, 25, 50), [0] * 50, color="red", linestyle="--", linewidth=0.2)
        self.axis_a1, = self.axis.plot([], [], color="white", marker="o", markeredgecolor="red", linestyle="none")
        self.axis_a2, = self.axis.plot([], [], color="blue", marker=".", markersize=3, linestyle="none")

        # self.comboBox1_functions_str = ["square", 'sine']
        # self.make_combobox(1, self.comboBox1_functions_str, (self.x_chapter_usual, 505, self.w_chapter_slider, 100), self.change_transfer_function,
        #    "label_f", "f", (self.x_chapter_slider_label + 20, 480, 150, 100))
        self.func1 = func1

        # self.comboBox2_divs = ["1/16", '1/14', '1/12', '1/10', '1/8']
        # self.make_combobox(2, self.comboBox2_divs, (self.x_chapter_usual, 595, self.w_chapter_slider, 50), self.change_freq,
        #    "label_div", "frequency", (self.x_chapter_slider_label, 570, 150, 50))
        self.freq = freq

        # self.make_slider("slider_w0", QtCore.Qt.Orientation.Horizontal, (-20, 20), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 3,
        #  (self.x_chapter_usual, 270, self.w_chapter_slider, 50), self.graph,
        #                  "label_w0", "iW(0): 0.3",
        #                  (self.x_chapter_slider_label, 240, 150, 50))
        # self.make_slider("slider_w1", QtCore.Qt.Orientation.Horizontal, (-20, 20), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 3,
        #                  (self.x_chapter_usual, 340, self.w_chapter_slider, 50), self.graph,
        #                  "label_w1", "iW(1): 0.3",
        #                  (self.x_chapter_slider_label, 310, 150, 50))
        # self.make_slider("slider_w2", QtCore.Qt.Orientation.Horizontal, (-20, 20), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 3,
        #                  (self.x_chapter_usual, 410, self.w_chapter_slider, 50), self.graph,
        #                  "label_w2", "iW(2): 0.3",
        #                  (self.x_chapter_slider_label, 380, 150, 50))

        # self.comboBox2.setCurrentIndex(2)

        if self.autoscale:
            self.figure.clf()
            self.axis = self.figure.add_subplot(1, 1, 1)

            self.axis.plot(np.linspace(0, 25, 50), [0] * 50, color="red", linestyle="--", linewidth=0.2)

        if self.func1 == "square":
            if self.freq == 1 / 16:
                p = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
            elif self.freq == 1 / 14:
                p = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
            elif self.freq == 1 / 12:
                p = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
            elif self.freq == 1 / 10:
                p = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
            elif self.freq == 1 / 8:
                p = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
        else:
            p = np.sin(np.arange(0, 24, 1) * 2 * np.pi * self.freq)

        a0, a_1, t, t1 = 0, 0, list(range(1, len(p) + 1)), list(range(len(p) + 1))
        num = np.array([weight_0, weight_1, weight_2])
        den = np.array([1])
        zi = np.array([a0, a_1])
        A = lfilter(num, den, p, zi=zi)

        if self.autoscale:
            self.axis.plot(t, p, color="white", marker="o", markeredgecolor="red", linestyle="none")
            self.axis.plot(t1, [a0] + list(A[0]), color="blue", marker=".", markersize=3, linestyle="none")
        else:
            self.axis_a1.set_data(t, p)
            self.axis_a2.set_data(t1, [a0] + list(A[0]))

        # self.axis.set_xlabel("Time")


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')

    hide_pages(pages_created)

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('FIR Network')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/14.svg")), unsafe_allow_html=True)
        st.markdown(
            'Select the input and frequency to the FIR network. \n \n Use the sliders to alter the network weights.')
        weight_0 = st.slider('iW(0)', -2.0, 2.0, 0.3)
        weight_1 = st.slider('iW(1)', -2.0, 2.0, 0.3)
        weight_2 = st.slider('iW(2)', -2.0, 2.0, 0.3)
        func1 = st.selectbox('Function (f)', ['square', 'sine'])
        freq = st.selectbox('Frequency', ['1/16', '1/14', '1/12', '1/10', '1/8'])
        freq = dict(zip(['1/16', '1/14', '1/12', '1/10', '1/8'], [1 / 16, 1 / 14, 1 / 12, 1 / 10, 1 / 8])).get(freq)
        autoscale = st.selectbox('Autoscale', ['No', 'Yes'])
        st.subheader('*Chapter14*')
        st.markdown('---')

    app = FIRNetwork(func1, freq, weight_0, weight_1, weight_2, autoscale)

    st.markdown(load_svg_2(get_image_path("Figures/nnd14_1.svg")), unsafe_allow_html=True)
    #st.image('media/Figures/nnd14_1.svg', width=550)
    st.text('')

    st.pyplot(app.figure, use_container_width=True)