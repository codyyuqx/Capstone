import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
from st_pages import hide_pages
from constants import pages_created
from scipy.signal import lfilter
import matplotlib
import os
import base64

font = {'size': 10}

matplotlib.rc('font', **font)


class DynamicDerivatives():
    def __init__(self, freq, func1, weight_0, weight_1):
        w_ratio, h_ratio, dpi = 1, 1, 100
        self.freq = freq
        self.func1 = func1
        self.weight_0 = weight_0
        self.weight_1 = weight_1

        self.figure, self.a = plt.subplots(1, 1, figsize=(4, 4))
        self.figure2, self.a4 = plt.subplots(1, 1, figsize=(4, 4))
        self.figure3, self.a2 = plt.subplots(1, 1, figsize=(4, 4))
        self.figure4, self.a3 = plt.subplots(1, 1, figsize=(4, 4))

    def graph(self):

        # self.figure.clf()
        # self.figure2.clf()
        # self.figure3.clf()
        # self.figure4.clf()

        a = self.a
        a4 = self.a4
        a2 = self.a2
        a3 = self.a3

        a.set_xlim(0, 25)
        a2.set_xlim(0, 25)
        a3.set_xlim(0, 25)
        a4.set_xlim(0, 25)
        a4.set_title("Incremental Response iw + 0.1", )
        a.set_title("Incremental Response lw + 0.1", )
        a3.set_title("Derivative with respect to lw", )
        a2.set_title("Derivative with respect to iw", )
        # if not self.autoscale:
        #     a.set_xlim(0, 25)
        #     a.set_ylim(-6, 6)

        # a.set_xticks([0], minor=True)
        # a.set_yticks([0], minor=True)
        # a.set_xticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0.5, 1, 1.5])
        # a.grid(which="minor")
        # a.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        # a.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        a.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        a2.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        a3.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        a4.plot(np.linspace(0, 26, 50), [0] * 50, color="red", linestyle="dashed", linewidth=0.5)
        # a.plot(np.linspace(-2, 2, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        # a.set_xlabel("$p$")
        # a.xaxis.set_label_coords(1, -0.025)
        # a.set_ylabel("$a$")
        # a.yaxis.set_label_coords(-0.025, 1)

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

        weight_0 = self.weight_0
        weight_1 = self.weight_1

        a0, a_1, t, t1 = 0, 0, list(range(1, len(p) + 1)), list(range(len(p) + 1))
        num = np.array([weight_0])
        den = np.array([1, weight_1])
        zi = np.array([a0])
        A = lfilter(num, den, p, zi=zi)
        # a.scatter(t, p, color="white", marker="o", edgecolor="red")
        a.scatter(t1, [a0] + list(A[0]), color="black", marker=".", s=[10] * len(t1))
        lw111 = weight_1
        iw11 = weight_0 + 0.1
        num = np.array([iw11])
        den = np.array([1, lw111])
        a1 = lfilter(num, den, p, zi=zi)
        a.scatter(t1, [a0] + list(a1[0]), color="blue", marker="x", s=50)
        da_diw_0 = 0
        da_diw = lfilter(np.array([1]), den, p, zi=np.array([da_diw_0]))
        a2.scatter(t1, [da_diw_0] + list(da_diw[0]), color="white", marker="D", edgecolor="blue")
        a2.scatter(t, p, color="white", marker="s", edgecolor="black", s=[8] * len(t))

        da_dlw_0 = 0
        ad = np.array([a0] + list(A[0])[:-1])
        da_dlw = lfilter(np.array([1]), den, ad, zi=np.array([da_dlw_0]))
        a3.scatter(t1, [da_dlw_0] + list(da_dlw[0]), color="white", marker="D", edgecolor="blue")
        a3.scatter(t, ad, color="white", marker="s", edgecolor="black", s=[8] * len(t))

        a4.scatter(t1, [a0] + list(A[0]), color="black", marker=".", s=[10] * len(t1))
        lw111 = weight_1 + .1
        iw11 = weight_0
        num = np.array([iw11])
        den = np.array([1, lw111])
        a1 = lfilter(num, den, p, zi=zi)
        a4.scatter(t1, [a0] + list(a1[0]), color="blue", marker="x", s=50)




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
        st.subheader('Dynamic Derivatives')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/14.svg")), unsafe_allow_html=True)
        st.markdown("Original responses - black dots.")
        st.markdown("Incremental responses - blue crosses.")
        st.markdown("Total derivatives - blue diamonds.")
        st.markdown("Static derivatives - black squares.")
        st.markdown(
            "Select the input and frequency to the IIR network.\n\nUse the sliders to alter the network weights.")
        freq = st.selectbox('Frequency', ['1/16', '1/14', '1/12', '1/10', '1/8'])
        freq = dict(zip(['1/16', '1/14', '1/12', '1/10', '1/8'], [1 / 16, 1 / 14, 1 / 12, 1 / 10, 1 / 8])).get(freq)
        func1 = st.selectbox('Function', ['square', 'sine'])
        weight_0 = st.slider('iW(0)', -2.0, 2.0, 0.5)
        weight_1 = st.slider('lW(1)', -2.0, 2.0, -0.5)

        st.subheader('*Chapter14*')
        st.markdown('---')

    st.markdown(load_svg_2(get_image_path("Figures/nnd14_2.svg")), unsafe_allow_html=True)

    app = DynamicDerivatives(freq, func1, weight_0, weight_1)
    app.graph()
    # st.pyplot(app.figure)
    col1 = st.columns(2)
    with col1[0]:
        st.pyplot(app.figure)
    with col1[1]:
        st.pyplot(app.figure2)

    col2 = st.columns(2)
    with col2[0]:
        st.pyplot(app.figure3)
    with col2[1]:
        st.pyplot(app.figure4)