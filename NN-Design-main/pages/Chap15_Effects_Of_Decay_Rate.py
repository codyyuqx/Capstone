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


class EffectsOfDecayRate():
    def __init__(self, slider_lr, slider_dr, clear, random):
        w_ratio, h_ratio, dpi = 1, 1, 100

        # self.make_plot(1, (20, 100, 470, 470))
        # self.axis = self.figure.add_subplot(1, 1, 1)
        self.figure, self.axis = plt.subplots(1, 1, figsize=(4, 4))
        self.slider_lr = slider_lr
        self.slider_dr = slider_dr

        self.axis.set_xlim(0, 30)
        self.axis.set_ylim(0, 10)
        self.axis.plot([0] * 30, np.linspace(0, 30, 30), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot(np.linspace(0, 10, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Weight")
        self.axis.set_title("Hebb Learning", fontsize=10)
        self.lines = st.session_state['lines'] if 'lines' in st.session_state else []

        # self.make_button("clear_button", "Clear", (self.x_chapter_button, 320, self.w_chapter_button, self.h_chapter_button), self.on_clear)
        # self.make_button("random_button", "Random", (self.x_chapter_button, 350, self.w_chapter_button, self.h_chapter_button), self.on_random)

        self.do_graph = True

        if clear or random:
            self.do_graph = False
            st.session_state['lines'] = []
            self.lines = []

        self.graph()

    def graph(self):

        lr = self.slider_lr
        dr = self.slider_dr
        # print(lr, dr, self.slider_lr, self.slider_dr)

        w = 0
        wtot = []
        for i in range(1, 31):
            a = self.hardlim(1 * (i % 2 == 0) + w * 1 - 0.5)
            w = w + lr * a * 1 - dr * w
            wtot.append(w)
        ind = [i for i in range(len(wtot)) if wtot[i] > 10]
        if ind:
            ind = ind[0]
            wtot = wtot[:ind - 1]
        while len(self.lines) > 3:
            self.lines.pop(0)
        for line in self.lines:
            line[3] = "gray"
            line[4] = 0.5
        self.lines.append([range(len(wtot)), wtot, "o", "red", 1])
        for line in self.lines:
            self.axis.plot(line[0], line[1], line[2], color=line[3], alpha=line[4])
        st.session_state['lines'] = self.lines

    def on_clear(self):
        while len(self.lines) > 1:
            self.lines.pop(0)
        st.session_state['lines'] = self.lines
        # self.canvas.draw()

    # def on_random(self):
    #     self.do_graph = False
    #     st.session_state['slider_lr'] = round(np.random.uniform(0, 1))
    #     # print('aaaaaa')
    #     self.do_graph = True
    #     st.session_state['slider_dr'] = round(np.random.uniform(0, 1))
    #     # print('bbbbbb')

    @staticmethod
    def hardlim(x):
        if x < 0:
            return 0
        else:
            return 1


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

    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)


    image_path = 'media'


    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('Effects of Decay Rate')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")
    val1, val2 = 0.1, 0.1

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/15.svg")), unsafe_allow_html=True)
        st.markdown("Use the slider bars to\nadjust learning and\ndecay rates.\n\n"
                    "Click [Clear] to remove\nold responses.\n\nClick [Random] to get\n"
                    "random parameters.")

        slider_lr = st.slider('Learning Rate', 0.0, 1.0, val1, key='slider_lr')
        slider_dr = st.slider('Decay Rate', 0.0, 1.0, val2, key='slider_dr')

        clear = st.button('Clear')
        random = st.button('Random')

    if random:
        val1 = np.random.uniform(0, 1)
        val2 = np.random.uniform(0, 1)



    app = EffectsOfDecayRate(slider_lr, slider_dr, clear, random)
    col1 = st.columns([1, 9, 1])
    with col1[1]:
        st.pyplot(app.figure)


