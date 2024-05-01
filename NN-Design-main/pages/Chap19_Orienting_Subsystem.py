import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
from st_pages import hide_pages
from scipy.signal import lfilter
import matplotlib
from constants import pages_created
from scipy.integrate import ode
import base64
import os


class OrientingSubsystem():
    def __init__(self, slider_input_pos, slider_input_neg, slider_bias_pos, slider_bias_neg, slider_tcte, slider_tcte1):

        self.t = np.arange(0, 0.201, 0.001)
        self.figure = plt.figure(figsize=(7, 6))

        self.figure.subplots_adjust(left=0.175, right=0.95, bottom=0.125, top=0.9)
        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.201)
        self.axis.set_ylim(-1, 1)
        self.axis.plot(np.linspace(0, 0.201, 10), [0] * 10, color="black", linestyle="--", linewidth=0.5)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Reset a0")
        self.axis.set_title("Response")
        self.axis.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        # self.lines = []
        self.lines = st.session_state['lines'] if 'lines' in st.session_state else []

        # self.make_slider("slider_input_pos", QtCore.Qt.Orientation.Horizontal, (0, 1), QtWidgets.QSlider.TickPosition.TicksAbove, 1, 1,
        #                  (self.x_chapter_usual, 330, self.w_chapter_slider, 50), self.slide,
        #                  "label_input_pos", "Input p(1): 1", (self.x_chapter_usual + 60, 330 - 25, 150, 50))
        # self.make_slider("slider_input_neg", QtCore.Qt.Orientation.Horizontal, (0, 1), QtWidgets.QSlider.TickPosition.TicksAbove, 1, 1,
        #                  (self.x_chapter_usual, 390, self.w_chapter_slider, 50), self.slide,
        #                  "label_input_neg", "Input p(2): 1", (self.x_chapter_usual + 60, 390 - 25, 150, 50))
        # self.make_slider("slider_bias_pos", QtCore.Qt.Orientation.Horizontal, (0, 1), QtWidgets.QSlider.TickPosition.TicksAbove, 1, 1,
        #                  (self.x_chapter_usual, 450, self.w_chapter_slider, 50), self.slide,
        #                  "label_bias_pos", "Input a1(1): 1", (self.x_chapter_usual + 50, 450 - 25, 150, 50))
        # self.make_slider("slider_bias_neg", QtCore.Qt.Orientation.Horizontal, (0, 1), QtWidgets.QSlider.TickPosition.TicksAbove, 1, 0,
        #                  (self.x_chapter_usual, 510, self.w_chapter_slider, 50), self.slide,
        #                  "label_bias_neg", "Input a1(2): 0", (self.x_chapter_usual + 50, 510 - 25, 150, 50))

        # self.make_slider("slider_tcte", QtCore.Qt.Orientation.Horizontal, (1, 50), QtWidgets.QSlider.TickPosition.TicksAbove, 1, 30,
        #                  (20, 560, 480, 50), self.slide, "label_tcte", "+W0 Elements: 3.00", (200, 535, 170, 50))
        # self.make_slider("slider_tcte1", QtCore.Qt.Orientation.Horizontal, (1, 50), QtWidgets.QSlider.TickPosition.TicksAbove, 1, 40,
        #                  (20, 630, 480, 50), self.slide, "label_tcte1", "-W0 Elements: 4.00", (200, 605, 170, 50))

        # self.make_button("clear_button", "Clear", (self.x_chapter_button, 575, self.w_chapter_button, self.h_chapter_button), self.on_clear)
        # self.make_button("random_button", "Update", (self.x_chapter_button, 605, self.w_chapter_button, self.h_chapter_button), self.graph)
        self.slider_input_pos = slider_input_pos
        self.slider_input_neg = slider_input_neg
        self.slider_bias_pos = slider_bias_pos
        self.slider_bias_neg = slider_bias_neg
        self.slider_tcte = slider_tcte
        self.slider_tcte1 = slider_tcte1

        self.graph()

    def slide(self):
        self.pp = self.slider_input_pos.value()
        self.pn = self.slider_input_neg.value()
        self.label_input_pos.setText("Input p(1): " + str(self.pp))
        self.label_input_neg.setText("Input p(2): " + str(self.pn))
        self.bp = self.slider_bias_pos.value()
        self.bn = self.slider_bias_neg.value()
        self.label_bias_pos.setText("Input a1(1): " + str(round(self.bp, 2)))
        self.label_bias_neg.setText("Input a1(2): " + str(round(self.bn, 2)))
        self.A = self.slider_tcte.value() / 10
        self.B = self.slider_tcte1.value() / 10
        self.label_tcte.setText("+W0 Elements: " + str(round(self.A, 2)))
        self.label_tcte1.setText("-W0 Elements: " + str(round(self.B, 2)))

    def shunt(self, t, y):
        return (-y + (1 - y) * self.A * (self.p[0, 0] + self.p[1, 0]) - (y + 1) * self.B * (
                    self.a[0, 0] + self.a[1, 0])) / 0.1

    def graph(self):
        self.pp = self.slider_input_pos
        self.pn = self.slider_input_neg
        self.bp = self.slider_bias_pos
        self.bn = self.slider_bias_neg
        self.A = self.slider_tcte
        self.B = self.slider_tcte1
        # self.label_input_pos.setText("Input p(1): " + str(self.pp))
        # self.label_input_neg.setText("Input p(2): " + str(self.pn))
        # self.label_bias_pos.setText("Input a1(1): " + str(self.bp))
        # self.label_bias_neg.setText("Input a1(2): " + str(self.bn))
        # self.label_tcte.setText("+W0 Elements: " + str(round(self.A, 2)))
        # self.label_tcte1.setText("-W0 Elements: " + str(round(self.B, 2)))
        self.p = np.array([[self.pp], [self.pn]])
        self.a = np.array([[self.bp], [self.bn]])
        r = ode(self.shunt).set_integrator("zvode")
        r.set_initial_value(np.array([0, 0]), 0)
        t1 = 0.201
        dt = 0.001
        out = []
        while r.successful() and r.t < t1:
            out.append(r.integrate(r.t + dt)[0].item())
        out[0] = 0
        while len(self.lines) > 3:
            self.lines.pop(0)
        for line in self.lines:
            line[2] = "gray"
            line[3] = 0.5
        # self.lines.append(self.axis.plot(self.t, out, color="red")[0])
        self.lines.append([self.t, out, "red", 1])

        for line in self.lines:
            self.axis.plot(line[0], line[1], color=line[2], alpha=line[3])

        st.session_state['lines'] = self.lines
        # self.canvas.draw()

    def on_clear(self):
        while len(self.lines) > 1:
            self.lines.pop(0).remove()
        self.canvas.draw()


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')

    hide_pages(pages_created)
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

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.text('')
        st.subheader('Orientation Subsystem')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/19.svg")), unsafe_allow_html=True)
        st.markdown(
            "Adjust the inputs, biases and gain.\nThen click [Update] to see the layer respond.\n\nn2(1) is red, n2(2) is green.\n\nClick [Clear] to remove old responses.")
        clear = st.button('Clear')
        if clear:
            st.session_state['lines'] = []
        slider_input_pos = st.slider('Input p(1)', 0, 1, 1)
        slider_input_neg = st.slider('Input p(2)', 0, 1, 1)
        slider_bias_pos = st.slider('Input a1(1)', 0, 1, 1)
        slider_bias_neg = st.slider('Input a1(2)', 0, 1, 0)
        slider_tcte = st.slider('+W0 Elements', 0.1, 5.0, 3.0)
        slider_tcte1 = st.slider('-W0 Elements', 0.1, 5.0, 4.0)
        st.subheader('*Chapter19*')
        st.markdown('---')



    app = OrientingSubsystem(slider_input_pos, slider_input_neg, slider_bias_pos, slider_bias_neg, slider_tcte,
                             slider_tcte1)

    gap_cols = st.columns([1, 12, 1])
    with gap_cols[1]:
        st.pyplot(app.figure)