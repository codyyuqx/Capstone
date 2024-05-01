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


class ART1Layer2():
    def __init__(self, slider_input_pos, slider_input_neg, slider_bias_pos, slider_bias_neg, slider_tcte):

        self.t = np.arange(0, 0.21, 0.001)

        # self.make_plot(1, (20, 90, 480, 480))
        self.figure = plt.figure(figsize=(7, 6))

        self.figure.subplots_adjust(left=0.175, right=0.95, bottom=0.125, top=0.9)
        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.2)
        self.axis.set_ylim(-1, 1)
        self.axis.plot([0] * 20, np.linspace(-1, 1, 20), color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Outputs n2(1), n2(2)")
        self.axis.set_title("Response")
        self.axis.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        self.axis.plot(np.linspace(0, 0.2, 100), [0] * 100, linestyle="dashed", linewidth=0.5, color="gray")

        self.lines1 = st.session_state['lines1'] if 'lines1' in st.session_state else []
        self.lines2 = st.session_state['lines2'] if 'lines2' in st.session_state else []

        self.slider_input_pos = slider_input_pos
        self.slider_input_neg = slider_input_neg
        self.slider_bias_pos = slider_bias_pos
        self.slider_bias_neg = slider_bias_neg
        self.slider_tcte = slider_tcte

        # self.make_button("clear_button", "Clear", (self.x_chapter_button, 575, self.w_chapter_button, self.h_chapter_button), self.on_clear)
        # self.make_button("random_button", "Update", (self.x_chapter_button, 605, self.w_chapter_button, self.h_chapter_button), self.graph)

        self.do_graph = True

        self.graph()

    def slide(self):
        self.pp = self.slider_input_pos.value()
        self.pn = self.slider_input_neg.value()
        self.label_input_pos.setText("Input a(1): " + str(self.pp))
        self.label_input_neg.setText("Input a(2): " + str(self.pn))
        self.bp = self.slider_bias_pos.value() / 10
        self.bn = self.slider_bias_neg.value() / 10
        self.label_bias_pos.setText("Bias b+: " + str(round(self.bp, 2)))
        self.label_bias_neg.setText("Bias b-: " + str(round(self.bn, 2)))
        self.e = self.slider_tcte.value() / 10
        self.label_tcte.setText("Transfer Function Gain: " + str(round(self.e, 2)))

    def layer1(self, t, y):
        i1 = np.dot(np.array([[0.5, 0.5]]), self.p).item()
        i2 = np.dot(np.array([[1, 0]]), self.p).item()
        a1, a2 = 0, 0
        if y[0] > 0:
            a1 = (self.e * y[0] ** 2)
        if y[1] > 0:
            a2 = (self.e * y[1] ** 2)
        return [(-y[0] + (self.bp - y[0]) * (a1 + i1) - (y[0] + self.bn) * a2) / 0.1,
                (-y[1] + (self.bp - y[1]) * (a2 + i2) - (y[1] + self.bn) * a1) / 0.1]

    def graph(self):
        if self.do_graph:
            self.pp = self.slider_input_pos
            self.pn = self.slider_input_neg
            self.bp = self.slider_bias_pos
            self.bn = self.slider_bias_neg
            self.e = self.slider_tcte
            # self.label_input_pos.setText("Input a1(1): " + str(self.pp))
            # self.label_input_neg.setText("Input a1(2): " + str(self.pn))
            # self.label_bias_pos.setText("Bias b+: " + str(round(self.bp, 2)))
            # self.label_bias_neg.setText("Bias b-: " + str(round(self.bn, 2)))
            # self.label_tcte.setText("Transfer Function Gain: " + str(round(self.e, 2)))
            self.p = np.array([[self.pp], [self.pn]])
            r = ode(self.layer1).set_integrator("zvode")
            r.set_initial_value([0, 0], 0)
            t1 = 0.21
            dt = 0.001
            out_1, out_2 = [], []
            while r.successful() and r.t < t1:
                out = r.integrate(r.t + dt)
                out_1.append(out[0].item())
                out_2.append(out[1].item())
            out_1[0], out_2[0] = 0, 0
            while len(self.lines1) > 1:
                self.lines1.pop(0)
            while len(self.lines2) > 1:
                self.lines2.pop(0)
            for line in self.lines1:
                # line.set_color("gray")
                line[3] = 0.5
            for line in self.lines2:
                # line.set_color("gray")
                line[3] = 0.5

            # self.lines1.append(self.axis.plot(self.t, out_1, color="red")[0])
            # self.lines2.append(self.axis.plot(self.t, out_2, color="green")[0])

            self.lines1.append([self.t, out_1, 'red', 1, "solid"])
            self.lines2.append([self.t, out_2, 'green', 1, "solid"])

            for line in self.lines1:
                self.axis.plot(line[0], line[1], color=line[2], alpha=line[3], linestyle=line[4])
            for line in self.lines2:
                self.axis.plot(line[0], line[1], color=line[2], alpha=line[3], linestyle=line[4])

            st.session_state['lines1'] = self.lines1
            st.session_state['lines2'] = self.lines2

            # self.canvas.draw()

    def on_clear(self):
        while len(self.lines1) > 1:
            self.lines1.pop(0).remove()
        while len(self.lines2) > 1:
            self.lines2.pop(0).remove()
        self.canvas.draw()

    def on_random(self):
        self.do_graph = False
        self.slider_input_pos.setValue(np.random.uniform(0, 1) * 100)
        self.slider_input_neg.setValue(np.random.uniform(0, 1) * 100)
        self.slider_bias_pos.setValue(np.random.uniform(0, 1) * 50)
        self.slider_bias_neg.setValue(np.random.uniform(0, 1) * 50)
        self.do_graph = True
        self.slider_tcte.setValue(np.random.uniform(0, 1) * 50)


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
        st.subheader('ART1 Layer 2')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/19.svg")), unsafe_allow_html=True)
        st.markdown(
            "Adjust the inputs, biases and gain.\nThen click [Update] to see the layer respond.\n\nn2(1) is red, n2(2) is green.\n\nClick [Clear] to remove old responses.")
        clear = st.button("Clear")
        if clear:
            st.session_state['lines1'] = []
            st.session_state['lines2'] = []
        slider_input_pos = st.slider("Input a(1)", 0, 1, 0)
        slider_input_neg = st.slider("Input a(2)", 0, 1, 1)
        slider_bias_pos = st.slider("Bias b+", 0.0, 3.0, 1.0)
        slider_bias_neg = st.slider("Bias b-", 0.0, 3.0, 1.0)
        slider_tcte = st.slider("Transfer Function Gain", 0.1, 20.0, 10.0)
        st.subheader('*Chapter19*')
        st.markdown('---')



    app = ART1Layer2(slider_input_pos, slider_input_neg, slider_bias_pos, slider_bias_neg, slider_tcte)


    gap_cols = st.columns([1, 12, 1])
    with gap_cols[1]:
        st.pyplot(app.figure)
