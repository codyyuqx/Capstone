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


class ART1Layer1():
    def __init__(self, slider_input_pos, slider_input_neg, slider_bias_pos, slider_bias_neg, w_11, w_12, w_21, w_22):
        self.t = np.arange(0, 0.2, 0.001)

        self.bp, self.bn, self.e = 1, 0, 0.1

        # self.make_plot(1, (10, 90, 500, 450))
        self.figure = plt.figure(figsize=(8, 7))
        self.figure.subplots_adjust(left=0.15, right=0.95, bottom=0.125, top=0.9)
        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.2)
        self.axis.set_ylim(-0.5, 0.5)
        self.axis.plot([0] * 10, np.linspace(-0.5, 0.5, 10), color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Net inputs n1(1), n1(2)")
        self.axis.set_title("Response")
        self.axis.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        self.axis.plot(np.linspace(0, 0.2, 100), [0] * 100, linestyle="dashed", linewidth=0.5, color="gray")
        # self.lines1, self.lines2 = [], []
        self.lines1 = st.session_state['lines1'] if 'lines1' in st.session_state else []
        self.lines2 = st.session_state['lines2'] if 'lines2' in st.session_state else []

        self.slider_input_pos = slider_input_pos
        self.slider_input_neg = slider_input_neg
        self.slider_bias_pos = slider_bias_pos
        self.slider_bias_neg = slider_bias_neg
        self.w_11 = w_11
        self.w_12 = w_12
        self.w_21 = w_21
        self.w_22 = w_22

        # self.paint_latex_string("latex_W21", "$W2:1 =$", 16, (30, 510, 250, 200))
        # self.paint_latex_string("latex_W22", "$[$", 45, (215, 510, 250, 200))
        # self.paint_latex_string("latex_W23", "$]$", 45, (335, 510, 250, 200))

        # self.make_label("label_a", "W2:1 =", (145, 503, 200, 200), font_size=25)

        # self.make_label("label_a1", "[   ]", (226, 494, 500, 200), font_size=100)
        # self.label_a.setStyleSheet("color:black")
        # self.label_a1.setStyleSheet("color:black")
        # self.make_input_box("w_11", "1", (239, 530, 60, 100))
        # self.make_input_box("w_12", "1", (295, 530, 60, 100))
        # self.make_input_box("w_21", "0", (239, 580, 60, 100))
        # self.make_input_box("w_22", "1", (295, 580, 60, 100))

        # self.make_button("clear_button", "Clear", (self.x_chapter_button, 575, self.w_chapter_button, self.h_chapter_button), self.on_clear)
        # self.make_button("random_button", "Update", (self.x_chapter_button, 605, self.w_chapter_button, self.h_chapter_button), self.graph)

        self.graph()

    def layer1(self, t, y):
        return [(-y[0] + (self.bp - y[0]) * (self.p[0, 0] + self.W2[0, 1]) - (y[0] + self.bn)) / 0.1,
                (-y[1] + (self.bp - y[1]) * (self.p[1, 0] + self.W2[1, 1]) - (y[1] + self.bn)) / 0.1]

    def slide(self):
        self.pp = self.slider_input_pos.value()
        self.pn = self.slider_input_neg.value()
        self.label_input_pos.setText("Input p(1): " + str(self.pp))
        self.label_input_neg.setText("Input p(2): " + str(self.pn))
        self.bp = self.slider_bias_pos.value() / 10
        self.bn = self.slider_bias_neg.value() / 10
        self.label_bias_pos.setText("Bias b+: " + str(round(self.bp, 2)))
        self.label_bias_neg.setText("Bias b-: " + str(round(self.bn, 2)))

    def graph(self):
        self.pp = self.slider_input_pos
        self.pn = self.slider_input_neg
        # self.label_input_pos.setText("Input p(1): " + str(self.pp))
        # self.label_input_neg.setText("Input p(2): " + str(self.pn))
        self.bp = self.slider_bias_pos
        self.bn = self.slider_bias_neg
        # self.label_bias_pos.setText("Bias b+: " + str(round(self.bp, 2)))
        # self.label_bias_neg.setText("Bias b-: " + str(round(self.bn, 2)))
        w11, w12 = self.w_11, self.w_12
        w21, w22 = self.w_21, self.w_22
        for idx, param in enumerate([w11, w12, w21, w22]):
            if param not in [0, 1]:
                if idx == 0:
                    w11 = 0
                    self.w_11.setText("0")
                elif idx == 1:
                    w12 = 0
                    self.w_12.setText("0")
                elif idx == 2:
                    w21 = 0
                    self.w_21.setText("0")
                else:
                    w22 = 0
                    self.w_22.setText("0")
        self.W2 = np.array([[w11, w12], [w21, w22]])
        self.p = np.array([[self.pp], [self.pn]])
        r1 = ode(self.layer1).set_integrator("vode")
        r1.set_initial_value(np.array([0, 0]), 0)
        t1 = 0.2
        dt = 0.001
        out_1, out_2 = [], []
        while r1.successful() and r1.t < t1:
            out = r1.integrate(r1.t + dt)
            out_1.append(out[0].item())
            out_2.append(out[1].item())
        out_1[0], out_2[0] = 0, 0
        while len(self.lines1) > 1:
            self.lines1.pop(0)
        while len(self.lines2) > 1:
            self.lines2.pop(0)
        for line in self.lines1:
            # line.set_color("gray")
            line[3] = 0.2

        for line in self.lines2:
            # line.set_color("gray")
            line[3] = 0.2

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
        # st.text('')
        # st.text('')
        st.subheader('ART1 Layer 1')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/19.svg")), unsafe_allow_html=True)
        st.markdown(
            "Adjust the inputs, biases and weights.\nThen click [Update] to see the layer respond.\n\nn1(1) is red, n1(2) is green.\n\nClick [Clear] to remove old responses.")
        clear = st.button("Clear")
        if clear:
            st.session_state['lines1'] = []
            st.session_state['lines2'] = []
        slider_input_pos = st.slider("Input p(1)", 0, 1, 0)
        slider_input_neg = st.slider("Input p(2)", 0, 1, 1)
        slider_bias_pos = st.slider("Bias b+", 0.0, 3.0, 1.0)
        slider_bias_neg = st.slider("Bias b-", 0.0, 3.0, 1.5)
        st.subheader("W2:1 = ")
        ww_cols = st.columns(2)
        with ww_cols[0]:
            w_11 = st.number_input("w_11", 0, 1, 1)
            w_21 = st.number_input("w_21", 0, 1, 0)
        with ww_cols[1]:
            w_12 = st.number_input("w_12", 0, 1, 1)
            w_22 = st.number_input("w_22", 0, 1, 1)
        st.subheader('*Chapter19*')
        st.markdown('---')

    # w_cols = st.columns([2, 2, 2, 2])
    # with w_cols[1]:
    #     st.title('')
    #     st.title('')
    #     st.subheader("W2:1 = ")
    #
    # with w_cols[2]:
    #
    #     ww_cols = st.columns(2)
    #     with ww_cols[0]:
    #         w_11 = st.number_input("w_11", 0, 1, 1)
    #         w_21 = st.number_input("w_21", 0, 1, 0)
    #     with ww_cols[1]:
    #         w_12 = st.number_input("w_12", 0, 1, 1)
    #         w_22 = st.number_input("w_22", 0, 1, 1)

    app = ART1Layer1(slider_input_pos, slider_input_neg, slider_bias_pos, slider_bias_neg, w_11, w_12, w_21, w_22)

    gap_col = st.columns([1, 15, 1])
    with gap_col[1]:
        st.pyplot(app.figure)
