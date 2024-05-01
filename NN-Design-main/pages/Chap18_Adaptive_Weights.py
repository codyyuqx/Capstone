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


class AdaptiveWeights():
    def __init__(self, n_11, n_12, n_21, n_22, learning_rule):

        self.t = np.arange(0, 2, 0.01)

        self.figure = plt.figure(figsize=(8, 6))
        self.figure.subplots_adjust(left=0.15, right=0.95, bottom=0.125, top=0.9)

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(-0.1, 2.1)
        self.axis.set_ylim(-0.1, 1.1)
        for i in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
            self.axis.plot([i] * 10, np.linspace(-0.1, 1.1, 10), color="black", linestyle="--", linewidth=0.5)
            if i in [0, 0.4, 0.8, 1.2, 1.6, 2]:
                if i != 2:
                    self.axis.text(i + 0.0175, -0.075, "1st")
            else:
                self.axis.text(i + 0.015, -0.075, "2nd")
        self.axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Weights W2")
        self.axis.set_title("Learning")
        self.axis.set_xticks([0, 0.5, 1, 1.5, 2])
        # self.lines1, self.lines2, self.lines3, self.lines4 = [], [], [], []

        self.lines1 = st.session_state['lines1'] if 'lines1' in st.session_state else []
        self.lines2 = st.session_state['lines2'] if 'lines2' in st.session_state else []
        self.lines3 = st.session_state['lines3'] if 'lines3' in st.session_state else []
        self.lines4 = st.session_state['lines4'] if 'lines4' in st.session_state else []

        # self.paint_latex_string("latex_n11", "$1st$", 10, (20, 510, 500, 200))
        # self.paint_latex_string("latex_n12", "$n1 =$", 10, (75, 510, 500, 200))
        # self.paint_latex_string("latex_n13", "$[$", 40, (140, 510, 500, 200))
        # self.paint_latex_string("latex_n14", "$]$", 40, (200, 510, 500, 200))
        # self.make_label("label_a", "1st n1 =", (60, 503, 500, 200), font_size=25)
        # self.make_label("label_a1", "[ ]", (145, 494, 500, 200), font_size=100)
        # self.label_a.setStyleSheet("color:black")
        # self.label_a1.setStyleSheet("color:black")
        # self.make_input_box("n_11", "0.9", (161, 530, 60, 100))
        # self.make_input_box("n_12", "0.45", (161, 577, 60, 100))
        self.n_11, self.n_12 = n_11, n_12
        self.n_21, self.n_22 = n_21, n_22

        # self.paint_latex_string("latex_n21", "$2nd$", 10, (270, 510, 500, 200))
        # self.paint_latex_string("latex_n22", "$n1 =$", 10, (335, 510, 500, 200))
        # self.paint_latex_string("latex_n23", "$[$", 40, (400, 510, 500, 200))
        # self.paint_latex_string("latex_n24", "$]$", 40, (460, 510, 500, 200))
        # self.make_label("label_aa", "2nd n1 =", (320, 503, 500, 200), font_size=25)
        # self.make_label("label_aa1", "[ ]", (410, 494, 500, 200), font_size=100)
        # self.label_aa.setStyleSheet("color:black")
        # self.label_aa1.setStyleSheet("color:black")
        # self.make_input_box("n_21", "0.45", (426, 530, 60, 100))
        # self.make_input_box("n_22", "0.90", (426, 577, 60, 100))

        self.comboBox1_functions_str = ['Instar', 'Hebb']

        self.rule = self.comboBox1_functions_str.index(learning_rule) + 1

        self.graph()

    # def paintEvent(self, event):
    #     super(AdaptiveWeights, self).paintEvent(event)
    #     painter = QtGui.QPainter()
    #     painter.begin(self)
    #     pen = QtGui.QPen(QtGui.QColor("black"), 2, QtCore.Qt.PenStyle.SolidLine)
    #     painter.setPen(pen)
    #     self.paint_bracket(painter, 161, 559, 646, 60)
    #     self.paint_bracket(painter, 426, 559, 646, 60)
    #     painter.end()

    def change_learning_rule(self, idx):
        self.rule = idx + 1
        # self.graph()

    def adapt(self, t, w):
        if np.fix(t / 0.2) % 2 == 0:
            n1 = self.n1
            n2 = np.array([[1], [0]])
        else:
            n1 = self.n2
            n2 = np.array([[0], [1]])
        w = w.reshape((2, 2))
        if self.rule == 1:
            wprime = (4 * np.dot(n2, np.ones((1, 2)))) * (np.dot(np.ones((2, 1)), n1.T) - w)
        else:
            wprime = 4 * np.dot(n2, n1.T) - 2 * w
        return wprime.reshape(-1)

    def graph(self):
        n11, n12 = float(self.n_11), float(self.n_12)
        n21, n22 = float(self.n_21), float(self.n_22)
        self.n1, self.n2 = np.array([[n11], [n12]]), np.array([[n21], [n22]])
        r1 = ode(self.adapt).set_integrator("zvode")
        r1.set_initial_value(np.zeros((4,)), 0)
        t1 = 2
        dt = 0.01
        out_11, out_21, out_12, out_22 = [], [], [], []
        while r1.successful() and r1.t < t1:
            out = r1.integrate(r1.t + dt)
            out_11.append(out[0].item())
            out_12.append(out[1].item())
            out_21.append(out[2].item())
            out_22.append(out[3].item())
        out_11[0], out_12[0], out_21[0], out_22[0] = 0, 0, 0, 0
        while len(self.lines1) > 1:
            self.lines1.pop(0)
        while len(self.lines2) > 1:
            self.lines2.pop(0)
        while len(self.lines3) > 1:
            self.lines3.pop(0)
        while len(self.lines4) > 1:
            self.lines4.pop(0)
        for line in self.lines1:
            line[3] = 0.2
        for line in self.lines2:
            line[3] = 0.2
        for line in self.lines3:
            line[3] = 0.2
        for line in self.lines4:
            line[3] = 0.2
        # self.lines1.append(self.axis.plot(self.t, out_11, color="red")[0])
        # self.lines2.append(self.axis.plot(self.t, out_12, color="red", linestyle="dashed")[0])
        # self.lines3.append(self.axis.plot(self.t, out_21, color="green")[0])
        # self.lines4.append(self.axis.plot(self.t, out_22, color="green", linestyle="dashed")[0])
        self.lines1.append([self.t, out_11, 'red', 1, "solid"])
        self.lines2.append([self.t, out_12, 'red', 1, "dashed"])
        self.lines3.append([self.t, out_21, 'green', 1, "solid"])
        self.lines4.append([self.t, out_22, 'green', 1, "dashed"])

        for line in self.lines1:
            self.axis.plot(line[0], line[1], color=line[2], alpha=line[3], linestyle=line[4])
        for line in self.lines2:
            self.axis.plot(line[0], line[1], color=line[2], alpha=line[3], linestyle=line[4])
        for line in self.lines3:
            self.axis.plot(line[0], line[1], color=line[2], alpha=line[3], linestyle=line[4])
        for line in self.lines4:
            self.axis.plot(line[0], line[1], color=line[2], alpha=line[3], linestyle=line[4])

        st.session_state['lines1'] = self.lines1
        st.session_state['lines2'] = self.lines2
        st.session_state['lines3'] = self.lines3
        st.session_state['lines4'] = self.lines4

        # self.canvas.draw()

    def on_clear(self):
        while len(self.lines1) > 1:
            self.lines1.pop(0).remove()
        while len(self.lines2) > 1:
            self.lines2.pop(0).remove()
        while len(self.lines3) > 1:
            self.lines3.pop(0).remove()
        while len(self.lines4) > 1:
            self.lines4.pop(0).remove()
        # self.canvas.draw()


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
        st.subheader('Adaptive Learning')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/18.svg")), unsafe_allow_html=True)
        st.markdown(
            "Edit the two input vectors and click [Update] to see the network learn them.\n\n W2(1,1) - solid red\n\nW2(1,2) - broken red\n\nW2(2,1) - solid green\n\nW2(2,2) - broken green\n\nClick [Clear] to remove old responses.")
        learning_rule = st.selectbox("Learning Rule", ['Instar', 'Hebb'])
        clear = st.button("Clear")
        st.subheader('*Chapter19*')
        st.markdown('---')

    n1_cols = st.columns([4, 4, 1, 4, 4])
    with n1_cols[0]:
        # white text
        st.title('')
        st.title('')
        st.subheader("1st n1 =")

    with n1_cols[1]:
        n_11 = st.number_input("n1_1", 0.0, 100.0, 0.9)
        n_12 = st.number_input("n1_2", 0.0, 100.0, 0.45)

    with n1_cols[3]:
        st.title('')
        st.title('')

        st.subheader("2nd n1 =")

    with n1_cols[4]:
        n_21 = st.number_input("n2_1", 0.0, 100.0, 0.45)
        n_22 = st.number_input("n2_2", 0.0, 100.0, 0.9)

    st.title('')

    app = AdaptiveWeights(n_11, n_12, n_21, n_22, learning_rule)

    st.pyplot(app.figure)