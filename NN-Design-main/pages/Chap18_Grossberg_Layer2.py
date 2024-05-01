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
import base64
import os

from scipy.integrate import ode

font = {'size': 10}

matplotlib.rc('font', **font)


class GrossbergLayer2():
    def __init__(self, slider_input_pos, slider_input_neg, w_11, w_12, w_21, w_22, func, clear, random):
        self.t = np.arange(0, 0.51, 0.01)

        # self.make_plot(1, (25, 90, 450, 450))
        self.figure = plt.figure(figsize=(5, 5))

        self.bp, self.bn, self.e = 1, 0, 0.1

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.set_xlim(0, 0.51)
        self.axis.set_ylim(0, 1)
        self.axis.plot([0] * 10, np.linspace(0, 1, 10), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot([0.25] * 10, np.linspace(0, 1, 10), color="black", linestyle="--", linewidth=0.2)
        self.axis.plot(np.linspace(0, 0.5, 10), [0] * 10, color="black", linestyle="--", linewidth=0.2)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Net inputs n2(1), n2(2)")
        self.axis.set_title("Response")

        self.lines1 = st.session_state['lines1'] if 'lines1' in st.session_state else []
        self.lines2 = st.session_state['lines2'] if 'lines2' in st.session_state else []

        self.slider_input_pos = slider_input_pos
        self.slider_input_neg = slider_input_neg
        self.w_11 = w_11
        self.w_12 = w_12
        self.w_21 = w_21
        self.w_22 = w_22

        if clear:
            # self.do_graph = False
            st.session_state['lines1'] = []
            st.session_state['lines2'] = []
            self.lines1 = []
            self.lines2 = []

        self.comboBox1_functions = [self.f2, self.purelin, self.f3, self.f4]
        self.comboBox1_functions_str = ['(10n^2)/(1 + n^2)', "purelin", '10n^2', '1 - exp(-n)']
        self.func1 = self.comboBox1_functions[func_list.index(func)]

        # self.make_button("clear_button", "Clear", (self.x_chapter_button, 560, self.w_chapter_button, self.h_chapter_button), self.on_clear)
        # self.make_button("random_button", "Random", (self.x_chapter_button, 590, self.w_chapter_button, self.h_chapter_button), self.on_random)
        # self.make_button("run_button", "Update", (self.x_chapter_button, 530, self.w_chapter_button, self.h_chapter_button), self.graph)

        self.do_graph = True

        self.graph()

    @staticmethod
    def f2(n):
        return 10 * n ** 2 / (1 + n ** 2)

    @staticmethod
    def f3(n):
        return 10 * n ** 2

    @staticmethod
    def f4(n):
        return 1 - np.exp(-n)

    def change_transfer_function(self, idx):
        self.func1 = self.comboBox1_functions[idx]
        self.graph()

    def layer2(self, t, y):
        i1 = np.dot(self.W2[0, :], self.p).item()
        i2 = np.dot(self.W2[1, :], self.p).item()
        y = np.array([[y[0]], [y[1]]])
        a = self.func1(y)
        y_out = np.zeros(y.shape)
        y_out[0, 0] = (-y[0, 0] + (self.bp - y[0, 0]) * (a[0, 0] + i1) - (y[0, 0] + self.bn) * a[1, 0]) / self.e
        y_out[1, 0] = (-y[1, 0] + (self.bp - y[1, 0]) * (a[1, 0] + i2) - (y[1, 0] + self.bn) * a[0, 0]) / self.e
        return y_out

    def graph(self):
        if self.do_graph:
            self.pp = self.slider_input_pos
            self.pn = self.slider_input_neg
            # self.label_input_pos.setText("Input a1(1): " + str(round(self.pp, 2)))
            # self.label_input_neg.setText("Input a1(2): " + str(round(self.pn, 2)))
            w11, w12 = float(self.w_11), float(self.w_12)
            w21, w22 = float(self.w_21), float(self.w_22)
            self.W2 = np.array([[w11, w12], [w21, w22]])
            self.p = np.array([[self.pp], [self.pn]])
            r1 = ode(self.layer2).set_integrator("zvode")
            r1.set_initial_value(np.array([[0], [0]]), 0)
            t1 = 0.26
            dt = 0.01
            out_11, out_21 = [], []
            while r1.successful() and r1.t < t1:
                out = r1.integrate(r1.t + dt)
                out_11.append(out[0, 0].item())
                out_21.append(out[1, 0].item())
            self.p = np.array([[0], [0]])
            r2 = ode(self.layer2).set_integrator("zvode")
            r2.set_initial_value(np.array([[out_11[-1]], [out_21[-1]]]), 0.26)
            t2 = 0.51
            out_12, out_22 = [], []
            while r2.successful() and r2.t < t2:
                out = r2.integrate(r2.t + dt)
                out_12.append(out[0, 0].item())
                out_22.append(out[1, 0].item())
            out_1, out_2 = out_11 + out_12, out_21 + out_22
            out_1[0], out_2[0] = 0, 0
            while len(self.lines1) > 1:
                self.lines1.pop(0)
            while len(self.lines2) > 1:
                self.lines2.pop(0)
            for line in self.lines1:
                # line.set_color("gray")
                # line.set_alpha(0.2)
                line[3] = 0.2
            for line in self.lines2:
                # line.set_color("gray")
                # line.set_alpha(0.2)
                line[3] = 0.2

            # self.lines1.append(self.axis.plot(self.t, out_1, color="red")[0])
            # self.lines2.append(self.axis.plot(self.t, out_2, color="green")[0])
            self.lines1.append([self.t, out_1, "red", 1])
            self.lines2.append([self.t, out_2, "green", 1])

            for line in self.lines1:
                self.axis.plot(line[0], line[1], color=line[2], alpha=line[3])

            for line in self.lines2:
                self.axis.plot(line[0], line[1], color=line[2], alpha=line[3])

            st.session_state['lines1'] = self.lines1
            st.session_state['lines2'] = self.lines2

            # self.canvas.draw()

    def on_clear(self):
        while len(self.lines1) > 1:
            self.lines1.pop(0).remove()
        while len(self.lines2) > 1:
            self.lines2.pop(0).remove()
        # self.canvas.draw()

    def on_random(self):
        self.do_graph = False
        self.slider_input_pos.setValue(round(np.random.uniform(0, 1) * 100))
        self.do_graph = True
        self.slider_input_neg.setValue(round(np.random.uniform(0, 1) * 100))

    @staticmethod
    def logsigmoid(n):
        return 1 / (1 + np.exp(-n))

    @staticmethod
    def logsigmoid_stable(n):
        n = np.clip(n, -100, 100)
        return 1 / (1 + np.exp(-n))

    @staticmethod
    def logsigmoid_der(n):
        return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))

    @staticmethod
    def purelin(n):
        return n

    @staticmethod
    def purelin_der(n):
        return np.array([1]).reshape(n.shape)

    @staticmethod
    def lin_delta(a, d=None, w=None):
        na, ma = a.shape
        if d is None and w is None:
            return -np.kron(np.ones((1, ma)), np.eye(na))
        else:
            return np.dot(w.T, d)

    @staticmethod
    def log_delta(a, d=None, w=None):
        s1, _ = a.shape
        if d is None and w is None:
            return -np.kron((1 - a) * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
        else:
            return (1 - a) * a * np.dot(w.T, d)

    @staticmethod
    def tan_delta(a, d=None, w=None):
        s1, _ = a.shape
        if d is None and w is None:
            return -np.kron(1 - a * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
        else:
            return (1 - a * a) * np.dot(w.T, d)

    @staticmethod
    def marq(p, d):
        s, _ = d.shape
        r, _ = p.shape
        return np.kron(p.T, np.ones((1, s))) * np.kron(np.ones((1, r)), d.T)

    @staticmethod
    def compet(n, axis=None):
        if axis is not None:
            max_idx = np.argmax(n, axis=axis)
            out = np.zeros(n.shape)
            for i in range(out.shape[1]):
                out[max_idx[i], i] = 1
            return out
        else:
            max_idx = np.argmax(n)
            out = np.zeros(n.shape)
            out[max_idx] = 1
            return out

    @staticmethod
    def poslin(n):
        return n * (n > 0)

    @staticmethod
    def hardlim(x):
        if x < 0:
            return 0
        else:
            return 1

    @staticmethod
    def hardlims(x):
        if x < 0:
            return -1
        else:
            return 1

    @staticmethod
    def satlin(x):
        if x < 0:
            return 0
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    def satlins(x):
        if x < -1:
            return 0
        elif x < 1:
            return x
        else:
            return 1

    @staticmethod
    def logsig(x):
        return 1 / (1 + math.e ** (-x))

    @staticmethod
    def tansig(x):
        return 2 / (1 + math.e ** (-2 * x)) - 1

    def nndtansig(self, x):
        self.a = self.tansig(x)


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')

    hide_pages(pages_created)

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([4, 2])
    with header_cols[0]:
        st.text('')
        st.text('')
        st.subheader('Grossberg Layer 2')
        # st.subheader('')

    with header_cols[1]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown("Use the slide bars to adjust the inputs, biases and the time constant (eps).  ")
        st.markdown("Output n2(1) is red, output n2(2) is green.  Click [Clear] to remove old responses.")
        st.markdown("Click [Random] to get random inputs.")

        clear = st.button('Clear', use_container_width=True)
        random = st.button('Random', use_container_width=True)

    if 'slider_input_pos' not in st.session_state:
        st.session_state['slider_input_pos'] = 0.7
        st.session_state['slider_input_neg'] = 0.7
        st.session_state['w_11'] = 0.9
        st.session_state['w_12'] = 0.45
        st.session_state['w_21'] = 0.45
        st.session_state['w_22'] = 0.9

    if random:
        st.session_state['slider_input_pos'] = round(np.random.uniform(0, 1) * 100) / 10
        st.session_state['slider_input_neg'] = round(np.random.uniform(0, 1) * 100) / 10
        st.session_state['w_11'] = round(np.random.uniform(0, 1) * 100) / 100
        st.session_state['w_12'] = round(np.random.uniform(0, 1) * 100) / 100
        st.session_state['w_21'] = round(np.random.uniform(0, 1) * 100) / 100
        st.session_state['w_22'] = round(np.random.uniform(0, 1) * 100) / 100

    with st.sidebar:

        slider_input_pos = st.slider('Input p(1)', 0.0, 10.0, st.session_state['slider_input_pos'])
        slider_input_neg = st.slider('Input p(2)', 0.0, 10.0, st.session_state['slider_input_neg'])

        func_list = ['(10n^2)/(1 + n^2)', "purelin", '10n^2', '1 - exp(-n)']
        func = st.selectbox('Transfer function', func_list)

    w_gap = st.columns([2, 2, 2])
    with w_gap[1]:
        w_cols_1 = st.columns(2)
        with w_cols_1[0]:
            w_11 = st.number_input('W11', 0.0, 10.0, st.session_state['w_11'])
            w_21 = st.number_input('W21', 0.0, 10.0, st.session_state['w_21'])
        with w_cols_1[1]:
            w_12 = st.number_input('W12', 0.0, 10.0, st.session_state['w_12'])
            w_22 = st.number_input('W22', 0.0, 10.0, st.session_state['w_22'])

    grossberg = GrossbergLayer2(slider_input_pos, slider_input_neg, w_11, w_12, w_21, w_22, func, clear, random)

    # with plot_area:
    gap_cols = st.columns([1, 6, 1])
    with gap_cols[1]:
        st.pyplot(grossberg.figure)