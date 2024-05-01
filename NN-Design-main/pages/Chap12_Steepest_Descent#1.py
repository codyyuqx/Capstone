import numpy as np
from scipy.io import loadmat
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
import streamlit as st
import math
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
import time
from st_pages import hide_pages

from constants import pages_created
import os
import base64


class SteepestDescentBackprop1():
    def __init__(self, pair_param, x, y):
        self.pair_of_params = pair_param
        self.w_ratio, self.h_ratio, self.dpi = 1, 1, 96
        self.P = np.arange(-2, 2.1, 0.1).reshape(1, -1)
        self.W1, self.b1 = np.array([[10], [10]]), np.array([[-5], [5]])
        self.W2, self.b2 = np.array([[1, 1]]), np.array([[-1]])
        A1 = self.logsigmoid(np.dot(self.W1, self.P) + self.b1)
        self.T = self.logsigmoid(np.dot(self.W2, A1) + self.b2)
        self.lr, self.epochs = None, None

        self.figure = plt.figure()
        self.ax = self.figure.axes
        self.figure2 = go.Figure(layout=dict(height=250, margin=dict(l=0, r=0, b=0, t=0, pad=4)))

        # self.make_plot(1, (255, 380, 260, 260))
        # self.make_plot(2, (5, 380, 260, 260))
        self.w_ratio, self.h_ratio, self.dpi = 1, 1, 96

        self.axes = self.figure.add_subplot(1, 1, 1)
        self.path, = self.axes.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
        self.x_data, self.y_data = [], []
        self.init_point_1, = self.axes.plot([x], [y], "o", fillstyle="none", markersize=11, color="k")
        self.end_point_1, = self.axes.plot([], "o", fillstyle="none", markersize=11, color="k")
        # self.canvas.draw()
        # # self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        # self.ani, self.event = None, None
        # # self.axes2 = self.figure2.add_subplot(projection='3d')
        # self.axes2.view_init(30, -30)
        # self.axes2.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # # self.canvas2.mpl_connect("motion_notify_event", self.print_view)
        # self.axes2.set_title("Sum Sq. Error", fontdict={'fontsize': 10})

        self.pair_of_params = pair_param
        self.pair_params = [["W1(1, 1)", "W2(1, 1)"], ["W1(1, 1)", "b1(1)"], ["b1(1)", "b1(2)"]]
        plot_surface = self.plot_data()
        self.figure2.add_trace(plot_surface)

        self.x, self.y = x, y

        # self.make_combobox(1, ["W1(1, 1), W2(1, 1)", 'W1(1, 1), b1(1)', 'b1(1), b1(2)'],
        #                    (520, 370, 175, 50), self.change_pair_of_params,
        #                    "label_combo", "Pair of parameters", (545, 350, 150, 50))

        self.animation_speed = 0
        # self.canvas.draw()

    # def print_view(self, event):
    #     print(self.axes2.elev, self.axes2.azim)

    def change_pair_of_params(self, idx):
        if self.ani and self.ani.event_source:
            self.ani.event_source.stop()
        self.pair_of_params = idx + 1
        self.init_point_1.set_data([], [])
        self.end_point_1.set_data([], [])
        self.init_params()
        self.plot_data()

    def plot_data(self):
        self.x_data = []
        self.y_data = []
        self.path.set_data(self.x_data, self.y_data)
        while self.axes.collections:
            for collection in self.axes.collections:
                collection.remove()
        # while self.axes2.collections:
        #     for collection in self.axes2.collections:
        #         collection.remove()
        f_data = loadmat("SteepestDescent1/nndbp_new_{}.mat".format(self.pair_of_params))
        x1, y1 = np.meshgrid(f_data["x1"], f_data["y1"])
        self.axes.contour(x1, y1, f_data["E1"], list(f_data["levels"].reshape(-1)))
        # self.axes2.plot_surface(x1, y1, f_data["E1"], color="cyan")
        plot2_surface = go.Surface(x=x1, y=y1, z=f_data["E1"], colorscale='Viridis', name="Sum Sq. Error")

        if self.pair_of_params == 1:
            self.axes.scatter([self.W1[0, 0]], [self.W2[0, 0]], color="black", marker="+")
            self.axes.set_xlim(-5, 15)
            self.axes.set_ylim(-5, 15)

            self.axes.set_xticks([-5, 0, 5, 10])
            self.axes.set_yticks([-5, 0, 5, 10])
            # self.axes2.set_xticks([-5, 0, 5, 10])
            # self.axes2.set_yticks([-5, 0, 5, 10])
            # self.axes2.view_init(30, -30)
        elif self.pair_of_params == 2:
            self.axes.scatter([self.W1[0, 0]], [self.b1[0, 0]], color="black", marker="+")
            self.axes.set_xlim(-10, 30)
            self.axes.set_ylim(-20, 10)
            self.axes.set_xticks([-10, 0, 10, 20])
            self.axes.set_yticks([-20, -10, -5, 0])
            # self.axes2.set_xticks([-10, 0, 10, 20])
            # self.axes2.set_yticks([-20, -10, 0, 10, 20])
            # self.axes2.set_zticks([0, 1, 2])
            # self.axes2.view_init(30, -30)
        elif self.pair_of_params == 3:
            self.axes.scatter([self.b1[0, 0]], [self.b1[1, 0]], color="black", marker="+")
            self.axes.set_xlim(-10, 10)
            self.axes.set_ylim(-10, 10)
            self.axes.set_xticks([-10, -5, 0, 5])
            self.axes.set_yticks([-10, -5, 0, 5])
            # self.axes2.set_xticks([-10, -5, 0, 5])
            # self.axes2.set_yticks([-5, 0, 5, 10])
            # self.axes2.set_zticks([0, 1])
            # self.axes2.view_init(30, -30)
        self.axes.set_xlabel(self.pair_params[self.pair_of_params - 1][0], fontsize=12)
        self.axes.xaxis.set_label_coords(0.95, -0.025)
        self.axes.set_ylabel(self.pair_params[self.pair_of_params - 1][1], fontsize=12)
        self.axes.yaxis.set_label_coords(-0.025, 0.95)
        return plot2_surface
        # self.axes2.tick_params(pad=0)
        # self.axes2.set_xlabel(self.pair_params[self.pair_of_params - 1][0], labelpad=1)
        # self.axes2.set_ylabel(self.pair_params[self.pair_of_params - 1][1], labelpad=1)
        # self.canvas.draw()
        # self.canvas2.draw()

    def animate_init(self):
        self.end_point_1.set_data([], [])
        self.path.set_data(self.x_data, self.y_data)
        return self.path, self.end_point_1

    def on_animate(self, idx):

        n1 = np.dot(self.W1, self.P) + self.b1
        a1 = self.logsigmoid(n1)
        n2 = np.dot(self.W2, a1) + self.b2
        a2 = self.logsigmoid(n2)

        e = self.T - a2

        D2 = a2 * (1 - a2) * e
        D1 = a1 * (1 - a1) * np.dot(self.W2.T, D2)
        dW1 = np.dot(D1, self.P.T) * self.lr
        db1 = np.dot(D1, np.ones((D1.shape[1], 1))) * self.lr
        dW2 = np.dot(D2, a1.T) * self.lr
        db2 = np.dot(D2, np.ones((D2.shape[1], 1))) * self.lr

        if self.pair_of_params == 1:
            self.W1[0, 0] += dW1[0, 0]
            self.W2[0, 0] += dW2[0, 0]
            self.x, self.y = self.W1[0, 0], self.W2[0, 0]
        elif self.pair_of_params == 2:
            self.W1[0, 0] += dW1[0, 0]
            self.b1[0, 0] += db1[0, 0]
            self.x, self.y = self.W1[0, 0], self.b1[0, 0]
        elif self.pair_of_params == 3:
            self.b1[0, 0] += db1[0, 0]
            self.b1[1, 0] += db1[1, 0]
            self.x, self.y = self.b1[0, 0], self.b1[1, 0]

        if idx == self.epochs - 1 or np.linalg.norm([dW1[0, 0], dW2[0, 0], db1[0, 0], db2[0, 0]]) < 0.01:
            self.end_point_1.set_data(self.x_data[-1], self.y_data[-1])

        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.path.set_data(self.x_data, self.y_data)
        return self.path, self.end_point_1

    # def on_mouseclick(self, event):
    #     self.init_params()
    #     self.event = event
    #     if self.ani and self.ani.event_source:
    #         self.ani.event_source.stop()
    #     self.path.set_data([], [])
    #     self.x_data, self.y_data = [], []
    #     self.init_point_1.set_data([event.xdata], [event.ydata])
    #     self.canvas.draw()
    #     self.run_animation(event)

    def run_animation(self):

        self.x_data, self.y_data = [self.x], [self.y]
        if self.pair_of_params == 1:
            self.W1[0, 0], self.W2[0, 0] = self.x, self.y
            self.lr, self.epochs = 3.5, 600
        elif self.pair_of_params == 2:
            self.W1[0, 0], self.b1[0, 0] = self.x, self.y
            self.lr, self.epochs = 50, 300
        elif self.pair_of_params == 3:
            self.b1[0, 0], self.b1[1, 0] = self.x, self.y
            self.lr, self.epochs = 25, 60
        self.ani = FuncAnimation(self.figure, self.on_animate, init_func=self.animate_init, frames=self.epochs,
                                 interval=self.animation_speed, repeat=False, blit=True)

    def init_params(self):
        self.W1, self.b1 = np.array([[10.], [10.]]), np.array([[-5.], [5.]])
        self.W2, self.b2 = np.array([[1., 1.]]), np.array([[-1.]])

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


    hide_pages(pages_created)
    image_path = 'media'

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('Steepest Descent Algorithm')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')
    st.markdown('---')

    st.text('')
    st.markdown(load_svg_2(get_image_path("Figures/nnd12_1.svg")), unsafe_allow_html=True)
    #st.image("SteepestDescent1/nnd12_1.svg", width=450)
    cols1 = st.columns(3)
    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/12.svg")), unsafe_allow_html=True)
        st.markdown('''
        Use the popup menu beloww to select the network prarameters to train with backpropagation..

        The corresponding error surface andc ontour are shown below.

        Click in the contour graph to start the steepest descent algorithm.
        ''')
        pair_param = st.selectbox("Select pair of parameters",
                                  ['W1(1, 1), W2(1, 1)', 'W1(1, 1), b1(1)', 'b1(1), b1(2)'], 0)
        pair_param_dict = {'W1(1, 1), W2(1, 1)': 1, 'W1(1, 1), b1(1)': 2, 'b1(1), b1(2)': 3}
        pair_param = pair_param_dict[pair_param]

        #st.markdown('---')

        x = st.number_input("Enter x", value=4)

        y = st.number_input("Enter y", value=4)
        st.subheader('*Chapter12*')
        st.markdown('---')

    app = SteepestDescentBackprop1(pair_param, x, y)
    app.init_params()

    run = st.button("Run")
    cols = st.columns(2)
    with cols[0]:
        the_plot1 = st.pyplot(plt)

    # app.figure.axes[1].set_xticks([])

    app.animate_init()

    if run:
        app.init_params()
        app.animate_init()
        app.run_animation()
        for i in range(app.epochs):
            # app.figure.axes[0].set/_xticks([])

            if len(app.end_point_1.get_data()[0]):
                break

            app.on_animate(i)
            if i % 20 == 0:
                the_plot1.pyplot(plt)

    with cols[1]:
        st.plotly_chart(app.figure2, use_container_width=True)
