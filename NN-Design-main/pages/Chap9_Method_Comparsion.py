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

font = {'size': 10}

matplotlib.rc('font', **font)


class ComparisonOfMethods():
    def __init__(self, xdata, ydata):

        x, y = np.linspace(-2, 0 + (4 / 31 * 17), 100, endpoint=False), np.linspace(-2, 0 + (4 / 31 * 17), 200,
                                                                                    endpoint=False)
        X, Y = np.meshgrid(x, y)
        self.a, self.b, c = np.array([[2, 1], [1, 2]]), np.array([0, 0]), 0
        F = (self.a[0, 0] * X ** 2 + self.a[0, 1] + self.a[1, 0] * X * Y + self.a[1, 1] * Y ** 2) / 2 \
            + self.b[0] * X + self.b[1] * Y + c

        # self.make_plot(1, (115, 100, 290, 290))
        # self.make_plot(2, (115, 385, 290, 290))
        self.figure = plt.figure(figsize=(4, 4))
        self.figure2 = plt.figure(figsize=(4, 4))

        self.event, self.ani_1, self.ani_2 = None, None, None
        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Steepest Descent Path", fontdict={'fontsize': 10})
        self.axes_1.contour(X, Y, F, levels=[0.5, 1, 2, 4, 6, 8])
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes_1.set_yticks([-2, -1, 0, 1, 2])
        self.path_1, = self.axes_1.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Gradient Descent Path")
        self.init_point_1, = self.axes_1.plot([], "o", fillstyle="none", markersize=11, color="k")
        self.x_data_1, self.y_data_1 = [], []
        # self.canvas.mpl_connect('button_press_event', self.on_mouseclick)
        # self.canvas.draw()

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Conjugate Gradient Path")
        self.axes_2.contour(X, Y, F, levels=[0.5, 1, 2, 4, 6, 8])
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_yticks([-2, -1, 0, 1, 2])
        self.path_2, = self.axes_2.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Conjugate Gradient Path")
        self.init_point_2, = self.axes_2.plot([], "o", fillstyle="none", markersize=11, color="k")
        self.x_data_2, self.y_data_2 = [], []
        # self.canvas2.mpl_connect('button_press_event', self.on_mouseclick)
        # self.canvas2.draw()

        self.animation_speed = 500
        self.on_mouseclick(xdata, ydata)
        # self.run_animation(xdata, ydata)
        # self.make_slider("slider_anim_speed", QtCore.Qt.Orientation.Horizontal, (0, 6), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 2,
        #                  (self.x_chapter_usual, 380, self.w_chapter_slider, 100), self.slide, "label_anim_speed", "Animation Delay: 200 ms")

    # def slide(self):
    #     self.animation_speed = int(self.slider_anim_speed.value()) * 100
    #     self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
    #     if self.x_data_1:
    #         if self.ani_1:
    #             self.ani_1.event_source.stop()
    #             self.ani_2.event_source.stop()
    #         self.path_1.set_data([], [])
    #         self.path_2.set_data([], [])
    #         self.x_data_1, self.y_data_1 = [self.x_data_1[0]], [self.y_data_1[0]]
    #         self.x_data_2, self.y_data_2 = [self.x_data_2[0]], [self.y_data_2[0]]
    #         self.canvas.draw()
    #         self.canvas2.draw()
    #         self.run_animation(self.event)

    def on_mouseclick(self, xdata, ydata):
        self.path_1.set_data([], [])
        self.path_2.set_data([], [])
        self.x_data_1, self.y_data_1 = [], []
        self.x_data_2, self.y_data_2 = [], []
        self.init_point_1.set_data([xdata], [ydata])
        self.init_point_2.set_data([xdata], [ydata])
        # self.canvas.draw()
        # self.canvas2.draw()
        self.run_animation(xdata, ydata)

    def animate_init_1(self):
        self.path_1, = self.axes_1.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Gradient Descent Path")
        return self.path_1,

    def animate_init_2(self):
        self.path_2, = self.axes_2.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Conjugate Gradient Path")
        return self.path_2,

    def on_animate_1(self, idx):
        gradient = np.dot(self.a, np.array([self.x_1, self.y_1])) + self.b.T
        p_g = -gradient
        hess = self.a
        lr = -np.dot(gradient, p_g.T) / np.dot(p_g.T, np.dot(hess, p_g))
        self.x_1 -= lr * gradient[0]
        self.y_1 -= lr * gradient[1]
        # lr = 0.07
        # gradient = np.dot(a, np.array([self.x_1, self.y_1])) + b.T
        # self.x_1 -= lr * gradient[0]
        # self.y_1 -= lr * gradient[1]
        self.x_data_1.append(self.x_1)
        self.y_data_1.append(self.y_1)
        self.path_1.set_data(self.x_data_1, self.y_data_1)
        return self.path_1,

    def on_animate_2(self, idx):
        if self.i == 0:
            self.gradient = np.dot(self.a, np.array([self.x_2, self.y_2])) + self.b.T
            self.p = -self.gradient
            self.i += 1
        elif self.i == 1:
            gradient_old = self.gradient
            self.gradient = np.dot(self.a, np.array([self.x_2, self.y_2])) + self.b.T
            beta = np.dot(self.gradient.T, self.gradient) / np.dot(gradient_old.T, gradient_old)
            self.p = -self.gradient + np.dot(beta, self.p)
        hess = self.a
        lr = -np.dot(self.gradient, self.p.T) / np.dot(self.p.T, np.dot(hess, self.p))
        self.x_2 += lr * self.p[0]
        self.y_2 += lr * self.p[1]
        self.x_data_2.append(self.x_2)
        self.y_data_2.append(self.y_2)
        self.path_2.set_data(self.x_data_2, self.y_data_2)

        return self.path_2,

    def run_animation(self, xdata, ydata):
        if xdata != None and xdata != None:
            self.x_data_1, self.y_data_1 = [xdata], [ydata]
            self.x_data_2, self.y_data_2 = [xdata], [ydata]
            self.x_1, self.y_1 = xdata, ydata
            self.x_2, self.y_2 = xdata, ydata
            # self.ani_1 = FuncAnimation(self.figure, self.on_animate_1, init_func=self.animate_init_1, frames=5,
            #                            interval=self.animation_speed, repeat=False, blit=True)
            self.i = 0
            # self.ani_2 = FuncAnimation(self.figure2, self.on_animate_2, init_func=self.animate_init_2, frames=2,
            #                            interval=self.animation_speed, repeat=False, blit=True)


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')
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

    hide_pages(pages_created)

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.text('')
        st.subheader('Comparison of Methods')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/9.svg")), unsafe_allow_html=True)

        st.markdown(
            "Click in either graph to start a search point. Then watch the two algorithms attempt to find the minima. The two algorithms are:")
        st.markdown("- Steepest Descent using line search.")
        st.markdown("- Conjugate Gradient using line search.")
        xdata = st.slider("Set X", -2.0, 2.0, -1.0)
        ydata = st.slider("Set Y", -2.0, 2.0, -1.0)
        run = st.button("Train")
        st.subheader('*Chapter9*')
        st.markdown('---')

    app = ComparisonOfMethods(xdata, ydata)
    app.animate_init_1()
    app.animate_init_2()

    gap_col = st.columns([2, 8, 3])
    with gap_col[1]:
        plot_1 = st.pyplot(app.figure)
        plot_2 = st.pyplot(app.figure2)

    if run:
        for i in range(10):
            app.on_animate_1(i)
            app.on_animate_2(i)
            plot_1.pyplot(app.figure)
            plot_2.pyplot(app.figure2)
