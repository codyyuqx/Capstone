import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
from st_pages import hide_pages
import matplotlib
from scipy.signal import lfilter
from constants import pages_created
import base64
import os

font = {'size': 10}

matplotlib.rc('font', **font)


class TaylorSeries1():
    def __init__(self, function_cbx, order0_cbx, order1_cbx, order2_cbx, order3_cbx, order4_cbx, x_data):

        self.figure = plt.figure(figsize=(4, 4))
        self.figure2 = plt.figure(figsize=(4, 4))
        # self.make_plot(1, (115, 100, 290, 290))
        # self.make_plot(2, (115, 385, 290, 290))

        self.function_cbx = function_cbx
        self.order0_cbx = order0_cbx
        self.order1_cbx = order1_cbx
        self.order2_cbx = order2_cbx
        self.order3_cbx = order3_cbx
        self.order4_cbx = order4_cbx

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("cos(x)", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-6, 6)
        self.axes_1.set_ylim(-2, 2)
        self.axes_1.set_xticks([-6, -4, -2, 0, 2, 4])
        self.axes_1.set_yticks([-2, -1, 0, 1])
        self.axes_1.set_xlabel("$x$")
        self.axes_1.xaxis.set_label_coords(1, -0.025)
        self.axes_1.set_ylabel("$y$")
        self.axes_1.yaxis.set_label_coords(-0.025, 1)
        self.x_points = np.linspace(-6, 6)
        self.axes_1.plot(self.x_points, np.cos(self.x_points), "-")
        self.axes1_point_draw, = self.axes_1.plot([], 'mo')
        # self.axes_1.text(-3.5, 1.5, "<CLICK ON ME>")
        # self.canvas.draw()
        # self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation", fontdict={'fontsize': 10})
        self.f0, self.f1, self.f2, self.f3, self.f4 = None, None, None, None, None
        self.axes2_point_draw, = self.axes_2.plot([], 'mo')
        self.axes2_function, = self.axes_2.plot([], '-')
        self.axes2_approx_0, = self.axes_2.plot([], 'r-')
        self.axes2_approx_1, = self.axes_2.plot([], 'b-')
        self.axes2_approx_2, = self.axes_2.plot([], 'g-')
        self.axes2_approx_3, = self.axes_2.plot([], 'y-')
        self.axes2_approx_4, = self.axes_2.plot([], 'c-')
        self.axes_2.set_xlim(-6, 6)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_xlim(-6, 6)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_xticks([-6, -4, -2, 0, 2, 4])
        self.axes_2.set_yticks([-2, -1, 0, 1])
        self.axes_2.set_xlabel("$x$")
        self.axes_2.xaxis.set_label_coords(1, -0.025)
        self.axes_2.set_ylabel("$y$")
        self.axes_2.yaxis.set_label_coords(-0.025, 1)
        # self.canvas2.draw()
        self.on_mouseclick(x_data)

    def on_mouseclick(self, xdata):
        # if event.xdata != None and event.xdata != None:
        self.axes1_point_draw.set_data([xdata], [np.cos(xdata)])
        self.axes2_point_draw.set_data([xdata], [np.cos(xdata)])
        # self.canvas.draw()
        self.f0 = np.cos(xdata) + np.zeros(self.x_points.shape)
        self.f1 = self.f0 - np.sin(xdata) * (self.x_points - xdata)
        self.f2 = self.f1 - np.cos(xdata) * (self.x_points - xdata) ** 2 / 2
        self.f3 = self.f2 + np.sin(xdata) * (self.x_points - xdata) ** 3 / 6
        self.f4 = self.f3 + np.cos(xdata) * (self.x_points - xdata) ** 4 / 24
        self.draw_taylor()

    def draw_taylor(self):
        if self.function_cbx:
            self.axes2_function.set_data(self.x_points, np.cos(self.x_points))
        if self.order0_cbx:
            self.axes2_approx_0.set_data(self.x_points, self.f0)
        if self.order1_cbx:
            self.axes2_approx_1.set_data(self.x_points, self.f1)
        if self.order2_cbx:
            self.axes2_approx_2.set_data(self.x_points, self.f2)
        if self.order3_cbx:
            self.axes2_approx_3.set_data(self.x_points, self.f3)
        if self.order4_cbx:
            self.axes2_approx_4.set_data(self.x_points, self.f4)
        # self.canvas2.draw()

    # def function_checked(self, state):
    #     if state == QtCore.Qt.CheckState.Checked.value:
    #         self.axes2_function.set_data(self.x_points, np.cos(self.x_points))
    #     else:
    #         self.axes2_function.set_data([], [])
    #     self.canvas2.draw()

    # def order0_checked(self, state):
    #     if state == QtCore.Qt.CheckState.Checked.value:
    #         if self.f0 is not None:
    #             self.axes2_approx_0.set_data(self.x_points, self.f0)
    #     else:
    #         self.axes2_approx_0.set_data([], [])
    #     self.canvas2.draw()

    # def order1_checked(self, state):
    #     if state == QtCore.Qt.CheckState.Checked.value:
    #         if self.f1 is not None:
    #             self.axes2_approx_1.set_data(self.x_points, self.f1)
    #     else:
    #         self.axes2_approx_1.set_data([], [])
    #     self.canvas2.draw()

    # def order2_checked(self, state):
    #     if state == QtCore.Qt.CheckState.Checked.value:
    #         if self.f2 is not None:
    #             self.axes2_approx_2.set_data(self.x_points, self.f2)
    #     else:
    #         self.axes2_approx_2.set_data([], [])
    #     self.canvas2.draw()

    # def order3_checked(self, state):
    #     if state == QtCore.Qt.CheckState.Checked.value:
    #         if self.f3 is not None:
    #             self.axes2_approx_3.set_data(self.x_points, self.f3)
    #     else:
    #         self.axes2_approx_3.set_data([], [])
    #     self.canvas2.draw()

    # def order4_checked(self, state):
    #     if state == QtCore.Qt.CheckState.Checked.value:
    #         if self.f4 is not None:
    #             self.axes2_approx_4.set_data(self.x_points, self.f4)
    #     else:
    #         self.axes2_approx_4.set_data([], [])
    #     self.canvas2.draw()


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
        st.subheader('Taylor Series 1')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/8.svg")), unsafe_allow_html=True)
        st.markdown('Click on the checkbox to turn the function on and off.')
        st.markdown('set x value to change the point of approximation')

        function_cbx = st.checkbox('Function', value=True)
        order0_cbx = st.checkbox('Order 0', value=False)
        order1_cbx = st.checkbox('Order 1', value=True)
        order2_cbx = st.checkbox('Order 2', value=False)
        order3_cbx = st.checkbox('Order 3', value=False)
        order4_cbx = st.checkbox('Order 4', value=False)

        x_data = st.slider('Set value of X', -6, 6, 0)
        st.subheader('*Chapter8*')
        st.markdown('---')

    app = TaylorSeries1(function_cbx, order0_cbx, order1_cbx, order2_cbx, order3_cbx, order4_cbx, x_data)

    col1 = st.columns([1, 3, 1])
    with col1[1]:
        st.pyplot(app.figure)
    col2 = st.columns([1, 3, 1])
    with col2[1]:
        st.pyplot(app.figure2)
