import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_pages import hide_pages
import matplotlib
from constants import pages_created
import plotly.express as px
import plotly.graph_objects as go
import base64
import os

font = {'size': 12}

matplotlib.rc('font', **font)


class NewtonsMethod():
    def __init__(self, xdata, ydata):

        x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0,
                      0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
        y = np.copy(x)
        self.X, self.Y = np.meshgrid(x, y)
        self.max_epoch = 50
        self.F = (self.Y - self.X) ** 4 + 8 * self.X * self.Y - self.X + self.Y + 3
        self.F[self.F < 0] = 0
        self.F[self.F > 12] = 12

        # self.make_plot(1, (115, 100, 290, 290))
        # self.make_plot(2, (115, 385, 290, 290))
        self.figure = plt.figure(figsize=(4, 4))
        self.figure2 = plt.figure(figsize=(4, 4))

        self.x_data, self.y_data = [], []
        self.ani_1, self.ani_2, self.event, self.x, self.y = None, None, None, None, None

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        # self.axes_1.text(-0.9, 1.7, ">Click me<")
        self.axes_1.contour(self.X, self.Y, self.F)
        self.axes_1.set_title("Function F", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.path_1, = self.axes_1.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Newton's Method Path")
        self.init_point_1, = self.axes_1.plot([], "o", fillstyle="none", markersize=11, color="k")
        self.axes_1.set_yticks([-2, -1, 0, 1, 2])
        # self.canvas.draw()
        # self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation Fa", fontdict={'fontsize': 10})
        self.path_2, = self.axes_2.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Newton's Method Path")
        self.init_point_2, = self.axes_2.plot([], "o", fillstyle="none", markersize=11, color="k")
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.axes_2.set_yticks([-2, -1, 0, 1, 2])
        # self.canvas2.draw()

        self.animation_speed = 100

        self.on_mouseclick(xdata, ydata)
        # self.make_slider("slider_anim_speed", QtCore.Qt.Orientation.Horizontal, (0, 6), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 2,
        #                  (self.x_chapter_usual, 380, self.w_chapter_slider, 100), self.slide, "label_anim_speed", "Animation Delay: 200 ms")

    # def slide(self):
    #     self.animation_speed = int(self.slider_anim_speed.value()) * 100
    #     self.label_anim_speed.setText("Animation Delay: " + str(self.animation_speed) + " ms")
    #     if self.x_data:
    #         if self.ani_1:
    #             self.ani_1.event_source.stop()
    #             self.ani_2.event_source.stop()
    #         self.x_data, self.y_data = [self.x_data[0]], [self.y_data[0]]
    #         self.x, self.y = self.x_data[0], self.y_data[0]
    #         self.path_1, = self.axes_1.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
    #         self.path_2, = self.axes_2.plot([], linestyle='--', marker='*', label="Gradient Descent Path")
    #         self.canvas.draw()
    #         self.canvas2.draw()
    #         self.run_animation(self.event)

    def animate_init_1(self):
        self.path_1, = self.axes_1.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Newton's Method Path")
        return self.path_1,

    def animate_init_2(self):
        self.path_2, = self.axes_2.plot([], linestyle='--', marker="o", fillstyle="none", color="k",
                                        label="Newton's Method Path")
        return self.path_2,

    def on_animate_1(self, idx):
        gx = -4 * (self.y - self.x) ** 3 + 8 * self.y - 1
        gy = 4 * (self.y - self.x) ** 3 + 8 * self.x + 1
        gradient = np.array([[gx], [gy]])
        temp = 12 * (self.y - self.x) ** 2
        hess = np.array([[temp, 8 - temp], [8 - temp, temp]])
        dxy = -np.dot(np.linalg.inv(hess), gradient)
        self.x += dxy[0, 0]
        self.y += dxy[1, 0]
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.path_1.set_data(self.x_data, self.y_data)
        return self.path_1,

    def on_animate_2(self, idx):
        self.path_2.set_data(self.x_data[:2], self.y_data[:2])
        return self.path_2,

    def on_mouseclick(self, xdata, ydata):

        # Checks whether the clicked point is in the contour
        x_event, y_event = xdata, ydata
        if round(x_event, 1) * 10 % 2 == 1:
            x_event += 0.06
            if round(x_event, 1) * 10 % 2 == 1:
                x_event = xdata - 0.06
        if round(y_event, 1) * 10 % 2 == 1:
            y_event += 0.06
            if round(y_event, 1) * 10 % 2 == 1:
                y_event = ydata - 0.06
        x_event = round(x_event, 1)
        y_event = round(y_event, 1)
        if self.F[np.bitwise_and(self.X == x_event, self.Y == y_event)].item() == 12:
            return

        self.x, self.y = xdata, ydata
        # Removes contours from second plot
        # while self.axes_2.collections:
        #     for collection in self.axes_2.collections:
        #         collection.remove()
        # Draws new contour
        Fo = (self.y - self.x) ** 4 + 8 * self.x * self.y - self.x * self.y + 3
        gx = -4 * (self.y - self.x) ** 3 + 8 * self.y - 1
        gy = 4 * (self.y - self.x) ** 3 + 8 * self.x + 1
        gradient = np.array([[gx], [gy]])
        temp = 12 * (self.y - self.x) ** 2
        hess = np.array([[temp, 8 - temp], [8 - temp, temp]])
        dX, dY = self.X - self.x, self.Y - self.y
        Fa = (hess[0, 0] * dX ** 2 + (hess[0, 1] + hess[1, 0]) * dX * dY + hess[1, 1] * dY ** 2) / 2
        Fa += gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
        self.axes_2.contour(self.X, self.Y, Fa)

        # self.event = event
        # if self.ani_1 and self.ani_1.event_source:
        #     self.ani_1.event_source.stop()
        # if self.ani_2 and self.ani_2.event_source:
        #     self.ani_2.event_source.stop()
        self.path_1.set_data([], [])
        self.path_2.set_data([], [])
        self.x_data, self.y_data = [], []
        self.init_point_1.set_data([xdata], [ydata])
        self.init_point_2.set_data([xdata], [ydata])
        # self.canvas.draw()
        # self.canvas2.draw()
        self.run_animation(xdata, ydata)

    def run_animation(self, xdata, ydata):
        self.x_data, self.y_data = [self.x], [self.y]
        # self.ani_1 = FuncAnimation(self.figure, self.on_animate_1, init_func=self.animate_init_1, frames=self.max_epoch,
        #                            interval=self.animation_speed, repeat=False, blit=True)
        # self.ani_2 = FuncAnimation(self.figure, self.on_animate_2, init_func=self.animate_init_2, frames=self.max_epoch,
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
        st.subheader('Newton\'s Method')
        # st.subheader('Regularization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/8.svg")), unsafe_allow_html=True)
        st.markdown(
            'Click anywhere on the graph to start an initial guess. Then the steepest descent trajectory will be shown.')
        st.markdown("The bottom graph shows the approximation of function F at the initial point.")
        xdata = st.slider("Set X", -2.0, 2.0, -1.0)
        ydata = st.slider("Set Y", -2.0, 2.0, -1.0)
        run = st.button("Train")
        st.subheader('*Chapter9*')
        st.markdown('---')

    app = NewtonsMethod(xdata, ydata)

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
