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


class TaylorSeries2():
    def __init__(self, order, x, y):

        self.x_ = np.linspace(-2, 2, 1000)
        self.y_ = np.copy(self.x_)
        self.X, self.Y = np.meshgrid(self.x_, self.y_)
        self.F = (self.Y - self.X) ** 4 + 8 * self.X * self.Y - self.X + self.Y + 3
        self.F[self.F < 0] = 0
        self.F[self.F > 12] = 12

        xs = np.linspace(-2, 2, 100)
        ys = np.linspace(-2, 2, 100)
        self.XX, self.YY = np.meshgrid(xs, ys)
        FF = (self.YY - self.XX) ** 4 + 8 * self.XX * self.YY - self.XX + self.YY + 3
        FF[FF < 0] = 0
        FF[FF > 12] = 12

        self.x, self.y = x, y
        orders = ['Order 0', 'Order 1', 'Order 2']
        self.order = orders.index(order)
        # self.make_combobox(1, ["Order 0", "Order 1", "Order 2"], (self.x_chapter_usual, 300, self.w_chapter_slider, 100),
        #                    self.change_approx_order)

        # self.comboBox1.setCurrentIndex(self.order)

        # self.make_plot(1, (20, 135, 230, 230))
        # self.make_plot(2, (270, 135, 230, 230))
        # self.make_plot(3, (20, 400, 230, 230))
        # self.make_plot(4, (270, 400, 230, 230))

        # self.figure2 = go.Figure(
        #     layout=dict(height=250,
        #                 title="Function F",
        #                 margin=dict(l=0, r=0, b=0, t=0, pad=4, ),
        #                 )
        # )
        self.figure = plt.figure(figsize=(4, 4))
        self.figure2 = plt.figure(figsize=(4, 4))
        # self.figure3 = plt.figure(figsize=(4,4))
        # self.figure4 = plt.figure(figsize=(4,4))
        self.figure3 = go.Figure(
            layout=dict(
                title="Function",
                # margin=dict(l=0, r=0, b=0, t=0, pad=4, ),
            )
        )
        self.figure4 = go.Figure(
            layout=dict(
                title="Approximation",
                # margin=dict(l=0, r=0, b=0, t=0, pad=4, ),
            )
        )

        self.x_data, self.y_data = [], []

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.contour(self.X, self.Y, self.F, colors='blue')
        self.axes_1.set_title("Function", fontdict={'fontsize': 10})
        self.axes_1.set_xlim(-2, 2)
        self.axes_1.set_ylim(-2, 2)
        self.axes1_point, = self.axes_1.plot([], "o", fillstyle="none", color="k")
        self.axes1_point1, = self.axes_1.plot([], "o", fillstyle="none", markersize=11, color="k")
        # self.axes_1.text(-1.5, 1.65, "<CLICK ON ME>")
        # self.canvas.draw()
        # self.canvas.mpl_connect('button_press_event', self.on_mouseclick)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_title("Approximation", fontdict={'fontsize': 10})
        self.axes_2.set_xlim(-2, 2)
        self.axes_2.set_ylim(-2, 2)
        self.axes2_point, = self.axes_2.plot([], "o", fillstyle="none", color="k")
        self.axes2_point1, = self.axes_2.plot([], "o", fillstyle="none", markersize=11, color="k")
        # self.canvas2.draw()

        # self.axis1 = self.figure3.add_subplot(projection='3d')
        # self.axis1.set_title("Function", fontdict={'fontsize': 10}, pad=3)
        # self.axis1.plot_surface(self.XX, self.YY, FF)
        # self.axis1.view_init(30, -30)
        # self.axis1.autoscale()
        self.figure3.add_trace(go.Surface(x=self.XX, y=self.YY, z=FF, colorscale='tealgrn', showscale=False))

        # self.axis2 = self.figure4.add_subplot(projection='3d')
        # self.axis2.set_title("Approximation", fontdict={'fontsize': 10}, pad=3)
        # self.axis2.view_init(30, -30)

        self.on_mouseclick(x, y)
        self.draw_approx()

    def on_mouseclick(self, xdata, ydata):
        d_x, d_y = xdata - self.x_, ydata - self.y_
        x_event = self.x_[np.argmin(np.abs(d_x))]
        y_event = self.y_[np.argmin(np.abs(d_y))]
        if self.F[np.bitwise_and(self.X == x_event, self.Y == y_event)].item() == 12:
            return

        self.axes1_point.set_data([xdata], [ydata])
        self.axes1_point1.set_data([xdata], [ydata])
        self.axes2_point.set_data([xdata], [ydata])
        self.axes2_point1.set_data([xdata], [ydata])
        self.x, self.y = xdata, ydata
        self.draw_approx()
        # self.canvas.draw()

    def change_approx_order(self, idx):
        self.order = idx
        if self.x and self.y:
            self.draw_approx()

    def draw_approx(self):
        # Removes contours from second plot
        while self.axes_2.collections:
            for collection in self.axes_2.collections:
                collection.remove()
        # Draws new contour
        Fo = (self.y - self.x) ** 4 + 8 * self.x * self.y - self.x * self.y + 3
        gx = -4 * (self.y - self.x) ** 3 + 8 * self.y - 1
        gy = 4 * (self.y - self.x) ** 3 + 8 * self.x + 1
        gradient = np.array([[gx], [gy]])
        temp = 12 * (self.y - self.x) ** 2
        hess = np.array([[temp, 8 - temp], [8 - temp, temp]])
        dX, dY = self.X - self.x, self.Y - self.y
        if self.order == 0:
            Fa = np.zeros(self.X.shape) + Fo
        elif self.order == 1:
            Fa = gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
        elif self.order == 2:
            Fa = (hess[0, 0] * dX ** 2 + (hess[0, 1] + hess[1, 0]) * dX * dY + hess[1, 1] * dY ** 2) / 2
            Fa += gradient[0, 0] * dX + gradient[1, 0] * dY + Fo
        Fa[Fa < 0] = 0
        Fa[Fa > 12] = 12
        self.axes_2.contour(self.X, self.Y, Fa, colors="blue")
        # self.canvas2.draw()

        # Removes surface from fourth plot
        # while self.axis2.collections:
        #     for collection in self.axis2.collections:
        #         collection.remove()
        # Draws new surface
        dXX, dYY = self.XX - self.x, self.YY - self.y
        if self.order == 0:
            Fa = np.zeros(self.XX.shape) + Fo
        elif self.order == 1:
            Fa = gradient[0, 0] * dXX + gradient[1, 0] * dYY + Fo
        elif self.order == 2:
            Fa = (hess[0, 0] * dXX ** 2 + (hess[0, 1] + hess[1, 0]) * dXX * dYY + hess[1, 1] * dYY ** 2) / 2
            Fa += gradient[0, 0] * dXX + gradient[1, 0] * dYY + Fo
        Fa[Fa < 0] = 0
        Fa[Fa > 12] = 12
        # self.axis2.plot_surface(self.XX, self.YY, Fa, color='#1f77b4')
        self.figure4.add_trace(go.Surface(x=self.XX, y=self.YY, z=Fa, colorscale='tealgrn', showscale=False))
        # self.canvas4.draw()


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
        st.subheader('Taylor Series 2')
        # st.subheader('Regularization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/8.svg")), unsafe_allow_html=True)
        st.markdown("Select X and Y values for approximation.")
        st.markdown("You can rotate the 3D plots by clicking and dragging in the plot window.")
        st.markdown("Select the approximation order below.")

        order = st.selectbox("Approximation Order", ['Order 0', 'Order 1', 'Order 2'], index=1)
        x = st.slider("X", -2.0, 2.0, 0.0)
        y = st.slider("Y", -2.0, 2.0, 0.0)
        st.subheader('*Chapter8*')
        st.markdown('---')

    app = TaylorSeries2(order, x, y)
    app.figure3.update_layout(scene=dict(
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ),
        # add title
        title="Function",
    )
    app.figure4.update_layout(scene=dict(
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ),
        # add title
        title="Aproximation",
    )

    col_1, col_2 = st.columns([1, 1])
    with col_1:
        st.pyplot(app.figure)
        st.plotly_chart(app.figure3, use_container_width=True)
    with col_2:
        st.pyplot(app.figure2)
        st.plotly_chart(app.figure4, use_container_width=True)
