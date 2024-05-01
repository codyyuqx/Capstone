import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_pages import hide_pages
from constants import pages_created
import matplotlib
import base64
import os


font = {'size': 18}

matplotlib.rc('font', **font)


class QuadraticFunction():
    def __init__(self, a11, a12, a21, a22, d1, d2, c):
        self.a11 = a11
        self.a12 = a12
        self.a21 = a21
        self.a22 = a22
        self.d1 = d1
        self.d2 = d2
        self.c = c

        self.x = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0,
                           0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
        self.y = np.copy(self.x)

        self.figure = plt.figure()
        self.figure2 = go.Figure(
            layout=dict(height=250,
                        title="Function F",
                        margin=dict(l=0, r=0, b=0, t=0, pad=4, ),
                        )
        )

        self.axes_1 = self.figure.add_subplot(1, 1, 1)
        self.axes_1.set_title("Function F", fontdict={'fontsize': 18})
        self.axes_1.set_xticks([-2, 0])

        self.on_run()

    def on_run(self):
        # if self.a_12.text() != self.a_21.text():
        #     self.a_21.setText(self.a_12.text())
        A = np.array([[self.a11, self.a12], [self.a21, self.a22]])
        d = np.array([[self.d1], [self.d2]])
        c = self.c
        self.update(A, d, c)

    def update(self, A, d, c):
        minima = -np.dot(np.linalg.pinv(A), d)
        x0, y0 = minima[0, 0], minima[1, 0]
        xx = self.x + x0
        yy = self.y + y0
        XX, YY = np.meshgrid(xx, yy)
        F = (A[0, 0] * XX ** 2 + (A[0, 1] + A[1, 0]) * XX * YY + A[1, 1] * YY ** 2) / 2 + d[0, 0] * XX + d[
            1, 0] * YY + c
        e, v = np.linalg.eig(A)


        # Draws new stuff
        self.axes_1.set_xlim(np.min(xx), np.max(xx))
        self.axes_1.set_ylim(np.min(yy), np.max(yy))
        self.axes_1.plot([x0] * 20, np.linspace(np.min(yy), np.max(yy), 20), linestyle="dashed", linewidth=0.6,
                         color="gray", )
        self.axes_1.plot(np.linspace(np.min(xx), np.max(xx), 20), [y0] * 20, linestyle="dashed", linewidth=0.5,
                         color="gray")
        self.axes_1.contour(XX, YY, F)
        self.axes_1.quiver([x0], [y0], [-v[0, 0]], [-v[1, 0]], units="xy", scale=1, label="Eigenvector 1")
        self.axes_1.quiver([x0], [y0], [-v[0, 1]], [-v[1, 1]], units="xy", scale=1, label="Eigenvector 2")
        self.figure2.add_trace(go.Surface(z=F, x=xx, y=yy, colorscale='tealgrn', showscale=False))
        # self.canvas.draw()
        # self.canvas2.draw()


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
        st.subheader('Quadratic Function')
        # st.subheader('Regularization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/8.svg")), unsafe_allow_html=True)
        st.markdown(
            "Change the values of the Hessian matrix A, the vector d, and the constant c.\n\nThen click [Update] to see the new function.\n\nNote that the Hessian matrix will always be symmetric.\n\nYou can rotate the 3D plots by clicking and dragging in the plot window.")
        col_a = st.columns(2)
        with col_a[0]:
            a11 = st.number_input('A11', value=1.5)
            a12 = st.number_input('A12', value=-0.7, )
        with col_a[1]:
            a21 = st.number_input('A21', value=-0.7)
            a22 = st.number_input('A22', value=1.0)
        d1 = st.number_input('d1', value=0.35)
        d2 = st.number_input('d2', value=0.25)
        c = st.number_input('c', value=1.0)
        st.subheader('*Chapter8*')
        st.markdown('---')
    # input_cols = st.columns([4, 2, 1])
    # with input_cols[0]:
    #     col_a = st.columns(2)
    #     with col_a[0]:
    #         a11 = st.number_input('A11', value=1.5)
    #         a12 = st.number_input('A12', value=-0.7, )
    #     with col_a[1]:
    #         a21 = st.number_input('A21', value=-0.7)
    #         a22 = st.number_input('A22', value=1.0)
    #
    # with input_cols[1]:
    #     d1 = st.number_input('d1', value=0.35)
    #     d2 = st.number_input('d2', value=0.25)
    #
    # with input_cols[2]:
    #     c = st.number_input('c', value=1.0)

    app = QuadraticFunction(a11, a12, a21, a22, d1, d2, c)


    # with cols[0]:
    col1 = st.columns(2)
    with col1[0]:
        st.pyplot(app.figure)
    with col1[1]:
        app.figure2.update_layout(scene=dict(
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
            # add title
            title="Function F",
        )
        st.plotly_chart(app.figure2, use_container_width=True)

    image_cols = st.columns([1, 5, 1])
    with image_cols[1]:
        st.markdown(
            load_svg(get_image_path("Figures/equation1.svg")), unsafe_allow_html=True
        )
