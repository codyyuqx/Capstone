import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_pages import hide_pages
from constants import pages_created
import matplotlib
import os
import base64

font = {'size': 18}

matplotlib.rc('font', **font)


# saperate function of create_variable is created to make the lag as minimum as possible...
# as streamlit cant do the realtime dynamic change directly

@st.cache
def create_variables():
    p1, p2 = np.array([[1], [1]]), np.array([[-1], [1]])
    t1, t2 = 1, -1
    prob1, prob2 = 0.75, 0.25
    R = prob1 * p1.dot(p1.T) + prob2 * p2.dot(p2.T)
    h = prob1 * t1 * p1 + prob2 * t2 * p2
    c = prob1 * t1 ** 2 + prob2 * t2 ** 2
    a, b = 2 * R, -2 * h
    a1, b1, c1 = np.array([[2, 0], [0, 2]]), np.zeros((2, 1)), 0

    x1, y1 = np.linspace(-0.5, 1.5, 50), np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x1, y1)

    max_epoch = 100

    F = (a[0, 0] * X ** 2 + (a[0, 1] + a[1, 0]) * X * Y + a[1, 1] * Y ** 2) / 2 + b[0] * X + b[1] * Y + c
    F = np.clip(F, np.min(F), 3)
    sol = -np.linalg.pinv(a).dot(b)

    F1 = (a1[0, 0] * X ** 2 + (a1[0, 1] + a1[1, 0]) * X * Y + a1[1, 1] * Y ** 2) / 2 + b1[0] * X + b1[1] * Y + c1
    sol1 = -np.linalg.pinv(a1).dot(b1)
    F1 = np.clip(F1, np.min(F1), 3)

    x1, x2 = 0, 0
    lr_path_x, lr_path_y = [x1], [x2]
    lr = 0.05
    for epoch_ in range(max_epoch):
        grad = a.dot(np.array([[x1], [x2]])) + b
        x1 -= lr * grad[0, 0]
        x2 -= lr * grad[1, 0]
        lr_path_x.append(x1)
        lr_path_y.append(x2)

    return F, F1, sol, sol1, lr_path_x, lr_path_y, X, Y, a, b, a1, b1, c, c1


@st.cache
def create_variables_2(sol, a, b, a1, b1):
    x1, x2 = sol[0], sol[1]
    ro_path_x, ro_path_y = [x1], [x2]
    ro_list = []
    for alpha in np.linspace(0, 1, 101):
        beta = 1 - alpha
        x = -np.linalg.inv(beta * a + alpha * a1).dot((beta * b))
        if beta == 0:
            ro_list.append("inf")
        else:
            ro_list.append(alpha / beta)
        x1 = x[0, 0]
        x2 = x[1, 0]
        ro_path_x.append(x1)
        ro_path_y.append(x2)
    return ro_path_x, ro_path_y, ro_list


class EarlyStoppingRegularization():
    def __init__(self, epoch, ro_epoch):
        # w_ratio, h_ratio = 1, 1
        self.w_ratio, self.h_ratio = 1, 1

        # self.make_plot(1, (100, 90, 300, 300))
        self.figure1 = plt.figure()
        self.axes_1 = self.figure1.add_subplot(1, 1, 1)

        # self.figure.subplots_adjust(left=0.175, bottom=0.175, right=0.95)
        # self.make_plot(2, (100, 380, 300, 300))
        # self.figure2.subplots_adjust(left=0.175, bottom=0.175, right=0.95)

        F, F1, sol, sol1, lr_path_x, lr_path_y, X, Y, a, b, a1, b1, c, c1 = create_variables()

        self.figure2 = plt.figure()
        self.axes_2 = self.figure2.add_subplot(1, 1, 1)

        self.axes_1 = self.figure1.add_subplot(1, 1, 1)
        self.axes_1.set_xticks([0])
        self.axes_1.set_yticks([0])
        self.axes_1.set_title("Early Stopping")
        self.axes_1.set_xlabel("$x(1)$")
        self.axes_1.set_ylabel("$x(2)$")
        self.axes_1.contour(X, Y, F, levels=[0.02, 0.07, 0.15], colors="red")
        self.axes_1.contour(X, Y, F1, levels=[0.025, 0.08, 0.15], colors="red")

        # st.write(self.lr_path_x[epoch], self.lr_path_y[epoch])
        # self.axes_1_lr_pos.set_data([sol1[0]], [sol1[1]])
        self.axes_1.plot(sol[0], sol[1], ".", color="blue", markersize=15)
        self.axes_1.plot(sol1[0], sol1[1], ".", color="blue", markersize=15)
        self.axes_1.plot(lr_path_x, lr_path_y, color="blue", linewidth=4)

        self.axes_1.plot(lr_path_x[epoch], lr_path_y[epoch], "o", fillstyle="none",
                         markersize=int(6 * (self.w_ratio + self.h_ratio) / 2), color="k", markeredgewidth=4, )
        # self.canvas.draw()
        # self.axes_1_lr_pos.set_data([])

        ro_path_x, ro_path_y, ro_list = create_variables_2(sol, a, b, a1, b1)

        self.axes_2 = self.figure2.add_subplot(1, 1, 1)
        self.axes_2.set_xticks([0])
        self.axes_2.set_yticks([0])
        self.axes_2.set_title("Regularization")
        self.axes_2.set_xlabel("$x(1)$")
        self.axes_2.set_ylabel("$x(2)$")
        self.axes_2.contour(X, Y, F, levels=[0.02, 0.07, 0.15], colors="red")
        self.axes_2.contour(X, Y, F1, levels=[0.025, 0.08, 0.15], colors="red")
        self.axes_2.plot(sol[0], sol[1], ".", color="blue", markersize=15)
        self.axes_2.plot(sol1[0], sol1[1], ".", color="blue", markersize=15)

        ro_path_x = [(x[0] if isinstance(x, np.ndarray) else x) for x in ro_path_x]
        ro_path_y = [(y[0] if isinstance(y, np.ndarray) else y) for y in ro_path_y]

        self.axes_2.plot(ro_path_x, ro_path_y, color="blue", linewidth=4)
        self.axes_2_ro_pos, = self.axes_2.plot([], [], "o", fillstyle="none",
                                               markersize=int(6 * (self.w_ratio + self.h_ratio) / 2), color="k",
                                               markeredgewidth=4, )
        self.axes_2_ro_pos.set_data([sol[0]], [sol[1]])
        self.axes_2_ro_pos.set_data([ro_path_x[ro_epoch]], [ro_path_y[ro_epoch]])


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


    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)


    image_path = 'media/Logo/book_logos'
    hide_pages(pages_created)

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        # st.text('')
        st.subheader('Early Stopping')
        st.subheader('Regularization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("13.svg")), unsafe_allow_html=True)
        st.markdown(
            "Click on the epoch slider to see the steepest descent trajectory.\n\nClick on the ro slider to see the minimum of the regularized")
        epoch = st.slider("Epoch", 0, 100, 0)
        ro_epoch = st.slider("Ro Epoch", 0, 100, 1)
        st.subheader('*Chapter13*')
        st.markdown('---')

    app = EarlyStoppingRegularization(epoch, ro_epoch)

    cols = st.columns(2)
    with cols[0]:
        st.pyplot(app.figure1)
    with cols[1]:
        st.pyplot(app.figure2)
