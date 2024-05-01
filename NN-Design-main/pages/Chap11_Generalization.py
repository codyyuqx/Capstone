import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import os
from st_pages import Page, show_pages, add_page_title, hide_pages
from constants import pages_created

# from matplotlib.figure import Figure

import math
import time


class Generalization:

    def __init__(self, diff, S1):
        self.mu_initial = 0.01
        self.w_ratio, self.h_ratio = 1, 1
        # self.mingrad = 0.001
        self.S1 = S1
        self.diff = diff
        self.p = np.linspace(-2, 2, 100)
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.mu = None
        self.ani = None
        self.random_state = 0
        self.init_params()
        self.error_prev, self.ii = 1000, None
        self.RS, self.RS1, self.RSS, self.RSS1 = None, None, None, None
        self.RSS2, self.RSS3, self.RSS4 = None, None, None
        self.init_params()

        self.p = None
        self.fixed_frame = self.plot_f()
        self.fig = go.Figure(
            data=[go.Scatter(x=[0], y=[0], mode='lines', name='Network Approximation', line=dict(width=3)),
                  self.fixed_frame
                  ],
            layout=go.Layout(
                xaxis=dict(range=[-2, 2], autorange=False),
                yaxis=dict(range=[0, 2], autorange=False),
                title="Function Approximation",
                title_x=0.4,
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Train",
                                  method="animate",
                                  args=[None, {"frame": {"duration": anim_delay, "redraw": False},
                                               "fromcurrent": True,
                                               "transition": {"duration": 900, "easing": "linear"}}])
                             ]
                )]
            ),
            frames=[]
        )

        self.anim_delay = 20

        self.pause = True

    def init_params(self):
        # np.random.seed(self.random_state)
        p_min, p_max = np.min(self.p), np.max(self.p)
        magw = 2.8 * self.S1 ** (1 / 1)
        self.W1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.W1[self.W1 < 0] = -1
        self.W1[self.W1 >= 0] = 1
        self.W1 *= magw
        self.b1 = magw * np.random.uniform(-0.5, 0.5, (self.S1, 1))
        rng = p_min - p_max
        mid = 0.5 * (p_min + p_max)
        self.W1 = 2 * self.W1 / rng
        self.b1 = self.b1 - self.W1 * mid
        self.W2 = np.random.uniform(-0.5, 0.5, (1, self.S1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def plot_f(self):
        self.p = np.linspace(-2, 2, 11 * self.diff)
        # self.p = self.p.reshape(-1, 1)
        x = self.p
        y = 1 + np.sin(np.pi * self.p * self.diff / 5)
        # self.fig.data = [[x, y], []]

        return go.Scatter(x=x, y=y, mode='markers', name='Function to Approximate',
                          # plus red marker
                          marker=dict(color='red', size=8, symbol='cross'), )

    def f_to_approx(self, p):
        return 1 + np.sin(np.pi * p * self.diff / 5)

    def animate(self):
        frames = []
        for idx in range(1, 50):
            frames.append(go.Frame(data=self.on_animate_v2(idx)))
        self.fig.frames = frames
        return self.fig

    def animate_init_v2(self):
        np.random.seed(5)
        self.init_params()
        self.error_goal_reached = False
        self.p = self.p.reshape(1, -1)
        self.a1 = self.logsigmoid_stable(np.dot(self.W1, self.p) + self.b1)
        self.a2 = self.purelin(np.dot(self.W2, self.a1) + self.b2)
        self.e = self.f_to_approx(self.p) - self.a2
        self.error_prev = np.dot(self.e, self.e.T).item()
        self.mu = self.mu_initial
        self.RS = self.S1 * 1
        self.RS1 = self.RS + 1
        self.RSS = self.RS + self.S1
        self.RSS1 = self.RSS + 1
        self.RSS2 = self.RSS + self.S1 * 1
        self.RSS3 = self.RSS2 + 1
        self.RSS4 = self.RSS2 + 1
        self.ii = np.eye(self.RSS4)
        # self.net_approx.set_data([], [])
        # return self.net_approx,

    def on_animate_v2(self, idx):
        """ Marqdt version """

        self.mu /= 10

        self.a1 = np.kron(self.a1, np.ones((1, 1)))
        d2 = self.lin_delta(self.a2)
        d1 = self.log_delta(self.a1, d2, self.W2)
        jac1 = self.marq(np.kron(self.p, np.ones((1, 1))), d1)
        jac2 = self.marq(self.a1, d2)
        jac = np.hstack((jac1, d1.T))
        jac = np.hstack((jac, jac2))
        jac = np.hstack((jac, d2.T))
        je = np.dot(jac.T, self.e.T)

        # grad = np.sqrt(np.dot(je.T, je)).item()
        # if grad < self.mingrad:
        #     self.net_approx.set_data(self.p.reshape(-1), self.a2.reshape(-1))
        #     return self.net_approx,

        jj = np.dot(jac.T, jac)
        # Can't get this operation to produce the exact same results as MATLAB...
        dw = -np.dot(np.linalg.inv(jj + self.mu * self.ii), je)
        dW1 = dw[:self.RS]
        db1 = dw[self.RS:self.RSS]
        dW2 = dw[self.RSS:self.RSS2].reshape(1, -1)
        db2 = dw[self.RSS2].reshape(1, 1)

        self.a1 = self.logsigmoid_stable(np.dot((self.W1 + dW1), self.p) + self.b1 + db1)
        self.a2 = self.purelin(np.dot((self.W2 + dW2), self.a1) + self.b2 + db2)
        self.e = self.f_to_approx(self.p) - self.a2
        error = np.dot(self.e, self.e.T).item()

        while error >= self.error_prev:

            try:

                self.mu *= 10
                if self.mu > 1e10:
                    break

                dw = -np.dot(np.linalg.inv(jj + self.mu * self.ii), je)
                dW1 = dw[:self.RS]
                db1 = dw[self.RS:self.RSS]
                dW2 = dw[self.RSS:self.RSS2].reshape(1, -1)
                db2 = dw[self.RSS2].reshape(1, 1)

                self.a1 = self.logsigmoid_stable(np.dot((self.W1 + dW1), self.p) + self.b1 + db1)
                self.a2 = self.purelin(np.dot((self.W2 + dW2), self.a1) + self.b2 + db2)
                self.e = self.f_to_approx(self.p) - self.a2
                error = np.dot(self.e, self.e.T).item()

            except Exception as e:
                if str(e) == "Singular matrix":
                    print("The matrix was singular... Increasing mu 10-fold")
                    self.mu *= 10
                else:
                    raise e

        if error < self.error_prev:
            self.W1 += dW1
            self.b1 += db1
            self.W2 += dW2
            self.b2 += db2
            self.error_prev = error

        p = np.linspace(-2, 2, 100).reshape(1, -1)
        a1 = self.logsigmoid_stable(np.dot(self.W1, p) + self.b1)
        a2 = self.purelin(np.dot(self.W2, a1) + self.b2)

        return [go.Scatter(x=p.reshape(-1), y=a2.reshape(-1), mode='lines', name='Function to Approximate'),
                self.fixed_frame]

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
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered', initial_sidebar_state='auto')

    hide_pages(pages_created)

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        #st.subheader('*Chapter11*')
        st.subheader('Generalization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')
    st.markdown('---')

    

    with st.sidebar:
        st.markdown(load_svg(get_image_path("11.svg")), unsafe_allow_html=True)
        st.markdown("Click the [Train] button to train the logsig-linear network on the data points.")
        st.markdown("Use the slide bars to choose the number of neurons and the difficulty of the data points.")

        anim_delay = st.slider("Animation Delay:", 0, 50, 2, step=1) *10  # Ensure multiples of 10
        S1 = st.slider("Number of Hidden Neurons S1:", 1, 9, 4)

        diff = st.slider("Difficulty index:", 1, 9, 1)
        st.subheader('*Chapter11*')
        st.markdown('---')

    app = Generalization(diff, S1)

    app.animate_init_v2()
    # st.write(diff)
    fig = app.animate()
    fig.update_layout(
        legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'),  # Adjust y to move legend inside subplot
        legend_orientation='h',
        legend_font_size=15,
        # font_family='Droid Sans',
        font=dict(family='Droid Sans', size=15, color='black'),
        xaxis_title="Input",
        xaxis_title_font_color='black',
        yaxis_title="Target",
        yaxis_title_font_color='black',
        width=600,
        height=600,
    )

    col1 = st.columns([1, 9, 1])
    with col1[1]:
        st.plotly_chart(fig, use_container_width=True)

    



