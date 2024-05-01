import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
from st_pages import hide_pages
from constants import pages_created
import os
import base64


class Regularization():
    def __init__(self, slider_nsd, slider_rer, anim_delay):
        self.max_epoch = 10
        self.T = 2
        pp0 = np.linspace(-1, 1, 201)

        self.pp = np.linspace(-0.95, 0.95, 20)
        self.P = np.linspace(-1, 1, 100)

        self.ani, self.tt, self.clicked = None, None, False
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.S1 = 20
        self.random_state = 42
        np.random.seed(self.random_state)

        self.nsd = slider_nsd

        self.regularization_ratio = slider_rer

        self.slider_rer = slider_rer

        self.train_plot = self.plot_train_test_data()
        self.scatter_plot = go.Scatter(x=pp0, y=np.sin(2 * np.pi * pp0 / self.T), mode='lines', name='sin(2*pi*x/T)')

        self.fig = go.Figure(
            data=[self.scatter_plot,
                  self.train_plot,
                  go.Scatter(x=[0], y=[0], mode='lines', name='NN Approximation', line=dict(dash='dash'))],
            layout=go.Layout(
                title="Function Approximation",
                xaxis=dict(range=[-1.0, 1.0], dtick=0.25),
                yaxis=dict(range=[-1.5, 1.5], tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0]),
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
            frames=[],
        )

        self.animation_speed = 0

        self.init_params()

    def plot_train_test_data(self):
        self.tt = np.sin(2 * np.pi * self.pp / self.T) + np.random.uniform(-2, 2, self.pp.shape) * 0.2 * self.nsd
        # self.train_points.set_data(self.pp, self.tt)
        return go.Scatter(x=self.pp, y=self.tt, mode='markers', name='Train',
                          marker=dict(symbol='star', color='green', size=10))

    def animate(self):
        frames = []
        for idx in range(self.max_epoch):
            frames.append(go.Frame(data=[
                self.scatter_plot,
                self.train_plot,
                self.on_animate_v2(idx)], ))
        self.fig.frames = frames
        return self.fig

    def on_animate(self, idx):
        alpha = 0.03
        nn_output = []
        for sample, target in zip(self.pp, self.tt):
            # Propagates the input forward
            # Reshapes input as 1x1
            a0 = sample.reshape(-1, 1)
            # Hidden Layer's Net Input
            n1 = np.dot(self.W1, a0) + self.b1
            #  Hidden Layer's Transformation
            a1 = self.logsigmoid(n1)
            # Output Layer's Net Input
            n2 = np.dot(self.W2, a1) + self.b2
            # Output Layer's Transformation
            a = self.purelin(n2)  # (a2 = a)
            nn_output.append(a)

            # Back-propagates the sensitivities
            # Compares our NN's output with the real value
            e = target - a
            # error = np.append(error, e)
            # Output Layer
            F2_der = np.diag(self.purelin_der(n2).reshape(-1))
            s = -2 * np.dot(F2_der, e)  # (s2 = s)
            # Hidden Layer
            F1_der = np.diag(self.logsigmoid_der(n1).reshape(-1))
            s1 = np.dot(F1_der, np.dot(self.W2.T, s))

            # Updates the weights and biases
            # Hidden Layer
            self.W1 += -alpha * np.dot(s1, a0.T)
            self.b1 += -alpha * s1
            # Output Layer
            self.W2 += -alpha * np.dot(s, a1.T)
            self.b2 += -alpha * s
        return go.Scatter(x=self.pp, y=nn_output, mode='lines', name='NN Approximation')

    def animate_init_v2(self):
        self.init_params()
        self.error_goal_reached = False
        self.a1 = self.tansig(np.dot(self.W1, self.pp.reshape(1, -1)) + self.b1)
        self.a2 = self.purelin(np.dot(self.W2, self.a1) + self.b2)
        self.e = self.tt.reshape(1, -1) - self.a2
        self.error_prev = np.dot(self.e, self.e.T).item()
        for param in [self.W1, self.b1, self.W2, self.b2]:
            self.error_prev += self.regularization_ratio * np.dot(param.reshape(1, -1), param.reshape(-1, 1)).item()
        self.mu = 0.01
        self.RS = self.S1 * 1
        self.RS1 = self.RS + 1
        self.RSS = self.RS + self.S1
        self.RSS1 = self.RSS + 1
        self.RSS2 = self.RSS + self.S1 * 1
        self.RSS3 = self.RSS2 + 1
        self.RSS4 = self.RSS2 + 1
        self.ii = np.eye(self.RSS4)

    def on_animate_v2(self, idx):
        """ Marqdt version """

        self.mu /= 10

        self.a1 = np.kron(self.a1, np.ones((1, 1)))
        d2 = self.lin_delta(self.a2)
        d1 = self.tan_delta(self.a1, d2, self.W2)
        jac1 = self.marq(np.kron(self.pp.reshape(1, -1), np.ones((1, 1))), d1)
        jac2 = self.marq(self.a1, d2)
        jac = np.hstack((jac1, d1.T))
        jac = np.hstack((jac, jac2))
        jac = np.hstack((jac, d2.T))
        je = np.dot(jac.T, self.e.T)

        grad = np.sqrt(np.dot(je.T, je)).item()
        if grad < 1e-7:
            print("Error goal reached! 1")
            return go.Scatter(x=self.P, y=self.forward(self.P.reshape(1, -1)).reshape(-1), mode='lines',
                              name='NN Approximation')

        jj = np.dot(jac.T, jac)
        # Can't get this operation to produce the exact same results as MATLAB...
        dw = -np.dot(np.linalg.inv(jj + self.mu * self.ii), je)
        dW1 = dw[:self.RS]
        db1 = dw[self.RS:self.RSS]
        dW2 = dw[self.RSS:self.RSS2].reshape(1, -1)
        db2 = dw[self.RSS2].reshape(1, 1)

        self.a1 = self.tansig(np.dot((self.W1 + dW1), self.pp.reshape(1, -1)) + self.b1 + db1)
        self.a2 = self.purelin(np.dot((self.W2 + dW2), self.a1) + self.b2 + db2)
        self.e = self.tt.reshape(1, -1) - self.a2
        error = np.dot(self.e, self.e.T).item()
        for param in [self.W1 + dW1, self.b1 + db1, self.W2 + dW2, self.b2 + db2]:
            error += self.regularization_ratio * np.dot(param.reshape(1, -1), param.reshape(-1, 1)).item()

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

                self.a1 = self.tansig(np.dot((self.W1 + dW1), self.pp.reshape(1, -1)) + self.b1 + db1)
                self.a2 = self.purelin(np.dot((self.W2 + dW2), self.a1) + self.b2 + db2)
                self.e = self.tt.reshape(1, -1) - self.a2
                error = np.dot(self.e, self.e.T).item()
                for param in [self.W1 + dW1, self.b1 + db1, self.W2 + dW2, self.b2 + db2]:
                    error += self.regularization_ratio * np.dot(param.reshape(1, -1), param.reshape(-1, 1)).item()

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

        if self.error_prev <= 0:
            if self.error_goal_reached:
                print("Error goal reached! 2")
                self.error_goal_reached = None
            return go.Scatter(x=self.P, y=self.forward(self.P.reshape(1, -1)).reshape(-1), mode='lines',
                              name='NN Approximation')
        return go.Scatter(x=self.P, y=self.forward(self.P.reshape(1, -1)).reshape(-1), mode='lines',
                          name='NN Approximation')

    def forward(self, p_in):
        a1 = self.tansig(np.dot(self.W1, p_in) + self.b1)
        return self.purelin(np.dot(self.W2, a1) + self.b2)

    def init_params(self):
        np.random.seed(self.random_state)
        self.W1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.b1 = np.random.uniform(-0.5, 0.5, (self.S1, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (1, self.S1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

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


    hide_pages(pages_created)
    image_path = 'media/Logo/book_logos'

    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('Regularization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    # slider_nsd = st.slider("Noise standard deviation:", 0.0, 3.0, 1.0)
    # slider_rer = st.slider("Regularization Ratio:", 0.0, 1.0, 0.20)

    with st.sidebar:
        st.markdown(load_svg(get_image_path("13.svg")), unsafe_allow_html=True)
        st.markdown("Click the [Train] button to train the logsig-linear network on the data points.")
        st.markdown("Use the slide bars to choose the number of neurons and the difficulty of the data points.")
        anim_delay = st.slider("Animation Delay:", 0, 50, 10, step=1) * 10
        slider_nsd = st.slider("Noise standard deviation:", 0.0, 3.0, 1.0)
        slider_rer = st.slider("Regularization Ratio:", 0.0, 1.0, 0.20)
        st.subheader('*Chapter13*')
        st.markdown('---')

    app = Regularization(slider_nsd, slider_rer, anim_delay)
    app.animate_init_v2()
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
    )

    st.plotly_chart(fig, use_container_width=True)