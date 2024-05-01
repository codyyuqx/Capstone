import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st
import time
import plotly.graph_objects as go

class FunctionApproximation():
    def __init__(self, diff, S1):
        # super(FunctionApproximation, self).__init__(w_ratio, h_ratio, dpi, main_menu=1)

        self.mu_initial = 0.01
        self.w_ratio, self.h_ratio = 1, 1

        self.mu_initial = 0.01
        self.mingrad = 0.0001

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
        self.error_goal_reached = False
        self.init_params()

        # self.make_plot(1, (20, 90, 480, 480))
        
        self.fixed_frame  = self.plot_f()
        self.fig = go.Figure(
            data = [self.fixed_frame,
                    go.Scatter(x=[0], y=[0], mode='lines', name='Network Approximation', line=dict(color='blue')),
                ],
                layout = go.Layout(
                    xaxis=dict(range=[-2, 2], autorange=False),
                    yaxis=dict(range=[-1, 3], autorange=False),
                    title='Function Approximation',
                    updatemenus=[dict(
                        type='buttons',
                        buttons=[dict(label="Train",
                                  method="animate",
                                  args=[None, {"frame": {"duration": anim_delay, "redraw": False},
                                               "fromcurrent": True, "transition": {"duration": 900, "easing": "linear"}}])
                        ],
                    )]
                ),
                frames=[]
        )

        self.anim_delay = 20
        self.pause = True

   
    def init_params(self):
        self.W1 = 2 * np.random.uniform(0, 1, (self.S1, 1)) - 0.5
        self.b1 = 2 * np.random.uniform(0, 1, (self.S1, 1)) - 0.5
        self.W2 = 2 * np.random.uniform(0, 1, (1, self.S1)) - 0.5
        self.b2 = 2 * np.random.uniform(0, 0, (1, 1)) - 0.5

    def plot_f(self):
        x, y = self.p, 1 + np.sin(np.pi * self.p * self.diff / 5)
        return go.Scatter(x=x, y=y, mode='lines', name='Function to Approximate', line=dict(color='orange'))
    

    def f_to_approx(self, p):
        return 1 + np.sin(np.pi * p * self.diff / 5)

    # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    def on_run(self):
        # self.ani_stop()
        n_epochs = 5000
        # self.ani = FuncAnimation(self.figure, self.on_animate_v2, init_func=self.animate_init_v2, frames=n_epochs,
                                #  interval=20, repeat=False, blit=True)

    def animate(self):
        frames = []
        for idx in range(1,20):
            frames.append(go.Frame(data=self.on_animate_v2(idx)))
        self.fig.frames = frames
        return self.fig
    
    def animate_init_v2(self):
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

        grad = np.sqrt(np.dot(je.T, je)).item()
        if grad < self.mingrad:
            self.net_approx.set_data(self.p.reshape(-1), self.a2.reshape(-1))
            # self.ani_stop()
            # return self.net_approx,

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

        if self.error_prev <= 0.005:
            if self.error_goal_reached:
                st.write("Error goal reached!")
                self.error_goal_reached = None
            x, y = (self.p.reshape(-1), self.a2.reshape(-1))
            return [self.fixed_frame, 
                    go.Scatter(x=x, y=y, mode='lines', name='Network Approximation', line=dict(color='blue')),
                    ]

        return [self.fixed_frame,
                go.Scatter(x=self.p.reshape(-1), y=self.a2.reshape(-1), mode='lines', name='Network Approximation', line=dict(color='blue')),
                ]

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

    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered', initial_sidebar_state='auto')

    st.title('Function Approximation')

    S1 = st.slider("Number of Hidden Neurons S1:", 1, 9, 4)
    diff = st.slider("Difficulty index:", 1, 9, 1)
    with st.sidebar:
        st.markdown("Click the [Train] button to train the logsig-linear network on the data points.")
        st.markdown("Use the slide bars to choose the number of neurons and the difficulty of the data points.")

        anim_delay =  st.slider("Animation Delay:", 0, 50, 2, step=1) * 10  # Ensure multiples of 10

    app = FunctionApproximation(diff, S1)

    app.animate_init_v2()
    fig = app.animate()
    fig.update_layout(
        legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'), # Adjust y to move legend inside subplot
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