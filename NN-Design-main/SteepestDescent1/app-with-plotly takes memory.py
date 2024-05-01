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


class SteepestDescentBackprop1():
    def __init__(self, pair_param, x , y):
        self.pair_of_params = pair_param
        self.w_ratio, self.h_ratio, self.dpi = 1, 1, 96
        self.P = np.arange(-2, 2.1, 0.1).reshape(1, -1)
        self.W1, self.b1 = np.array([[10], [10]]), np.array([[-5], [5]])
        self.W2, self.b2 = np.array([[1, 1]]), np.array([[-1]])
        A1 = self.logsigmoid(np.dot(self.W1, self.P) + self.b1)
        self.T = self.logsigmoid(np.dot(self.W2, A1) + self.b2)
        self.lr, self.epochs = None, None

        self.pair_of_params = pair_param
        self.pair_params = [["W1(1, 1)", "W2(1, 1)"], ["W1(1, 1)", "b1(1)"], ["b1(1)", "b1(2)"]]
        #  self.plot_data()

        figure1, figure2 = self.plot_data()
        self.figure_objs = figure1      # path, scatter, contour
        self.figure2_objs = figure2        # surface
        self.figure = go.Figure(data=figure1,
            layout=go.Layout(
                title="Start Title",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {
                                               "fromcurrent": True, "transition": {"duration": 1, "mode": "immediate"}
                                               }
                                  ])    
                            ]
                )],
            ),
            
            frames=[]
        )
        
        self.figure2 = go.Figure(data=figure2)

        # Update layout for the 3D subplot
        self.figure2.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        self.figure.update_xaxes(title_text=self.pair_params[self.pair_of_params - 1][0], title_font=dict(size=8))
        self.figure.update_yaxes(title_text=self.pair_params[self.pair_of_params - 1][1], title_font=dict(size=8))




        

        self.x, self.y = x, y

        self.animation_speed = 0


    # def change_pair_of_params(self, idx):
    #     if self.ani and self.ani.event_source:
    #         self.ani.event_source.stop()
    #     self.pair_of_params = idx + 1
    #     self.init_point_1.set_data([], [])
    #     self.end_point_1.set_data([], [])
    #     self.init_params()
    #     self.plot_data()

    def plot_data(self):
        self.x_data = []
        self.y_data = []
        plot_path = go.Scatter(x=self.x_data, y=self.y_data, mode='lines', line=dict(dash='dash'), name="Gradient Descent Path")
        # self.figure.data[1].x = []
        # self.figure.data[1].y = []
        # self.figure.data[2].x = []
        # self.figure.data[2].y = []

        f_data = loadmat(f"nndbp_new_{self.pair_of_params}.mat")
        x1, y1 = np.meshgrid(f_data["x1"], f_data["y1"])
        z = np.meshgrid(f_data["E1"])
        levels = f_data["levels"].reshape(-1)
        plot_contour = go.Contour(
            x=f_data["x1"].flatten(), y=f_data["y1"].flatten(), z=f_data["E1"],  # Adjust x, y, z as needed
            contours_coloring='lines',
        )
        plot2_surface = go.Surface(x=x1, y=y1, z=f_data["E1"], colorscale='Viridis', name="Sum Sq. Error")
        
        if self.pair_of_params == 1:
            plot_scatter = go.Scatter(x=[self.W1[0, 0]], y=[self.W2[0, 0], 0], mode='markers', marker=dict(symbol='star'), name="Initial Point")
        elif self.pair_of_params == 2:
            plot_scatter = go.Scatter(x=[self.W1[0, 0]], y=[self.b1[0, 0]], mode='markers', marker=dict(symbol='star'), name="Initial Point")   
        elif self.pair_of_params == 3:
            plot_scatter = go.Scatter(x=[self.b1[0, 0]], y=[self.b1[1, 0]], mode='markers', marker=dict(symbol='star'), name="Initial Point")   

        # Update axes 

        return [plot_path, plot_scatter, plot_contour], [plot2_surface]

    def animate_init(self):
        # self.end_point_1.set_data([], [])
        pass
        # self.path.set_data(self.x_data, self.y_data)
        # return self.path, self.end_point_1

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

        # if idx == self.epochs - 1:
            # self.end_point_1.set_data(self.x_data[-1], self.y_data[-1])

        self.x_data.append(self.x)
        self.y_data.append(self.y)
        plot_path = go.Scatter(x=self.x_data, y=self.y_data, mode='lines', line=dict(dash='dash'), name="Gradient Descent Path")
        plot_scatter = go.Scatter(x=[self.x_data[-1]], y=[self.y_data[-1]] , mode='markers', marker=dict(symbol='star'), name="Current Point")
        # self.path.set_data(self.x_data, self.y_data)
        return plot_path, plot_scatter

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
            self.lr, self.epochs = 3.5, 1000
        elif self.pair_of_params == 2:
            self.W1[0, 0], self.b1[0, 0] = self.x, self.y
            self.lr, self.epochs = 25, 300
        elif self.pair_of_params == 3:
            self.b1[0, 0], self.b1[1, 0] = self.x, self.y
            self.lr, self.epochs = 25, 60

    def animate(self):
        frames = []
        self.init_params()
        self.run_animation()
        for idx in range(1000):
            plot_path, plot_scatter = self.on_animate(idx)
            if idx%100==0:
                frames.append(
                    go.Frame(data=[plot_path,
                                    plot_scatter,
                                    self.figure_objs[2]
                                ]
                    )
                )
        print(frames[0])
        self.figure.frames = frames
        return self.figure

            
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

if __name__=="__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='wide', initial_sidebar_state='auto')
    
    st.title("Stepest Descent Backpropagation #1")
    cols1 = st.columns(3)
    with cols1[0]:
        pair_param = st.selectbox("Select pair of parameters", [1, 2, 3])
    with cols1[1]:
        x = st.number_input("Enter x", value=4.0)
    with cols1[2]:
        y = st.number_input("Enter y", value=4.0)

    app = SteepestDescentBackprop1(pair_param, x, y)

    cols = st.columns(2)
    with cols[0]:
        chart = st.plotly_chart(app.animate())
        # st.write(chart.event_data())
    with cols[1]:
        st.plotly_chart(app.figure2)
    
    # st.plotly_chart(app.figure)
