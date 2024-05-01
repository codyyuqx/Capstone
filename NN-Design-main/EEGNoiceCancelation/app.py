import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import time
import plotly.graph_objects as go   
PACKAGE_PATH = 'eegdata.mat'


class EEGNoiseCancellation():
    def __init__(self,lr, delays):
        self.w_ratio, self.h_ratio = 1, 1
        self.dpi = 100


        N, f, max_t = 3.33, 60, 0.5
        s = N * f
        self.ts = s * max_t + 1
        A1, A2, theta1, theta2, k = 1, 0.75, np.pi / 2, np.pi / 2.5, 0.00001
        self.signal_ = k * loadmat(PACKAGE_PATH)["eegdata"][:, :int(self.ts) + 1]
        i = np.arange(self.ts).reshape(1, -1)
        noise1, noise2 = 1.2 * np.sin(2 * np.pi * (i - 1) / N), 0.6 * np.sin(4 * np.pi * (i - 1) / N)
        noise = noise1 + noise2
        filtered_noise1 = A1 * 1.20 * np.sin(2 * np.pi * (i - 1) / N + theta1)
        filtered_noise2 = A2 * 0.6 * np.sin(4 * np.pi * (i - 1) / N + theta1)
        filtered_noise = filtered_noise1 + filtered_noise2
        noisy_signal = self.signal_ + filtered_noise

        self.time = np.arange(1, self.ts + 1) / self.ts * max_t

        self.P_ = np.zeros((21, 101))
        for i in range(21):
            self.P_[i, i + 1:] = noise[:, :101 - i - 1]
        self.P_ = np.array(self.P_)
        self.T = noisy_signal[:]

        self.x_data, self.y_data = [], []
        self.ani, self.x, self.y = None, None, None
        self.R, self.P = None, None
        self.a, self.e = None, None

        self.fixed_frame = go.Scatter(x=self.time, y=self.signal_[0], mode='lines', name='Original Signal', line=dict(color='blue', dash='10,2'))
        
        self.fig = go.Figure(
            data=[self.fixed_frame,
                go.Scatter(x=[0], y=[0], mode='lines', name='Approx Signal', line=dict(color='red'))],

            layout=go.Layout(
                xaxis=dict(range=[0, 0.5]),
                yaxis=dict(range=[-2, 2]),
                title='Original (blue) and Estimated (red) Signals',
                xaxis_title='Time',
                yaxis_title='Amplitude',
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": anim_delay, "redraw": False},
                                               "fromcurrent": True, "transition": {"duration": 900, "easing": "linear"}}])
                    ]
                )]
            ),
            frames=[]
        )



        self.lr = lr    # 0.02
        self.delays = delays    # 20
        self.do_slide = False

        self.animation_speed = 20
 

        self.w, self.e = None, None



    def animate_init(self):
        self.R = self.delays + 1
        self.P = self.P_[:self.R]
        self.w = np.zeros((1, self.R))
        self.a, self.e = np.zeros((1, 101)), np.zeros((1, 101))


    def on_animate(self, idx):
        p = self.P[:, idx]
        self.a[0, idx] = np.dot(self.w, p)
        self.e[0, idx] = self.T[0, idx] - self.a[0, idx]
        self.w += self.lr * self.e[0, idx] * p.T
   
        return [self.fixed_frame,
                go.Scatter(x=self.time[:idx + 1], y=self.e[0, :idx + 1], mode='lines', name='Approx Signal', line=dict(color='red')),
                ]

    def animate_init_diff(self):
        self.R = self.delays + 1
        self.P = self.P_[:self.R]
        self.w = np.zeros((1, self.R))
        self.a, self.e = np.zeros((1, 101)), np.zeros((1, 101))


    def on_animate_diff(self, idx):
        p = self.P[:, idx]
        self.a[0, idx] = np.dot(self.w, p)
        self.e[0, idx] = self.T[0, idx] - self.a[0, idx]
        self.w += self.lr * self.e[0, idx] * p.T

        return [self.fixed_frame,
                go.Scatter(x=self.time[:idx + 1], y=(self.signal_ - self.e)[0, :idx + 1], mode='lines', name='Signal Difference', line=dict(color='red')),
                ]

    def animate(self):
        frames = []
        for idx in range(1,101):
            frames.append(go.Frame(data=self.on_animate(idx)))
        self.fig.frames = frames
        return self.fig
    
    def animate_diff(self):
        frames = []
        for idx in range(1,101):
            frames.append(go.Frame(data=self.on_animate_diff(idx)))
        self.fig.frames = frames
        return self.fig


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered', initial_sidebar_state='auto')
    
    st.title("EEG Noice Cancelation")

    
    lr = st.slider("Lr:", 0.01, 0.2, 0.02)
    cols = st.columns(2)
    with cols[0]:
        delay = st.slider("Delays:", 1, 20, 10)
    with cols[1]:
        signal_type = st.selectbox('Select Type', ['Signals', 'Difference'], 0, )
    with st.sidebar:
        st.markdown("An EEG signal has been contaminated with noise. \n An adaptive linear network is used to remvoe the nosise. \n Use the slider to set the learning rate and the number of delays. \n You can choose to display the original and estimated signals or their difference. ")

        anim_delay = st.slider("Animation Delay:", 0, 50, 2, step=1) * 10  # Ensure multiples of 10

    app = EEGNoiseCancellation(lr, delay)


    if signal_type == 'Signals':
        app.animate_init()

        fig = app.animate()

    else:
        app.animate_init_diff()

        fig = app.animate_diff()

    

    fig.update_layout(
        legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'), # Adjust y to move legend inside subplot
        legend_orientation='h',
        legend_font_size=15,
        font=dict(family='Droid Sans', size=15, color='black'),
        xaxis_title="Time",
        xaxis_title_font_color='black',
        yaxis_title="Amplitude",
        yaxis_title_font_color='black',
    )
    
    st.plotly_chart(fig)

