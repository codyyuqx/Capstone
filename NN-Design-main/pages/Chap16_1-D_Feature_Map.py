import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import time
from st_pages import hide_pages
from constants import pages_created
from scipy.signal import lfilter
import matplotlib
import os
import base64

font = {'size': 10}

matplotlib.rc('font', **font)


class OneDFeatureMap():
    def __init__(self, slider_lr, slider_nei, clear, random):
        if 'vars' not in st.session_state or st.session_state['vars'] is None:
            Sx, Sy = 1, 20
            S = Sx * Sy
            self.NDEC = 0.998

            self.W_ = np.zeros((S, 3))
            self.W_[:, -1] = 1
            Y, X = np.meshgrid(np.arange(1, Sy + 1), np.arange(1, Sx + 1))
            Ind2Pos = np.array([X.reshape(-1), Y.reshape(-1)]).T
            self.N = np.zeros((S, S))
            for i in range(S):
                for j in range(i):
                    self.N[i, j] = np.sqrt(np.sum((Ind2Pos[i, :] - Ind2Pos[j, :]) ** 2))

            self.Nfrom, self.Nto = list(range(2, 21)), list(range(1, 20))
            self.NN = len(self.Nfrom)

            self.N = self.N + self.N.T

            self.P = np.ones((3, 1000))
            self.P[:2, :] = np.random.random(
                (1000, 2)).T - 0.5  # The transpose is done so we get the same random numbers as in MATLAB
            self.P = np.divide(self.P, (np.ones((3, 1)) * np.sqrt(np.sum(self.P ** 2, axis=0))))

            up = np.arange(-0.5, 0.5, 0.1)
            down = -np.copy(up)
            flat = np.zeros((1, len(up))) + 0.5
            xx = np.array(list(up) + list(flat.reshape(-1)) + list(down) + list(-flat.reshape(-1)) + [up[0]])
            yy = np.array(list(-flat.reshape(-1)) + list(up) + list(flat.reshape(-1)) + list(down) + [-flat[0, 0]])
            zz = np.array([list(xx), list(yy)])
            zz = zz / (np.ones((2, 1)) * np.sqrt(np.sum(zz ** 2, axis=0) + 1))
            self.zz = zz

            self.lines = []
            self.lines_anim = []
            # self.canvas.draw()

            self.W = self.W_
            self.ani = None
            self.n_runs = 0

            self.lr = slider_lr
            self.nei = slider_nei

            self.do_slide = True

            # self.make_plot(1, (15, 100, 500, 500))

            st.session_state['vars'] = {
                'slider_lr': self.lr,
                'slider_nei': self.nei,
                'do_slide': self.do_slide,
                'n_runs': self.n_runs,
                'W': self.W,
                'W_': self.W_,
                'lines_anim': self.lines_anim,
                'N': self.N,
                'Nfrom': self.Nfrom,
                'Nto': self.Nto,
                'NN': self.NN,
                'P': self.P,
                'NDEC': self.NDEC,
                'lr': self.lr,
                'nei': self.nei,
                'zz': zz
            }
        else:
            self.lr = st.session_state['vars']['slider_lr']
            self.nei = st.session_state['vars']['slider_nei']
            self.do_slide = st.session_state['vars']['do_slide']
            self.n_runs = st.session_state['vars']['n_runs']
            self.W = st.session_state['vars']['W']
            self.W_ = st.session_state['vars']['W_']
            self.lines_anim = st.session_state['vars']['lines_anim']
            self.N = st.session_state['vars']['N']
            self.Nfrom = st.session_state['vars']['Nfrom']
            self.Nto = st.session_state['vars']['Nto']
            self.NN = st.session_state['vars']['NN']
            self.P = st.session_state['vars']['P']
            self.NDEC = st.session_state['vars']['NDEC']
            self.lr = st.session_state['vars']['lr']
            self.nei = st.session_state['vars']['nei']
            self.zz = st.session_state['vars']['zz']

        self.figure = plt.figure(figsize=(4, 4))
        self.axis1 = self.figure.add_subplot(1, 1, 1)
        self.axis1.set_xlim(-1, 1)
        self.axis1.set_ylim(-1, 1)
        self.axis1.plot(self.zz[0, :], self.zz[1, :])
        self.axis1.set_xticks([])
        self.axis1.set_yticks([])

    def on_reset(self):
        self.W = self.W_
        while self.lines_anim:
            self.lines_anim.pop().remove()
        self.canvas.draw()
        self.do_slide = False
        self.lr = 1
        self.nei = 21
        self.label_lr.setText("Learning rate: " + str(self.lr))
        self.label_nei.setText("Neighborhood: " + str(self.nei))
        self.slider_lr.setValue(self.lr * 100)
        self.slider_nei.setValue(self.nei * 10)
        self.do_slide = True
        self.n_runs = 0
        self.label_presentations.setText("Presentations: 0")

    def animate_init(self):
        while self.lines_anim:
            self.lines_anim.pop().remove()
        for _ in range(self.NN - 1):
            self.lines_anim.append(self.axis1.plot([], color="red")[0])
        st.session_state['vars']['lines_anim'] = self.lines_anim

    def on_animate(self, idx):

        s, r = self.W.shape
        Q = self.P.shape[1]

        for z in range(100):
            q = int(np.fix(np.random.random() * Q))
            p = self.P[:, q].reshape(-1, 1)

            a = self.compet_(np.dot(self.W, p))
            i = np.argmax(a)
            N_c = np.copy(self.N)[:, i]
            N_c[N_c <= self.nei] = 1
            N_c[N_c != 1] = 0
            a = 0.5 * (a + N_c.reshape(-1, 1))

            self.W = self.W + self.lr * np.dot(a, np.ones((1, r))) * (np.dot(np.ones((s, 1)), p.T) - self.W)
            self.lr = (self.lr - 0.01) * 0.998 + 0.01
            self.nei = (self.nei - 1) * self.NDEC + 1
            st.session_state['slider_lr'] = self.lr
            st.session_state['slider_nei'] = self.nei
            self.do_slide = False
            # self.slider_lr.setValue(round(self.lr * 100))
            # self.slider_nei.setValue(round(self.nei * 10))
            # self.label_lr.setText("Learning rate: " + str(round(self.lr, 2)))
            # self.label_nei.setText("Neighborhood: " + str(round(self.nei, 2)))
            self.do_slide = True
            # self.label_presentations.setText("Presentations: " + str((self.n_runs - 1) * 500 + idx * 100 + z + 1))

        for i in range(self.NN - 1):
            from_ = self.Nfrom[i] - 1
            to_ = self.Nto[i] - 1
            self.lines_anim[i].set_data([self.W[from_, 0], self.W[to_, 0]], [self.W[from_, 1], self.W[to_, 1]])

        st.session_state['vars']['W'] = self.W
        st.session_state['vars']['lr'] = self.lr
        st.session_state['vars']['nei'] = self.nei
        st.session_state['vars']['lines_anim'] = self.lines_anim
        st.session_state['vars']['n_runs'] = self.n_runs
        # self.canvas.draw()

    @staticmethod
    def compet_(n):
        max_idx = np.argmax(n)
        out = np.zeros(n.shape)
        out[max_idx] = 1
        return out


if __name__ == "__main__":
    st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')

    hide_pages(pages_created)
    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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


    image_path = 'media'

    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.text('')
        st.subheader('1 D Feature Map')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown("---")

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/16.svg")), unsafe_allow_html=True)
        st.markdown("Click [Train] to present 500\nvectors to the feature map.\n\n")
        st.markdown("Several clicks are required\nto obtain a stable network.\n\n")
        st.markdown("Click [Reset] to start over\nif the network develops\na twist.")

        if 'slider_lr' not in st.session_state:
            st.session_state['slider_lr'] = 1.0
        if 'slider_nei' not in st.session_state:
            st.session_state['slider_nei'] = 2.0
    with st.sidebar:
        reset = st.button("Reset")

    if reset:
        st.session_state['slider_lr'] = 1.0
        st.session_state['slider_nei'] = 21.0
        st.session_state['vars'] = None
        st.experimental_rerun()

    app = OneDFeatureMap(st.session_state['slider_lr'], st.session_state['slider_nei'], 0, 0)
    with st.sidebar:
        run = st.button("Train")
    col1 = st.columns([1, 3, 1])
    with col1[1]:
        the_plot = st.pyplot(plt)
    app.animate_init()

    if run:
        # app.on_run_2()
        app.n_runs += 500
        st.session_state['vars']['n_runs'] = app.n_runs
        for i in range(5):
            app.on_animate(i)
            the_plot.pyplot(plt)

            # time.sleep(0.5)
    with st.sidebar:
        if st.session_state['vars']:
            st.write("Iterations: ", str(st.session_state['vars']['n_runs']))
        else:
            st.write("Iterations: 0")
        slider_lr = st.slider('Learning Rate', 0.0, 1.0, app.lr, key='slider_lr')
        slider_nei = st.slider('Neighborhood', 0.0, 21.0, app.nei, key='slider_nei')