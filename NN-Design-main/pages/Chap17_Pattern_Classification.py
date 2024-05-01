import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_pages import hide_pages
import matplotlib
from constants import pages_created
import base64
import os

font = {'size': 18}

matplotlib.rc('font', **font)


class PatternClassification():
    def __init__(self, slider_w1_1, slider_w1_2, slider_b1_1, slider_b1_2, slider_w2_1, slider_w2_2, slider_b2,
                 slider_w1_12, slider_w1_22):

        # self.label_eq = QtWidgets.QLabel(self)
        # self.label_eq.setText("a = purelin(w2 * tansig(w1 * p + b1) + b2))")
        # self.label_eq.setFont(QtGui.QFont("Times New Roman", 12, italic=True))
        # self.label_eq.setGeometry(180 * self.w_ratio, 270 * self.h_ratio, (self.w_chapter_slider + 100) * self.w_ratio, 50 * self.h_ratio)

        p1, p2 = np.arange(-5, 5, 0.05), np.arange(-5, 5, 0.05)
        self.pp1, self.pp2 = np.meshgrid(p1, p2)

        # self.make_plot(1, (5, 400, 260, 260))
        # self.make_plot(2, (255, 400, 260, 260))
        self.figure = plt.figure(figsize=(5, 5))
        self.figure2 = plt.figure()
        # self.figure2.subplots_adjust(left=0.15, bottom=0.175, right=0.95)
        self.figure2 = go.Figure(
            layout=dict(height=270,
                        title="Function F",
                        margin=dict(l=0, r=0, b=0, t=0, pad=4, ),
                        )
        )

        # self.axis3d = self.figure.add_subplot(projection='3d')
        # self.axis3d.set_xlim(-5, 5)
        # self.axis3d.set_ylim(-5, 5)
        # self.axis3d.set_zlim(-2, 1)
        # self.axis3d.set_xlabel("$p1$")
        # self.axis3d.set_ylabel("$p2$")
        # self.axis3d.set_zlabel("$a$")

        self.figure2.update_layout(
            scene=dict(
                xaxis=dict(range=[min(p1), max(p1)]),
                yaxis=dict(range=[min(p2), max(p2)]),
                zaxis=dict(range=[-3, 2]),
                xaxis_title="p1",
                yaxis_title="p2",
                zaxis_title="a",
            )
        )

        x_0_surf, y_0_surf = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
        xx_0_surf, yy_0_surf = np.meshgrid(x_0_surf, y_0_surf)
        # self.axis3d.plot_surface(xx_0_surf, yy_0_surf, np.zeros((100, 100)), color="gray", alpha=0.5)

        self.figure2.add_trace(
            go.Surface(x=x_0_surf, y=y_0_surf, z=np.zeros((100, 100)), colorscale="Greys", opacity=0.5))
        # self.axis3d.set_xticks([-5, 0, 5])
        # self.axis3d.set_yticks([-5, 0, 5])
        # self.axis3d.set_zticks([-2, -1, 0, 1])
        # self.axis3d.set_xlabel("$p1$")
        # self.axis3d.set_ylabel("$p2$")

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.scatter([1, -1], [1, -1], marker="*", s=100)
        self.axis.scatter([1, -1], [-1, 1], marker="o", s=100)
        self.axis.set_xticks([-5, 0, 5])
        self.axis.set_yticks([-5, 0, 5])

        self.axis.set_xlabel("$p1$")
        self.axis.set_ylabel("$p2$")

        # self.make_slider("slider_w1_1", QtCore.Qt.Orientation.Horizontal, (-40, 40), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
        #                  (10, 115, 150, 50), self.graph, "label_w1_1", "W1(1,1)", (50, 115 - 25, 100, 50))
        # self.slider_w1_1.valueChanged.connect(self.slider_update)
        # self.slider_w1_1.sliderPressed.connect(self.slider_disconnect)
        # self.slider_w1_1.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_w1_2", QtCore.Qt.Orientation.Horizontal, (-40, 40), QtWidgets.QSlider.TickPosition.TicksBelow, 1, -10,
        #                  (10, 360, 150, 50), self.graph, "label_w1_2", "W1(2,1)", (50, 360 - 25, 100, 50))
        # self.slider_w1_2.valueChanged.connect(self.slider_update)
        # self.slider_w1_2.sliderPressed.connect(self.slider_disconnect)
        # self.slider_w1_2.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_b1_1", QtCore.Qt.Orientation.Horizontal, (-10, 10), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
        #                  (170, 115, 150, 50), self.graph, "label_b1_1", "b1(1):", (210, 115 - 25, 100, 50))
        # self.slider_b1_1.valueChanged.connect(self.slider_update)
        # self.slider_b1_1.sliderPressed.connect(self.slider_disconnect)
        # self.slider_b1_1.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_b1_2", QtCore.Qt.Orientation.Horizontal, (-10, 10), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
        #                  (170, 360, 150, 50), self.graph, "label_b1_2", "b1(2):", (210, 360 - 25, 100, 50))
        # self.slider_b1_2.valueChanged.connect(self.slider_update)
        # self.slider_b1_2.sliderPressed.connect(self.slider_disconnect)
        # self.slider_b1_2.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_w2_1", QtCore.Qt.Orientation.Horizontal, (-20, 20), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 20,
        #                  (330, 115, 150, 50), self.graph, "label_w2_1", "W2(1,1):", (370, 115 - 25, 100, 50))
        # self.slider_w2_1.valueChanged.connect(self.slider_update)
        # self.slider_w2_1.sliderPressed.connect(self.slider_disconnect)
        # self.slider_w2_1.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_w2_2", QtCore.Qt.Orientation.Horizontal, (-20, 20), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 20,
        #                  (330, 360, 150, 50), self.graph, "label_w2_2", "W2(1,2):", (370, 360 - 25, 100, 50))
        # self.slider_w2_2.valueChanged.connect(self.slider_update)
        # self.slider_w2_2.sliderPressed.connect(self.slider_disconnect)
        # self.slider_w2_2.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_b2", QtCore.Qt.Orientation.Horizontal, (-10, 10), QtWidgets.QSlider.TickPosition.TicksBelow, 1, -10,
        #                  (self.x_chapter_usual, 380, self.w_chapter_slider, 50), self.graph, "label_b2", "b2: -1.0")
        # self.slider_b2.valueChanged.connect(self.slider_update)
        # self.slider_b2.sliderPressed.connect(self.slider_disconnect)
        # self.slider_b2.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_w1_12", QtCore.Qt.Orientation.Horizontal, (-40, 40), QtWidgets.QSlider.TickPosition.TicksBelow, 1, -10,
        #                  (self.x_chapter_usual, 450, self.w_chapter_slider, 50), self.graph, "label_w1_12", "W1(1,2): 1")
        # self.slider_w1_12.valueChanged.connect(self.slider_update)
        # self.slider_w1_12.sliderPressed.connect(self.slider_disconnect)
        # self.slider_w1_12.sliderReleased.connect(self.slider_reconnect)

        # self.make_slider("slider_w1_22", QtCore.Qt.Orientation.Horizontal, (-40, 40), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
        #                  (self.x_chapter_usual, 520, self.w_chapter_slider, 50), self.graph, "label_w1_22", "W1(2,2): 1")
        # self.slider_w1_22.valueChanged.connect(self.slider_update)
        # self.slider_w1_22.sliderPressed.connect(self.slider_disconnect)
        # self.slider_w1_22.sliderReleased.connect(self.slider_reconnect)

        # self.weight1_1 = self.slider_w1_1.value() / 10
        # self.weight1_2 = self.slider_w1_2.value() / 10
        # self.bias1_1 = self.slider_b1_1.value() / 10
        # self.bias1_2 = self.slider_b1_2.value() / 10
        # self.weight2_1 = self.slider_w2_1.value() / 10
        # self.weight2_2 = self.slider_w2_2.value() / 10
        # self.bias2 = self.slider_b2.value() / 10
        # self.weight1_12 = self.slider_w1_12.value() / 10
        # self.weight1_22 = self.slider_w1_22.value() / 10
        self.weight1_1 = slider_w1_1
        self.weight1_2 = slider_w1_2
        self.bias1_1 = slider_b1_1
        self.bias1_2 = slider_b1_2
        self.weight2_1 = slider_w2_1
        self.weight2_2 = slider_w2_2
        self.bias2 = slider_b2
        self.weight1_12 = slider_w1_12
        self.weight1_22 = slider_w1_22

        # self.make_button("random_button", "Random", (self.x_chapter_button, 580, self.w_chapter_button, self.h_chapter_button), self.on_random)
        # self.make_button("random_button", "Reset", (self.x_chapter_button, 610, self.w_chapter_button, self.h_chapter_button), self.on_reset)

        # self.slider_update()

        self.do_graph = True
        self.graph()
        self.do_graph = False

    def slider_update(self):

        self.weight1_1 = self.slider_w1_1.value() / 10
        self.weight1_2 = self.slider_w1_2.value() / 10
        self.bias1_1 = self.slider_b1_1.value() / 10
        self.bias1_2 = self.slider_b1_2.value() / 10
        self.weight2_1 = self.slider_w2_1.value() / 10
        self.weight2_2 = self.slider_w2_2.value() / 10
        self.bias2 = self.slider_b2.value() / 10
        self.weight1_12 = self.slider_w1_12.value() / 10
        self.weight1_22 = self.slider_w1_22.value() / 10

        # self.label_w1_1.setText("W1(1,1): " + str(self.weight1_1))
        # self.label_w1_2.setText("W1(2,1): " + str(self.weight1_2))
        # self.label_b1_1.setText("b1(1): " + str(self.bias1_1))
        # self.label_b1_2.setText("b1(2): " + str(self.bias1_2))
        # self.label_w2_1.setText("W2(1,1): " + str(self.weight2_1))
        # self.label_w2_2.setText("W2(1,2): " + str(self.weight2_2))
        # self.label_b2.setText("b2: " + str(self.bias2))
        # self.label_w1_12.setText("W1(1,2): " + str(self.weight1_12))
        # self.label_w1_22.setText("W1(2,2): " + str(self.weight1_22))

    def slider_disconnect(self):
        self.sender().valueChanged.disconnect(self.graph)

    def slider_reconnect(self):
        self.do_graph = True
        self.sender().valueChanged.connect(self.graph)
        self.sender().valueChanged.emit(self.sender().value())
        self.do_graph = False

    def on_random(self):
        self.do_graph = False
        self.slider_w1_1.setValue(round(np.random.uniform(-40, 40)))
        self.slider_w1_2.setValue(round(np.random.uniform(-40, 40)))
        self.slider_w1_12.setValue(round(np.random.uniform(-40, 40)))
        self.slider_w1_22.setValue(round(np.random.uniform(-40, 40)))
        self.slider_b1_1.setValue(round(np.random.uniform(-10, 10)))
        self.slider_b1_2.setValue(round(np.random.uniform(-10, 10)))
        self.slider_w2_1.setValue(round(np.random.uniform(-20, 20)))
        self.slider_w2_2.setValue(round(np.random.uniform(-20, 20)))
        self.slider_b2.setValue(round(np.random.uniform(-10, 10)))
        self.do_graph = True
        self.graph()
        self.do_graph = False

    def on_reset(self):
        self.do_graph = False
        self.slider_w1_1.setValue(10)
        self.slider_w1_2.setValue(-10)
        self.slider_w1_12.setValue(-10)
        self.slider_w1_22.setValue(10)
        self.slider_b1_1.setValue(10)
        self.slider_b1_2.setValue(10)
        self.slider_w2_1.setValue(20)
        self.slider_w2_2.setValue(20)
        self.slider_b2.setValue(-20)
        self.do_graph = True
        self.graph()
        self.do_graph = False

    def graph(self):

        if not self.do_graph:
            return

        weight_1, bias_1 = np.array([[self.weight1_1, self.weight1_2]]), np.array([[self.bias1_1, self.bias1_2]])
        weight_2, bias_2 = np.array([[self.weight2_1], [self.weight2_2]]), np.array([[self.bias2]])

        # a = W2(1)*exp(-((p-W1(1)).*b1(1)).^2) + W2(2)*exp(-((p-W1(2)).*b1(2)).^2) + b2
        out = weight_2[0, 0] * np.exp(
            -((self.pp1 - weight_1[0, 0]) * bias_1[0, 0]) ** 2 - ((self.pp2 - self.weight1_12) * bias_1[0, 0]) ** 2)
        out += weight_2[1, 0] * np.exp(
            -((self.pp1 - weight_1[0, 1]) * bias_1[0, 1]) ** 2 - ((self.pp2 - self.weight1_22) * bias_1[0, 0]) ** 2) + \
               bias_2[0, 0]

        # if len(self.axis3d.collections) > 1:
        #     self.axis3d.collections[1].remove()
        # self.axis3d.plot_surface(self.pp1, self.pp2, out, color="cyan")
        self.figure2.add_trace(go.Surface(x=self.pp1, y=self.pp2, z=out, colorscale="tealgrn", showscale=False))
        # self.canvas.draw()

        out_gray = 1 * (out >= 0)
        while len(self.axis.collections) > 2:
            self.axis.collections[-1].remove()
        self.axis.contourf(self.pp1, self.pp2, out_gray, cmap=plt.cm.Paired, alpha=0.6)
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
        st.subheader('RBF Pattern Classification')
        # st.subheader('Regularization')
        # st.subheader('')

    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    st.markdown('---')

    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo/book_logos/17.svg")), unsafe_allow_html=True)
        st.markdown("Alter the network's parameters by dragging the slide bars.")
        st.markdown("Click on [Random]/[Reset] to set each parameter to a random/original value.")
        st.markdown("You can rotate the 3D plots by clicking and dragging in the plot window.")
        random = st.button('Random')
        reset = st.button('Reset')


    if 'slider_w1_1' not in st.session_state:
        st.session_state['slider_w1_1'] = 1.0
        st.session_state['slider_w1_2'] = -1.0
        st.session_state['slider_b1_1'] = 1.0
        st.session_state['slider_b1_2'] = 1.0
        st.session_state['slider_w2_1'] = 2.0
        st.session_state['slider_w2_2'] = 2.0
        st.session_state['slider_b2'] = -1.0
        st.session_state['slider_w1_12'] = -1.0
        st.session_state['slider_w1_22'] = 1.0


    if reset:
        st.session_state['slider_w1_1'] = 1.0
        st.session_state['slider_w1_2'] = -1.0
        st.session_state['slider_b1_1'] = 1.0
        st.session_state['slider_b1_2'] = 1.0
        st.session_state['slider_w2_1'] = 2.0
        st.session_state['slider_w2_2'] = 2.0
        st.session_state['slider_b2'] = -1.0
        st.session_state['slider_w1_12'] = -1.0
        st.session_state['slider_w1_22'] = 1.0

    if random:
        st.session_state['slider_w1_1'] = np.random.uniform(-4.0, 4.0)
        st.session_state['slider_w1_2'] = np.random.uniform(-4.0, 4.0)
        st.session_state['slider_b1_1'] = np.random.uniform(-1.0, 1.0)
        st.session_state['slider_b1_2'] = np.random.uniform(-1.0, 1.0)
        st.session_state['slider_w2_1'] = np.random.uniform(-2.0, 2.0)
        st.session_state['slider_w2_2'] = np.random.uniform(-2.0, 2.0)
        st.session_state['slider_b2'] = np.random.uniform(-1.0, 1.0)
        st.session_state['slider_w1_12'] = np.random.uniform(-4.0, 4.0)
        st.session_state['slider_w1_22'] = np.random.uniform(-4.0, 4.0)


    input_col_0 = st.columns(3)
    with input_col_0[0]:
        slider_w1_1 = st.slider('W1(1,1)', -4.0, 4.0, st.session_state['slider_w1_1'])
    with input_col_0[1]:
        slider_b1_1 = st.slider('b1(1)', -1.0, 1.0, st.session_state['slider_b1_1'])
    with input_col_0[2]:
        slider_w2_1 = st.slider('W2(1,1)', -2.0, 2.0, st.session_state['slider_w2_1'])

    st.markdown(load_svg_2(get_image_path("Figures/nnd17_1.svg")), unsafe_allow_html=True)

    #st.image('media/Figures/nnd17_1.svg', use_column_width=True, caption='RBF Network')

    input_col_1 = st.columns(3)
    with input_col_1[0]:
        slider_w1_2 = st.slider('W1(2,1)', -4.0, 4.0, st.session_state['slider_w1_2'])
    with input_col_1[1]:
        slider_b1_2 = st.slider('b1(2)', -1.0, 1.0, st.session_state['slider_b1_2'])
    with input_col_1[2]:
        slider_w2_2 = st.slider('W2(1,2)', -2.0, 2.0, st.session_state['slider_w2_2'])

    with st.sidebar:
        slider_b2 = st.slider('b2', -1.0, 1.0, st.session_state['slider_b2'])
        slider_w1_12 = st.slider('W1(1,2)', -4.0, 4.0, st.session_state['slider_w1_12'])
        slider_w1_22 = st.slider('W1(2,2)', -4.0, 4.0, st.session_state['slider_w1_22'])
        st.subheader('*Chapter17*')
        st.markdown('---')


    app = PatternClassification(slider_w1_1, slider_w1_2, slider_b1_1, slider_b1_2, slider_w2_1, slider_w2_2, slider_b2,
                                slider_w1_12, slider_w1_22)

    st.text('')
    st.text('')
    fig_cols = st.columns([10, 9])
    with fig_cols[1]:
        st.pyplot(app.figure)

    with fig_cols[0]:
        app.figure2.update_layout(scene=dict(
            aspectmode="cube",
            xaxis_title="p1",
            yaxis_title="p2",
            zaxis_title="a",
            camera=dict(
                eye=dict(x=1, y=1, z=1)
            ),
            # colorbar=Frralse  # Turn off colorbar
        ))
        app.figure2.update_traces(showscale=False)

        st.plotly_chart(app.figure2, use_container_width=True)