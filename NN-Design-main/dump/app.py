import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from generalization import Generalization
from dump.static_methods import *


st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='wide', initial_sidebar_state='auto')
st.title('Neural Network DESIGN')

with st.sidebar:
    st.markdown('Click the [Train] button to train the logsig-linear network on the data points.')
    st.markdown('Use the slider bars to choose the number of neurons and the difficulty of the data points.')

    train = st.button('Train')
    pause = st.button('Pause')
    animation_delay = st.slider('Animation delay (ms)', 0, 1000, 100, 100)

import numpy as np
import streamlit as st


def initialize_params(p, S1):
    p_min, p_max = np.min(p), np.max(p)
    magw = 2.8 * S1 ** (1 / 1)
    W1 = np.random.uniform(-0.5, 0.5, (S1, 1))
    W1[W1 < 0] = -1
    W1[W1 >= 0] = 1
    W1 *= magw
    b1 = magw * np.random.uniform(-0.5, 0.5, (S1, 1))
    rng = p_min - p_max
    mid = 0.5 * (p_min + p_max)
    W1 = 2 * W1 / rng
    b1 = b1 - W1 * mid
    W2 = np.random.uniform(-0.5, 0.5, (1, S1))
    b2 = np.random.uniform(-0.5, 0.5, (1, 1))
    return W1, b1, W2, b2



st.title("Generalization Demonstration")
st.write("This app shows a neural network's generalization capability.")

diff = st.slider("Difficulty Index", min_value=1, max_value=9, value=1)

# Load data points for approximation
p = np.linspace(-2, 2, 30)
p = p.reshape(-1, 1)  # Reshape for matrix operations

# Define the function to approximate
f_to_approx = lambda p: 1 + np.sin(np.pi * p * diff / 5)

# Set number of hidden neurons
S1 = 7

# Initialize network parameters
W1, b1, W2, b2 = initialize_params(p, S1)  # Implement initialize_params()


# Difficulty index (default: 1)

# st.write("Difficulty Index:", diff)
# st.write("Number of Hidden Neurons:", S1)   
# st.write("Initial Weights (W1):", W1)
# st.write("Initial Bias (b1):", b1)
# st.write("Initial Weights (W2):", W2)
# st.write("Initial Bias (b2):", b2)
# st.write(p)
# Number of hidden neurons (default: 4)
final_a1 = logsigmoid_stable(np.dot(W1, p.T) + b1)
final_a2 = purelin(np.dot(W2, final_a1) + b2)

fig, ax = plt.subplots()

# Plot the data to approximate
ax.plot(p, f_to_approx(p), "r+", label="Function to Approximate")

# Plot the network's final approximation
ax.plot(p, final_a2.T, label="Network Approximation")

ax.set_xlabel("Input")
ax.set_ylabel("Target")
ax.set_title("Function Approximation")
ax.legend()

st.pyplot(fig)

