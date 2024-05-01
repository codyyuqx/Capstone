# from nndesigndemos.nndesign_layout import NNDLayout
import numpy as np
from matplotlib.animation import FuncAnimation
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import math
import time

# Global variables and setup
mu_initial = 0.01
w_ratio, h_ratio = 1, 1
figure, axes = plt.subplots()

S1 = 4
diff = 1
p = np.linspace(-2, 2, 100)
W1, b1, W2, b2 = None, None, None, None
mu = 0.01
ani = None
random_state = 0

# axes = figure.add_subplot(111)
figure.subplots_adjust(bottom=0.2, left=0.1)
axes.set_xlim(-2, 2)
axes.set_ylim(0, 2)
axes.tick_params(labelsize=8)
axes.set_xlabel("Input", fontsize=int(10 * (w_ratio + h_ratio) / 2))
axes.xaxis.set_label_coords(0.5, 0.1)
axes.set_ylabel("Target", fontsize=int(10 * (w_ratio + h_ratio) / 2))
axes.yaxis.set_label_coords(0.05, 0.5)
data_to_approx, = axes.plot([], "r+", label="Function to Approximate")
net_approx, = axes.plot([], label="Network Approximation")
axes.legend(loc='lower center', fontsize=int(8 * (w_ratio + h_ratio) / 2), framealpha=0.9, numpoints=1, ncol=3,
            bbox_to_anchor=(0, -.24, 1, -.280), mode='expand')
axes.set_title("Function Approximation")

anim_delay = 20
pause = True

# Function to plot the target function
def plot_f():
    global p
    p = np.linspace(-2, 2, 11 * diff)
    p = p.reshape(-1, 1)
    data_to_approx.set_data(p, 1 + np.sin(np.pi * p * diff / 5))

# Function to approximate
def f_to_approx(p):
    return 1 + np.sin(np.pi * p * diff / 5)

def init_params():
    global W1, b1, W2, b2

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



    # # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    # def on_run(self):
    #     # self.pause_button.setText("Pause")
    #     self.pause = False
    #     if self.ani and self.ani.event_source:
    #         self.ani.event_source.stop()
    #     n_epochs = 100
    #     self.ani = FuncAnimation(self.figure, self.on_animate_v2, init_func=self.animate_init_v2, frames=n_epochs,
    #                              interval=self.anim_delay, repeat=True, blit=True)
    #     # self.ani.save('Animation.mp4')

def animate_init_v2():
    global W1, b1, W2, b2, mu, a1, a2, e, error_prev, ii, RS, RS1, RSS, RSS1, RSS2, RSS3, RSS4, p

    np.random.seed(5)  # Set a fixed seed for consistent animation initialization

    # Initialize weights and biases based on S1 (assuming initialization logic is the same)
    init_params()

    error_goal_reached = False  # Optional flag to indicate convergence (not used here)
    p = p.reshape(1, -1)

    # Calculate initial hidden layer activation using sigmoid function
    a1 = 1 / (1 + np.exp(-(W1 @ p + b1)))

    # Calculate initial network output using purelin (linear) activation
    a2 = W2 @ a1 + b2

    # Calculate initial error between network output and target function
    e = f_to_approx(p) - a2
    # Calculate initial squared error
    error_prev = np.dot(e, e.T).item()

    # Set initial learning rate
    mu = mu_initial

    # Define indices for efficient matrix operations (might need adjustments based on your specific network architecture)
    RS = S1 * 1
    RS1 = RS + 1
    RSS = RS + S1
    RSS1 = RSS + 1
    RSS2 = RSS + S1 * 1
    RSS3 = RSS2 + 1
    RSS4 = RSS2 + 1

    # Create identity matrix for backpropagation calculations
    ii = np.eye(RSS4)

    # Initialize network approximation line for plotting (assuming data is empty initially)
    net_approx.set_data(p, a2)

    # # Return the network approximation line for updating the plot (assuming this is still required)
    # return net_approx,

def animate_update(idx):
    global W1, b1, W2, b2, mu, a1, a2, e, error_prev, ii, RS, RS1, RSS, RSS1, RSS2, RSS3, RSS4, mu

    # Update learning rate based on convergence (might need adjustments)
    mu /= 10
    a1 = np.kron(a1, np.ones((1,1)))
    d2 = lin_delta(a2)
    d1 = tan_delta(a1, d2, W2)

    # Jacobian matrices for backpropagation (might need adjustments based on your network architecture)
    jac1 = marq(np.kron(p, np.ones((1,1))), d1)
    jac2 = marq(a1, d2)
    
    jac = np.hstack((jac1, d1.T))
    jac = np.hstack((jac, jac2))
    jac = np.hstack((jac, d2.T))

    # Calculate gradient of error using Jacobian and error
    je = np.dot(jac.T, e.T)

    # Update weights and biases using scaled gradient descent (might need adjustments)
    dW = -mu * np.dot(np.linalg.inv(ii + mu * ii), je)
    dW1 = dW[:RS]
    db1 = dW[RS:RSS]
    dW2 = dW[RSS:RSS2].reshape(1, -1)
    db2 = dW[RSS2].reshape(1, 1)

    # Update weights and biases
    W1 += dW1
    b1 += db1
    W2 += dW2
    b2 += db2

    # Calculate new hidden layer activation
    a1 = logsigmoid_stable(np.dot(W1 + dW1, p) + b1 + db1)

    # Calculate new network output
    a2 = purelin(np.dot(W2 + dW2, a1) + b2 + db2)

    # Calculate new error
    e = f_to_approx(p) - a2

    # Calculate new squared error
    error = np.dot(e, e.T).item()

    # Optional convergence check (might need adjustments)
    # if error < error_prev * 0.99:  # Adjust tolerance as needed
    #     mu *= 10

    # Update error for convergence check
    error_prev = error

    # Update network approximation line for plotting
    # net_approx.set_data(p.reshape(-1), a2.reshape(-1))
    # st.write('reached here')
    # Return the updated network approximation line
    # return net_approx,
    
    return [p.reshape(-1,1), a2.reshape(-1,1)]

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




run = st.button('Run')
plot_f()
# st.pyplot(figure)
animate_init_v2()

the_plot = st.pyplot(plt)

for i in range(10,100):
    update = animate_update(i)
    net_approx.set_data(update[0], update[1])
    # st.write(update[0], update[1])
    # st.pyplot(figure)
    the_plot.pyplot(plt)
    time.sleep(0.01)

# net_approx.set_data(p.reshape(-1), a2.reshape(-1))
    
st.pyplot(figure)

st.write(update[0].reshape(-1,1).shape)
        