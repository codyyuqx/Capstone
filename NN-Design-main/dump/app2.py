import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

global W1, b1, W2, b2

def init_params(p, S1):
    global W1, b1, W2, b2
    p_min, p_max = np.min(p), np.max(p)
    magw = 2.8 * S1 ** (1 / 1)

    # Initialize weights with random values between -0.5 and 0.5, scaled for sigmoid activation
    W1 = np.random.uniform(-0.5, 0.5, (S1, 1))
    W1[W1 < 0] = -1
    W1[W1 >= 0] = 1
    W1 *= magw

    # Initialize bias for hidden layer with random values between -0.5 and 0.5
    b1 = magw * np.random.uniform(-0.5, 0.5, (S1, 1))

    # Adjust weights and biases to account for input range
    rng = p_min - p_max
    mid = 0.5 * (p_min + p_max)
    W1 = 2 * W1 / rng
    b1 = b1 - W1 * mid

    # Initialize output layer weights with random values between -0.5 and 0.5
    W2 = np.random.uniform(-0.5, 0.5, (1, S1))

    # Initialize output layer bias with a random value between -0.5 and 0.5
    b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    # Update global variables (assuming these are defined elsewhere)
    


def plot_f(p, ax):
    # Calculate network output for given inputs
    a1 = 1 / (1 + np.exp(-(W1 @ p + b1)))
    y = W2 @ a1 + b2

    # Clear any previous plot lines
    ax.lines.clear()

    # Plot the input-output relationship
    ax.plot(p, y, "b-")

    # Set appropriate limits for better visualization
    ax.set_xlim(np.min(p), np.max(p))
    ax.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)

    # Refresh the plot to display the updates
    plt.draw()

def f_to_approx(p, diff):
    # ... (Implementation of f_to_approx function)
    return 1 + np.sin(np.pi * p * diff / 5)


def animate_init_v2():
    global W1, b1, W2, b2, p, S1, mu, a1, a2, e, error_prev, ii
    np.random.seed(5)  # Set a fixed seed for consistent animation initialization

    # Call init_params to initialize weights and biases based on S1
    init_params(p, S1)

    error_goal_reached = False  # Flag to indicate convergence (optional)

    # Reshape input data for matrix operations
    p = p.reshape(1, -1)

    # Calculate initial hidden layer activation using sigmoid function
    a1 = 1 / (1 + np.exp(-(W1 @ p + b1)))

    # Calculate initial network output using purelin (linear) activation
    a2 = (W2 @ a1) + b2


    # Calculate initial error between network output and target function
    e = f_to_approx(p, diff) - a2
    # Calculate initial squared error
    error_prev = np.dot(e, e.T).item()

    # Set initial learning rate
    mu = 0.01

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

    # Initialize network approximation line for animation
    net_approx.set_data([], [])

    # Return the network approximation line for updating the plot
    return net_approx,


def on_animate_v2(idx):
    global W1, b1, W2, b2, mu, a1, a2, e, error_prev, ii  # Ensure global variable access

    # Update learning rate based on convergence
    if error_prev <= 1e-3:  # Adjust tolerance as needed
        mu /= 10

    # Calculate Jacobian matrix for backpropagation (might need adjustments based on your network architecture)
    jac1 = np.kron(a1, np.ones((1, 1))) * np.diag(W2.T @ np.diag(1 - a2**2) @ W1)
    jac2 = np.diag(1 - a1**2) * W2

    jac = np.hstack((jac1, np.diag(a1.T)))
    jac = np.hstack((jac, jac2))
    jac = np.hstack((jac, np.diag(a2.T)))

    # Calculate error gradient (might need  adjustments based on your network architecture)
    je = np.dot(jac.T, e.T)

    # Update weights and biases using a scaled gradient descent approach
    dW = -mu * np.dot(np.linalg.inv(ii + mu * ii), je)
    dW1 = dW[:RS]
    db1 = dW[RS:RSS]
    dW2 = dW[RSS:RSS2].reshape(1, -1)
    db2 = dW[RSS2].reshape(1, 1)

    # Update network weights and biases
    W1 += dW1
    b1 += db1
    W2 += dW2
    b2 += db2

    # Calculate new hidden layer activation
    a1 = 1 / (1 + np.exp(-(W1 @ p + b1)))

    # Calculate new network output
    a2 = a1 @ W2 + b2

    # Calculate new error
    e = f_to_approx(p) - a2

    # Calculate new squared error
    error = np.dot(e.T, e).item()

    # Update error for convergence check
    error_prev = error

    # Update network approximation line for plotting
    net_approx.set_data(p.reshape(-1), a2.reshape(-1))

    # Return the updated network approximation line
    return net_approx,

st.title("Generalization")

st.markdown("Click the [Train] button to train the logsig-linear network on the data points.")
st.markdown("Use the slide bars to choose the number of neurons and the difficulty of the data points.")

S1 = st.slider("Number of Hidden Neurons S1:", 1, 9, 4)
diff = st.slider("Difficulty index:", 1, 9, 1)
anim_delay = st.slider("Animation Delay:", 0, 50, 2, step=1) * 10  # Ensure multiples of 10

# Create initial plot
fig, ax = plt.subplots()
p = np.linspace(-2, 2, 11 * diff)
data_to_approx, = ax.plot(p, 1 + np.sin(np.pi * p * diff / 5), "r+", label="Function to Approximate")
net_approx, = ax.plot([], [], label="Network Approximation")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 2)
ax.set_xlabel("Input")
ax.set_ylabel("Target")
ax.set_title("Function Approximation")
ax.legend()
st.pyplot(fig)

if st.button("Train"):
    anim = FuncAnimation(fig, on_animate_v2, init_func=animate_init_v2, frames=100, interval=anim_delay, blit=True)
    st.pyplot(fig)  # Update plot with animation
