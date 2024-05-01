import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from st_pages import hide_pages
from constants import pages_created
import os
import base64


# Differential equation for the shunting network
def grossberg_layer1(y, t, pp, pn, bp, bn, e):
   dydt = [(-y[0] + (bp - y[0]) * pp - (y[0] + bn) * pn) / e,
           (-y[1] + (bp - y[1]) * pn - (y[1] + bn) * pp) / e]
   return dydt





# Function to set random values and rerun the script
def set_random_values():
   st.session_state['pp'] = np.random.uniform(0, 10)
   st.session_state['pn'] = np.random.uniform(0, 10)
   st.session_state['bp'] = np.random.uniform(0, 5)
   st.session_state['bn'] = np.random.uniform(0, 5)
   st.session_state['e'] = np.random.uniform(0.1, 5)
   st.experimental_rerun()


# Initialize session state for sliders and history of responses if they don't exist
if 'pp' not in st.session_state:
   st.session_state['pp'] = 1.0
if 'pn' not in st.session_state:
   st.session_state['pn'] = 0.0
if 'bp' not in st.session_state:
   st.session_state['bp'] = 1.0
if 'bn' not in st.session_state:
   st.session_state['bn'] = 0.0
if 'e' not in st.session_state:
   st.session_state['e'] = 1.0
if 'history' not in st.session_state:
   st.session_state['history'] = []

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
      st.subheader('Grossberg Layer1')
      # st.subheader('')

   with header_cols[0]:
      st.subheader('*Neural Network*')
      st.subheader('DESIGN')

   st.markdown('---')

   with st.sidebar:
      st.markdown(load_svg(get_image_path("Logo/book_logos/18.svg")), unsafe_allow_html=True)
      st.markdown(
         'Use the slide bars to adjust the input and the time constant (eps) to the leaky integrator.\n\n Click [Clear] to remove old responses.'
         '\n\n Click [Random] for random parameters."'
         )
      pp = st.sidebar.slider("Input p+:", 0.0, 10.0, st.session_state['pp'], step=0.1, key='pp')
      pn = st.sidebar.slider("Input p-:", 0.0, 10.0, st.session_state['pn'], step=0.1, key='pn')
      bp = st.sidebar.slider("Bias b+:", 0.0, 5.0, st.session_state['bp'], step=0.1, key='bp')
      bn = st.sidebar.slider("Bias b-:", 0.0, 5.0, st.session_state['bn'], step=0.1, key='bn')
      e = st.sidebar.slider("Time Constant:", 0.1, 5.0, st.session_state['e'], step=0.1, key='e')
      clear = st.sidebar.button("Clear")
      random = st.sidebar.button("Random", on_click=set_random_values)

# Placeholder to show the plot in the main page
plot_placeholder = st.empty()


# Integrate the differential equation and update the history
fig, ax = plt.subplots()
t = np.arange(0, 5.01, 0.01)
y0 = [0, 0]
sol = odeint(grossberg_layer1, y0, t, args=(pp, pn, bp, bn, e))
if not clear and not random:  # Do not update history if clearing or generating random values
   st.session_state['history'].append(sol)
   if len(st.session_state['history']) > 3:
       st.session_state['history'].pop(0)  # Keep only the last 3 responses


# Plot the responses
fig, ax = plt.subplots()
colors = ['lightgrey', 'green', 'red']  # Colors for 3rd, 2nd, and 1st most recent responses
for i, past_response in enumerate(st.session_state['history'][-3:]):
   ax.plot(t, past_response, color=colors[i])


ax.set_xlim(0, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel("Time")
ax.set_ylabel("Output n")
ax.set_title("Response")


# Clear the history if the 'Clear' button is pressed
if clear:
   st.session_state['history'] = []


# Show the plot
plot_placeholder.pyplot(fig)

