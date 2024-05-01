import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from st_pages import hide_pages
from constants import pages_created
import os
import base64


# Function to calculate the Leaky Integrator response
def leaky_integrator(p, tcte, t):
   return p * (1 - np.exp(-t / tcte))



# Initialize session state variables for the sliders if they don't exist
if 'p' not in st.session_state:
   st.session_state['p'] = 1.0
if 'tcte' not in st.session_state:
   st.session_state['tcte'] = 1.0
if 'history' not in st.session_state:
   st.session_state['history'] = []


# Define a function to set random values
def set_random_values():
   st.session_state['p'] = np.random.uniform(0, 10)
   st.session_state['tcte'] = np.random.uniform(0.1, 5)
   # Add the random values to the history
   update_history(st.session_state['p'], st.session_state['tcte'])


def update_history(p, tcte):
   if len(st.session_state['history']) >= 3:
       st.session_state['history'].pop(0)  # Keep only the last 3 entries
   st.session_state['history'].append((p, tcte))

if __name__ == "__main__":
   st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                      initial_sidebar_state='auto')

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


   hide_pages(pages_created)

   def get_image_path(filename):
      # Use a raw string for the path
      return os.path.join(image_path, filename)


   image_path = 'media'

   header_cols = st.columns([4, 2])
   with header_cols[1]:
      st.text('')
      st.subheader('Leaky Integrator')
      # st.subheader('')

   with header_cols[0]:
      st.subheader('*Neural Network*')
      st.subheader('DESIGN')

   st.markdown('---')

   with st.sidebar:
      st.markdown(load_svg(get_image_path("Logo/book_logos/18.svg")), unsafe_allow_html=True)

      st.markdown("Use the slide bars to adjust the input and the time constant (eps) to the leaky integrator.\n\nClick [Clear] to remove old responses.\n\nClick [Random] for random parameters.")


# Create sliders with callbacks to update the history
      p = st.sidebar.slider("Input p:", 0.0, 10.0, st.session_state['p'], 0.1, key='p',
                     on_change=update_history, args=(st.session_state['p'], st.session_state['tcte']))
      tcte = st.sidebar.slider("Time Constant:", 0.1, 5.0, st.session_state['tcte'], 0.1, key='tcte',
                        on_change=update_history, args=(st.session_state['p'], st.session_state['tcte']))


# Buttons for clear and random
      clear = st.sidebar.button("Clear")
      random = st.sidebar.button("Random", on_click=set_random_values)


# Main plot area
fig, ax = plt.subplots()
t = np.arange(0, 5.1, 0.1)


# Plot previous lines in history in light grey
for past_p, past_tcte in st.session_state['history']:
   past_y = leaky_integrator(past_p, past_tcte, t)
   ax.plot(t, past_y, color="lightgrey")


# Plot the current line in red
current_y = leaky_integrator(p, tcte, t)
ax.plot(t, current_y, color="red")


# Set up the plot
ax.set_xlim(0, 5)
ax.set_ylim(0, 10)
ax.set_xlabel("Time")
ax.set_ylabel("Output n")
ax.set_title("Response")


# Clear the plot if clear button is pressed
if clear:
   # Reset the history
   st.session_state['history'] = []
   # Redraw the plot without any lines
   ax.clear()
   ax.set_xlim(0, 5)
   ax.set_ylim(0, 10)
   ax.set_xlabel("Time")
   ax.set_ylabel("Output n")
   ax.set_title("Response")


# Show the plot
st.pyplot(fig)



