import streamlit as st

# Define functions for each button click
def show_page_one():
  st.write("This is page one content!")

def show_page_two():
  st.write("This is page two content!")

# Create the home page
st.title("Welcome!")
col1, col2 = st.columns(2)  # Create two columns for buttons

# Add buttons with click events
button1 = col1.button("Go to Page One")
button2 = col2.button("Go to Page Two")

# Check for button clicks and display respective content
if button1:
  st.write("---")  # Add a separator after the home page content
  show_page_one()
  st.stop()  # Stop rendering to prevent showing home page again

if button2:
  st.write("---")
  show_page_two()
  st.stop()

# Display a message if no buttons are clicked
if not (button1 or button2):
  st.write("Click a button to navigate.")
