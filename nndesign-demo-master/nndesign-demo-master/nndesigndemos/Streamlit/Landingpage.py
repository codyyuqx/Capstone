import streamlit as st
import os
import base64

def run():
      # Set the layout to "wide"
    st.set_page_config(layout="wide")
    
    image_path = r"../Logo/"

    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)

    def load_jpg(jpg_file):
        with open(jpg_file, "rb") as f:  # Open the file in binary mode
            jpg_data = f.read()
        jpg_base64 = base64.b64encode(jpg_data).decode('utf-8')
        # Adjust 'max-width' and 'height' as needed to fit the image appropriately in your layout
        jpg_html = f'''
        <div style="text-align: center; width: 100%;">
            <img src="data:image/jpeg;base64,{jpg_base64}" alt="JPEG Image" style="max-width: 100%; height: 500px; margin: 20px;">
        </div>
        '''
        return jpg_html
    
    def load_css(file_name="../css/style.css"):
        with open(file_name) as f:
            css_file = f'<style>{f.read()}</style>'
        return css_file

    css = load_css()
    st.markdown(css, unsafe_allow_html=True)
    
    # Two columns for the two books
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="font">
            <span class="title-line"><em>Neural Network</em></span>
            <br>
            <span class="title-line">DESIGN</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="font">
                <span class="title-line"><em>Neural Network</em></span>
                <br>
                <span class="title-line">DESIGN:DEEP LEARNING</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
    # First column for "Neural Network Design"
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(load_jpg(get_image_path("Figure.jpg")), unsafe_allow_html=True)
        if st.button("Neural Network Design", type="primary"):
            st.session_state.page = 'nnd'
            st.rerun()

        st.markdown("""
            <div class="content-font">Click on the above to access the demonstrations for the Neural Network Design book.
            </div>
            <div class="content-font">
                Each demo is linked to a chapter section of the book. You can find more info at <a href="https://hagan.okstate.edu/nnd.html/" target="_blank">this link</a>.
            </div>
            """, unsafe_allow_html=True)

    # Second column for "Neural Network Design: Deep Learning"
    with col4:
        st.markdown(load_jpg(get_image_path("Figure_DL.jpg")), unsafe_allow_html=True)
        if st.button("Neural Network Design: Deep Learning", type="secondary"):
            st.session_state.page = 'nnd'
            st.rerun()

        st.markdown("""
            <div class="content-font">Click on the above to access the demonstrations for the Neural Network Design: Deep Learning book.

           <div class="content-font">This book is in progress.</div> """, unsafe_allow_html=True)

    st.markdown('<div class="footer">By Amir Jafari, Martin Hagan, Pedro Uría, Xiao Qi</div>', unsafe_allow_html=True)



