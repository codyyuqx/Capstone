import streamlit as st
import os
import base64

def run():
    st.set_page_config(layout="wide")
    image_path = r"D:\nndesign-demo-master\nndesign-demo-master\nndesigndemos\Logo"
    st.markdown("""
    <style>
    .streamlit-container {
        max-width: 95%;
    }
    .font {
        font-size: 28px !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .content-font {
        font-size: 18px !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    img {
        max-width:100%;
        height: 550px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .custom-button, .custom-button2 {
        width: 100%; /* Set the button to occupy the full column width */
        padding: 0.5rem 1rem;
        font-size: 1.25rem;
        font-family: 'Times New Roman', Times, serif; /* Set the font family to Times New Roman */
        color: white;
        border: none;
        border-radius: 0.3rem;
        transition: background-color 0.3s ease;
        cursor: pointer; /* To make it look like a button */
        text-align: center; /* Center the text inside the button */
        display: block; /* Make the button a block element */
        text-decoration: none; /* Remove underline from links */
    }
    .custom-button {
        background-color: #589dd4;
        border: 3px solid #589dd4;
    }
    .custom-button2 {
        background-color: #93c5d1;
        border: 3px solid #93c5d1;
    }
    .custom-button:hover {
        background-color: #93c5d1;
        color: black;
    }
    .custom-button2:hover {
        background-color: #589dd4;
        color: black;
    }
    .blue-line {
        height: 4px;
        background-color: darkblue;
        margin: 20px 0;
    }
    .footer {
        text-align: right;
        font-size: 18px;
        font-family: 'Times New Roman', Times, serif;
    }
    .space-top {
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title of the app
    # st.title('Neural Network Design & Deep Learning Demos')

    # Two columns for the two books
    col1, col2 = st.columns(2)
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
        # st.markdown('<p class="font"><em>Neural Network</em> <br> Design</p>', unsafe_allow_html=True)
        # st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
        st.markdown(load_jpg(get_image_path("Figure.jpg")), unsafe_allow_html=True)
        #st.image('D:\nndesign-demo-master\nndesign-demo-master\nndesigndemos\Logo\Figure.jpg', use_column_width=True)

        if st.button('Neural Network Design'):
            st.session_state.page = 'nnd'
            st.experimental_rerun()

        st.markdown("""
            <div class="content-font">Click on the above to access the demonstrations for the Neural Network Design book.

            </div>
            <div class="content-font">
                Each demo is linked to a chapter section of the book. You can find more info at <a href="https://hagan.okstate.edu/nnd.html/" target="_blank">this link</a>.
            </div>
            """, unsafe_allow_html=True)

    # Second column for "Neural Network Design: Deep Learning"
    with col4:
        # st.markdown('<p class="font"><em>Neural Network</em> <br> Design: Deep Learning</p>', unsafe_allow_html=True)
        # st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
        st.markdown(load_jpg(get_image_path("Figure_DL.jpg")), unsafe_allow_html=True)
        # st.image('D:\nndesign-demo-master\nndesign-demo-master\nndesigndemos\Logo\Figure_DL.jpg', use_column_width=True)
        st.markdown('<button class="custom-button2">Neural Network Design: Deep Learning</button>',
                    unsafe_allow_html=True)
        st.markdown("""
            <div class="content-font">Click on the above to access the demonstrations for the Neural Network Design: Deep Learning book.

           <div class="content-font">This book is in progress.
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="footer">By Amir Jafari, Martin Hagan, Pedro Uría, Xiao Qi</div>', unsafe_allow_html=True)



