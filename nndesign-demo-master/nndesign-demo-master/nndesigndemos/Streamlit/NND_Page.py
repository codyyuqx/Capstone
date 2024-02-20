import streamlit as st
import base64
import os


def run():
    # Set the layout to "wide"
    st.set_page_config(layout="wide")
    
    def load_svg(svg_file):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg = f.read()
        svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        svg_html = f'''
        <div style="text-align: center; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 100px; margin: 20px;">
        </div>
        '''
        return svg_html
    def load_css(file_name="../css/nnd.css"):
        with open(file_name) as f:
            css_file = f'<style>{f.read()}</style>'
        return css_file

    css = load_css()
    st.markdown(css, unsafe_allow_html=True)

    # Function to create a dropdown selector with class 'content-font'
    def create_chapter_demo_selector(chapter_name, options):
        demo_option = st.selectbox(
            f"Select a demo for {chapter_name}",
            [f"{chapter_name} demos"] + options,
            index=0,
            key=f'selectbox_{chapter_name}'
        )
        if demo_option == 'One-Input Neuron':
            st.session_state.page = 'One_input_neuron'

    if st.button('Back to Home Page'):
        st.session_state.page = 'landing'
        st.experimental_rerun()

    # Define the relative path for the images using a raw string
    image_path = r"../Logo/book_logos"

    # Title and subtitle
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="font">
            <span class="title-line"><em>Neural Network</em></span>
            <span class="title-line">DESIGN</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="header">Table of content</div>', unsafe_allow_html=True)

    st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)

    # Main content with images, buttons, and dropdowns
    col3, col4 = st.columns([1, 3])

    # Function to safely create an image path
    def get_image_path(filename):
        # Use a raw string for the path
        return os.path.join(image_path, filename)

    def display_chapters_2_to_5():
        with col3:
            st.markdown(load_svg(get_image_path("2.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("3.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("4.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("5.svg")), unsafe_allow_html=True)

        with col4:
            st.markdown('<p class="content-font">Neuron Model & Network Architecture</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 2", ['One-Input Neuron', 'Two-input Neuron'])


            st.markdown('<p class="content-font">An Illustrative Example</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 3",
                                         ['Perception Classification', 'Hamming Classification',
                                          'Hopfield Classification'])

            st.markdown('<p class="content-font">Perceptron Learning Rule</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 4", ['Decision Boundaries', 'Perceptron rules'])

            st.markdown('<p class="content-font">Signal & Weight Vector Spaces</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 5", ['Gram Schmidt', 'Reciprocal Basis'])

        if st.session_state.page == 'One_input_neuron':
            st.experimental_rerun()


    def display_chapters_6_to_9():
        with col3:
            st.markdown(load_svg(get_image_path("6.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("7.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("8.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("9.svg")), unsafe_allow_html=True)

        with col4:
            st.markdown('<p class="content-font">Linear Transformations For N. Networks</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 6", ["Linear transformations", "Eigenvector game"])

            st.markdown('<p class="content-font">Linear Transformations For N. Networks</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 7",
                                         ["Supervised Hebb"])

            st.markdown('<p class="content-font">Performance Surfaces & Optimum Points</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 8",
                                         ["Taylor series #1", "Taylor series #2", "Directional derivatives",
                                          "Quadratic function"])

            st.markdown('<p class="content-font">Performance Optimization</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 9",
                                         ["Steepest descent for Quadratic", "Method comparison", "Newton's method",
                                          "Steepest descent"])

    def display_chapters_10_to_13():
        with col3:
            st.markdown(load_svg(get_image_path("10.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("11.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("12.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("13.svg")), unsafe_allow_html=True)

        with col4:
            st.markdown('<p class="content-font">Widrow - Hoff Learning</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 10", ["Adaptive noise cancellation", "EEG noise cancellation",
                                                        "Linear classification"])

            st.markdown('<p class="content-font"Backpropagation</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 11",
                                         ["Network Function", "Function Approximation", "Generalization"])

            st.markdown('<p class="content-font">Variations on Backpropagation>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 12", ["Steepest Descent #1", "Steepest Descent #2", "Momentum",
                                                        "Variable Learning Rate", "CG Line Search",
                                                        "Conjugate Gradient", "Marquardt Step", "Marquardt"])

            st.markdown('<p class="content-font">Generalizations</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 13", ["Early Stopping", "Regularization", "Bayesian Regularization",
                                                        "Early Stopping-Regularization"])

    def display_chapters_14_to_17():
        with col3:
            st.markdown(load_svg(get_image_path("14.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("15.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("16.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("17.svg")), unsafe_allow_html=True)

        with col4:
            st.markdown('<p class="content-font">Dynamic Networks</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 14", ["FIR Network", "IIR Network", "Dynamic Derivatives",
                                                        "Recurrent Network Training"])

            st.markdown('<p class="content-font">Associative Learning</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 15",
                                         ["Unsupervised Hebb", "Effects of Decay Rate", "Hebb with Decay",
                                          "Graphical Instar", "Outstar"])

            st.markdown('<p class="content-font">Competitive Networks</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 16",
                                         ["Competitive Classification", "Competitive Learning", "1-D Feature Map",
                                          "2-D Feature Map", "LVQ 1", "LVQ 2"])

            st.markdown('<p class="content-font">Radial Basis Function</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 17",
                                         ["Network Function Radial", "Pattern Classification", "Linear Least Squares",
                                          "Orthogonal Least Squares", "Non Linear Optimization"])

    def display_chapters_18_to_21():
        with col3:
            st.markdown(load_svg(get_image_path("18.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("19.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("20.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("21.svg")), unsafe_allow_html=True)

        with col4:
            st.markdown('<p class="content-font">Grossberg Network</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 18", ["Leaky Integrator", "Shunting Network", "Grossberg Layer 1",
                                                        "Grossberg Layer 2", "Adaptive Weights"])

            st.markdown('<p class="content-font">Adaptive Resonance Theory</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 19",
                                         ["ART1 Layer 1", "ART1 Layer 2", "Orienting Subsystem", "ART1 Algorithm"])

            st.markdown('<p class="content-font">Stability</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 20", ["Dynamical System"])

            st.markdown('<p class="content-font">Hopfield Network</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 21", ["Hopfield Network"])

    nav_cols = st.columns(5)  # Creates 5 columns with equal width

    # Define the chapters for each box
    chapter_ranges = ['2–5', '6–9', '10–13', '14–17', '18–21']

    if 'selected_chapter_range' not in st.session_state:
        st.session_state.selected_chapter_range = '2–5'

    # Create a button in each column for chapter navigation
    for col, chapter_range in zip(nav_cols, chapter_ranges):
        if col.button(f'Chapters\n{chapter_range}', key=f'btn_{chapter_range}'):
            st.session_state.selected_chapter_range = chapter_range

    # Display content based on the selected chapter range
    if st.session_state.selected_chapter_range == '2–5':
        display_chapters_2_to_5()
    elif st.session_state.selected_chapter_range == '6–9':
        display_chapters_6_to_9()
    elif st.session_state.selected_chapter_range == '10–13':
        display_chapters_10_to_13()
    elif st.session_state.selected_chapter_range == '14–17':
        display_chapters_14_to_17()
    elif st.session_state.selected_chapter_range == '18–21':
        display_chapters_18_to_21()

    st.markdown('<div class="footer">By Hagan, Jafari, Uría</div>', unsafe_allow_html=True)