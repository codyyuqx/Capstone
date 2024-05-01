import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import base64
import os
from st_pages import Page, show_pages, add_page_title, hide_pages
from constants import pages_created


# Function to load and display SVG files as Base64 encoded HTML images
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

def get_image_path(filename):
    # Use a raw string for the path
    return os.path.join(image_path, filename)

image_path = 'media/Logo/book_logos'


def draw_nn_page():
    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # if st.button('Back to Landing Page'):
    #     st.session_state.page = 'landing'
    #     st.experimental_rerun()
    hide_pages(pages_created)
    header_cols = st.columns([3, 2])
    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    with header_cols[1]:
        st.subheader('')
        st.subheader('Table of Contents')
    st.markdown('---')

    def create_chapter_demo_selector(chapter_name, options):
        demo_option = st.selectbox(
            f"Select a demo for {chapter_name}",
            [f"{chapter_name} demos"] + options,
            index=0,
            key=f'selectbox_{chapter_name}'
        )
        if demo_option == 'One-Input Neuron':
            switch_page('Chap2_One_Input_Neuron')
        elif demo_option == 'Two-Input Neuron':
            switch_page('Chap2_Two_Input_Neuron')
        elif demo_option == 'Gram Schmidt':
            switch_page('Chap5_Gram_schmidt')
        elif demo_option == 'Reciprocal Basis':
            switch_page('Chap5_Reciprocal_basis')
        elif demo_option == 'Quadratic function':
            switch_page('Chap8_Quadratic_Function')
        elif demo_option == 'Reciprocal Basis':
            switch_page('Chap5_Reciprocal_basis')
        elif demo_option == 'Taylor series #1':
            switch_page('Chap8_Taylor_Series#1')
        elif demo_option == 'Taylor series #2':
            switch_page('Chap8_Taylor_Series#2')
        elif demo_option == 'Method comparison':
            switch_page('Chap9_Method_Comparsion')
        elif demo_option == "Newton's method":
            switch_page('Chap9_Newstons_Method')
        elif demo_option == 'EEG noise cancellation':
            switch_page('Chap10_EEG_Noise_Cancellation')
        elif demo_option == 'Function Approximation':
            switch_page('Chap11_Function_Approximation')
        elif demo_option == 'Generalization':
            switch_page('Chap11_Generalization')
        elif demo_option == 'Network Function':
            switch_page('Chap11_Network_Function')
        elif demo_option == 'Steepest Descent #1':
            switch_page('Chap12_Steepest_Descent#1')
        elif demo_option == 'Network Function':
            switch_page('Chap11_Network_Function')
        elif demo_option == 'Bayesian Regularization':
            switch_page('Chap13_Bayesia_ Regularization')
        elif demo_option == 'Early Stopping Regularization':
            switch_page('Chap13_Early-Stopping_Regularization')
        elif demo_option == 'Regularization':
            switch_page('Chap13_Regularization')
        elif demo_option == 'FIR Network':
            switch_page('Chap14_FIR_Network')
        elif demo_option == 'IIR Network':
            switch_page('Chap14_IIR_Network')
        elif demo_option == 'Effects of Decay Rate':
            switch_page('Chap15_Effects_Of_Decay_Rate')
        elif demo_option == '1-D Feature Map':
            switch_page('Chap16_1-D_Feature_Map')
        elif demo_option == 'Non Linear Optimization':
            switch_page('Chap17_NonLinear_Optimization')
        elif demo_option == 'Orthogonal Least Square':
            switch_page('Chap17_Orthogonal_Least_Square')
        elif demo_option == 'Pattern Classification':
            switch_page('Chap17_Pattern_Classification')
        elif demo_option == 'Adaptive Weights':
            switch_page('Chap18_Adaptive_Weights')
        elif demo_option == 'Grossberg Layer 1':
            switch_page('Chap18_Grossberg_Layer1')
        elif demo_option == 'Grossberg Layer 2':
            switch_page('Chap18_Grossberg_Layer2')
        elif demo_option == 'Leaky Integrator':
            switch_page('Chap18_Leaky_Integrator')
        elif demo_option == 'Shunting Network':
            switch_page('Chap18_Shunting_Network')
        elif demo_option == 'ART1 Layer 1':
            switch_page('Chap19_ART1_Layer_1')
        elif demo_option == 'ART1 Layer 2':
            switch_page('Chap19_ART1_Layer_2')
        elif demo_option == 'Orienting Subsystem':
            switch_page('Chap19_Orienting_Subsystem')
        # elif demo_option == 'Dynamic Derivatives':
        #     switch_page('Dynamic Derivatives')
        # elif demo_option == 'Effects of Decay Rate':
        #     switch_page('Effects of Decay Rate')


    col3, col4 = st.columns([1, 3])
    def display_chapters_2_to_5():
        with col3:
            st.markdown(load_svg(get_image_path("2.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("3.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("4.svg")), unsafe_allow_html=True)
            st.markdown(load_svg(get_image_path("5.svg")), unsafe_allow_html=True)

        with col4:
            st.markdown('<p class="content-font">Neuron Model & Network Architecture</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 2", ['One-Input Neuron', 'Two-Input Neuron'])

            st.markdown('<p class="content-font">An Illustrative Example</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 3",
                                         ['Perception Classification', 'Hamming Classification',
                                          'Hopfield Classification'])

            st.markdown('<p class="content-font">Perceptron Learning Rule</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 4", ['Decision Boundaries', 'Perceptron rules'])

            st.markdown('<p class="content-font">Signal & Weight Vector Spaces</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 5", ['Gram Schmidt', 'Reciprocal Basis'])

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

            st.markdown('<p class="content-font">Backpropagation</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 11",
                                         ["Network Function", "Function Approximation", "Generalization"])

            st.markdown('<p class="content-font">Variations on Backpropagation</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 12", ["Steepest Descent #1", "Steepest Descent #2", "Momentum",
                                                        "Variable Learning Rate", "CG Line Search",
                                                        "Conjugate Gradient", "Marquardt Step", "Marquardt"])

            st.markdown('<p class="content-font">Generalizations</p>', unsafe_allow_html=True)
            create_chapter_demo_selector("Chapter 13", ["Early Stopping", "Regularization", "Bayesian Regularization",
                                                        "Early Stopping Regularization"])

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
        if col.button(f'\n\n{chapter_range}\n', key=f'btn_{chapter_range}'):
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


    # # chapter 10
    # chapter_col_1 = st.columns([1,5])
    # with chapter_col_1[0]:
    #     st.markdown(load_svg(get_image_path("10.svg")), unsafe_allow_html=True)
    # with chapter_col_1[1]:
    #     st.markdown('Chapter 10: Widrow - Hoff Learning')
    #     chapter_select_1 = st.selectbox('',['Chapter 10 demos', 'EEG Noice Cancellation'])
    #
    # # st.text('.')
    #
    # # chapter 11
    # chapter_col_2 = st.columns([1,5])
    # with chapter_col_2[0]:
    #     st.markdown(load_svg(get_image_path("11.svg")), unsafe_allow_html=True)
    # with chapter_col_2[1]:
    #     st.markdown('Chapter 11: Backpropagation')
    #     chapter_select_2 = st.selectbox('.', ['Chapter 11 demos', 'Generalization', 'Function Approximation'])
    #
    # # st.text('.')
    #
    # # chapter 13
    # chapter_col_3 = st.columns([1,5])
    # with chapter_col_3[0]:
    #     st.image('media/Logo/book_logos/13.svg', use_column_width=True)
    # with chapter_col_3[1]:
    #     st.markdown('Chapter 12: Generalization')
    #     chapter_select_3 = st.selectbox('.', ['Chapter 13 demos', 'Regularization', 'Bayesian Regularization', 'Early-Stopping Regularization'])
    #
    # # st.text('.')
    #
    # # chapter 14
    # chapter_col_4 = st.columns([1,5])
    # with chapter_col_4[0]:
    #     st.image('media/Logo/Logo_Ch_14.svg', use_column_width=True)
    # with chapter_col_4[1]:
    #     st.markdown('Chapter 14: Dynamic Networks')
    #     chapter_select_4 = st.selectbox('.', ['Chapter 14 demos', 'FIR Network', 'IIR Network', 'Dynamic Derivatives'])
    #
    # # chapter 15
    # chapter_col_4 = st.columns([1,5])
    # with chapter_col_4[0]:
    #     st.image('media/Logo/Logo_Ch_15.svg', use_column_width=True)
    # with chapter_col_4[1]:
    #     st.markdown('Chapter 15: Associative Learning')
    #     chapter_select_5 = st.selectbox('.', ['Chapter 15 demos', 'Effects of Decay Rate'])
    
    # if chapter_select_1 == 'EEG Noice Cancellation':
    #     switch_page('EEG Noice Cancellation')
    # elif chapter_select_2 == 'Generalization':
    #     switch_page('generalization')
    # elif chapter_select_2 == 'Function Approximation':
    #     switch_page('Function Approximation')
    # elif chapter_select_3 == 'Regularization':
    #     switch_page('Regularization')
    # elif chapter_select_3 == 'Bayesian Regularization':
    #     switch_page('Bayesian Regularization')
    # elif chapter_select_3 == 'Early-Stopping Regularization':
    #     switch_page('Early-Stopping Regularization')
    # elif chapter_select_4 == 'FIR Network':
    #     switch_page('FIR Network')
    # elif chapter_select_4 == 'IIR Network':
    #     switch_page('IIR Network')
    # elif chapter_select_4 == 'Dynamic Derivatives':
    #     switch_page('Dynamic Derivatives')
    # elif chapter_select_5 == 'Effects of Decay Rate':
    #     switch_page('Effects of Decay Rate')
    # else:
    #     st.text('')

    





def draw_dl_page():
    with open('media/CSS/home_page.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    header_cols = st.columns([3, 2])
    with header_cols[0]:
        st.subheader('*Neural Network*')
        st.subheader('DESIGN')

    with header_cols[1]:
        st.subheader('')
        st.subheader('Table of Contents')
    st.markdown('---')

    # chapter 10
    chapter_col_1 = st.columns([1,5])
    with chapter_col_1[0]:
        st.image('media/Logo/Logo_Ch_10.svg', use_column_width=True)
    with chapter_col_1[1]:
        st.markdown('Chapter 10: Neuron Modle & Network Architectures')
        chapter_select_1 = st.selectbox('', ['Function Approximation', 'Generalization'])

    # st.text('.')

    # chapter 11
    chapter_col_2 = st.columns([1,5])
    with chapter_col_2[0]:
        st.image('media/Logo/Logo_Ch_11.svg', use_column_width=True)
    with chapter_col_2[1]:
        st.markdown('Chapter 11: Error Functions & Optimization')
        chapter_select_2 = st.selectbox('', ['Error Functions', 'Optimization'])

    # st.text('.')

    # chapter 12
    chapter_col_3 = st.columns([1,5])
    with chapter_col_3[0]:
        st.image('media/Logo/Logo_Ch_12.svg', use_column_width=True)
    with chapter_col_3[1]:
        st.markdown('Chapter 12: Training & Testing')
        chapter_select_3 = st.selectbox('', ['Training', 'Testing'])

    # st.text('.')

    # chapter 13
    chapter_col_4 = st.columns([1,5])
    with chapter_col_4[0]:
        st.image('media/Logo/Logo_Ch_13.svg', use_column_width=True)
    with chapter_col_4[1]:
        st.markdown('Chapter 13: Overfitting & Regularization')
        chapter_select_4 = st.selectbox('', ['Overfitting', 'Regularization'])