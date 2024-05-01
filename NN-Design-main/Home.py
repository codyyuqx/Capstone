import streamlit as st
from st_pages import Page, show_pages, add_page_title, hide_pages
from pages_content import draw_nn_page, draw_dl_page
from constants import pages_created



st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')


def back_home():
    st.session_state['page'] = 'home'
    # st.rerun()


def get_nn():
    container.empty()
    draw_nn_page()
    st.session_state['page'] = 'nn'
    back_to_home = st.button('Back to Home', on_click=back_home)
    st.stop()


def get_dl():
    container.empty()
    draw_dl_page()
    st.session_state['page'] = 'dl'
    st.text('')
    back_to_home = st.button('Back to Home', on_click=back_home)
    st.stop()

hide_pages(pages_created)
container = st.container()

if 'page' not in st.session_state or st.session_state['page'] == 'home':
    with container:
        with open('media/CSS/nnd.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        header_cols = st.columns(2)
        with header_cols[0]:
            st.subheader('*Neural Network*')
            st.subheader('DESIGN')

        with header_cols[1]:
            st.subheader('*Neural Network*')
            st.subheader('DESIGN: DEEP LEARNING')
        st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)

        logo_cols = st.columns(2)
        with logo_cols[0]:
            st.image('media/Logo/Figure.jpg', use_column_width=True)
            st.button('Neural Network Design', type="primary", on_click=get_nn)
            st.markdown('''
            Click on the button above to access the demonstrations of the Neural Network Design book.

            Each demo is linked to a chapter section of the book. You can find more info at https://hagan.okstate.edu/nnd.html/
            ''')

        with logo_cols[1]:
            st.image('media/Logo/Figure_DL.jpg', use_column_width=True)
            st.button('Neural Network Design: Deep Learning', type="secondary", on_click=get_dl)
            st.markdown('''
            Click on the button above to access the demonstrations of the Neural Network Design: Deep Learning book.

            This book is in progress. 
        ''')
    st.text('')
    st.text('')
    # Footer for copyright and attribution
    st.markdown('<div class="footer">By Amir Jafari, Martin Hagan, Pedro Uría, Xiao Qi</div>', unsafe_allow_html=True)

if 'page' in st.session_state:
    if st.session_state['page'] == 'nn':
        get_nn()
    elif st.session_state['page'] == 'dl':
        get_dl()


# if st.session_state['nn']:
#     st.write('NN')
# elif st.session_state['dl']:
#     st.write('DL')




