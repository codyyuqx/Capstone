import os
import sys
import numpy as np
import io
import base64
import streamlit as st 
import matplotlib.pyplot as plt
import matplotlib
from Clickable_bidirectional_graphs.src.clickable_graphs import clickable_graphs

font = {'size': 11}
matplotlib.rc('font', **font)

st.set_page_config(page_title='Neural Network DESIGN', page_icon='🧠', layout='centered',
                       initial_sidebar_state='auto')

def load_css(filename):
    with open(filename) as css:
        return st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


css_path = os.path.join(".", "media", "CSS", "nnd.css")
load_css(css_path)

def get_image_path(filename):
    return os.path.join(".", "media", "Logo", filename)

def load_svg(svg_file):
    with open(svg_file, "r", encoding="utf-8") as f:
        svg = f.read()
        svg_base64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        # Adjust 'max-width' to increase the size of the image, for example, to 60% of its container
        # You can also adjust 'height' if necessary, but 'auto' should maintain the aspect ratio
        svg_html = f'''
        <div style="text-align: left; width: 100%;">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="SVG Image" style="max-width: 80%; height: 100px; margin: 20px;">
        </div>
        '''
    return svg_html


class GramSchmidt:

    def __init__(self):

        self.fig2, self.ax2 = plt.subplots(figsize=(4, 4), dpi=100)
        # Controlling properties of text and its layout on the plot.
        bottom, height = 0.25, 0.5
        left, width = 0.25, 0.5
        right = left + width
        top = bottom + height

        # Fig object for second plot
        self.ax2_proj = self.ax2.quiver([0], [0], [0], [0],  units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0, linestyle='dashed', color="red")
        self.ax2_proj_line, = self.ax2.plot([], "--")
        self.ax2_proj_line.set_color("red")
        self.text3, self.text4 = None, None
        self.ax2_v1 = self.ax2.quiver([0], [0], [0], [0], units="xy", scale=0.5, color="black")
        self.ax2_v2 = self.ax2.quiver([0], [0], [0], [0], units="xy", scale=0.5, color="black")
        self.ax2.set_title('Orthogonalized vectors', fontsize=13)
        self.ax2.set_xlim(-2, 2)
        self.ax2.set_ylim(-2, 2)
        self.ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        self.ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        self.ax2.grid(False)

    def orthogonalization(self, vector1, vector2):
        v1 = [int(vector1['x']), int(vector1['y'])]
        v2 = [int(vector2['x']), int(vector2['y'])]
        
        if len(v1) & len(v2) == 2:
            
            dotProduct = v1[0] * v2[0] + v1[1] * v2[1]
            norm_vector1 = np.sqrt((v1[0]) **2) + np.sqrt((v1[1]) **2) 
            norm_vector2 = np.sqrt((v2[0]) **2) + np.sqrt((v2[1]) **2)
            cos_value = dotProduct / (norm_vector1 * norm_vector2)
            
            origin = {'x': 150,'y': 150}
            cartesian_v1 = ((v1[0] - origin['x']) / 100, (origin['y'] - v1[1]) / 100)
            cartesian_v2 = ((v2[0] - origin['x']) / 100, (origin['y'] - v2[1]) / 100)
            v1 = np.array(cartesian_v1)
            v2 = np.array(cartesian_v2)
            a = np.dot(v1.T, v2) / np.dot(v1.T, v1)
            proj = a * np.array(v1)   
            y2 = v2 - proj
            # self.ax2_proj.set_UVC(proj[0], proj[1])
            # self.ax2_proj_line.set_data([proj[0], v2[0]], [proj[1], v2[1]])
            self.ax2_v1 = self.ax2.quiver([0], [0], v1[0], v1[1], units="xy", scale=0.8, color="black")
            self.ax2_v2 = self.ax2.quiver([0], [0], y2[0], y2[1], units="xy", scale=0.8, color="black")
            self.ax2_v1.set_UVC(v1[0], v1[1])
            self.ax2_v2.set_UVC(y2[0], y2[1])
            self.text3 = self.ax2.text(v1[0] * 1.3, v1[1] * 1.3, "v1")              
            self.text4 = self.ax2.text(y2[0] * 1.3, y2[1] * 1.5, "v2")
            self.fig2.canvas.draw()           


    def clear_all(self):
        self.ax2_v1.set_UVC(0, 0)
        self.ax2_v2.set_UVC(0, 0)
        self.ax2_proj_line.set_data([], [])
        if self.text3:
            self.text3.remove()
            self.text3 = None
        if self.text4:
            self.text4.remove()
            self.text4 = None

    

def main(): 
    
    with st.sidebar:
        st.markdown(load_svg(get_image_path("Logo_Ch_5.svg")), unsafe_allow_html=True)
        st.markdown(
            '<p class="content-font">' 
            'Gram-Schmidt, 5, Click twice in the top graph to create two vectors to be orthogonalized'
            '<br>'
            '<br>'
            'Then click [Compute] to see the orthogonal vectors. Click [Start] to clear the vectors.'
            '<br>'
            '<br>'
            'Click [Clear] to clear all and restart</p>', unsafe_allow_html=True)
        st.subheader('*Chapter5*')
        st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
        

    
    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.markdown("""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                    <br>
                    <br>
                    <br>
                    <span class="right-title-line">GRAM-SCHMIDT</span>
                    """, unsafe_allow_html=True)

    with header_cols[0]:
        st.markdown("""
           <div style="display: flex; justify-content: space-between; align-items: center;">
               <div class="font">
                   <span class="left-title-line"><em>Neural Network</em></span>
                   <br>
                   <span class="left-title-line">DESIGN</span>
               </div>
           </div>
           """, unsafe_allow_html=True)
    st.markdown('<div class="blue-line" style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    
    value = clickable_graphs(300, 300, "grams", "onRender2Click")
    vectors = []
    #Class instance and second fig object
    gram_schmidt = GramSchmidt()
    
    with st.sidebar:
        if value is not None:
            vector1 = value[0]
            vector2 = value[1]
            vectors.append(vector1)
            vectors.append(vector2)
        
            if st.button("Compute", use_container_width=True):
                gram_schmidt.orthogonalization(vectors[0], vectors[1])
        
            if st.button("Start", use_container_width=True):
                gram_schmidt.clear_all()
                st.rerun()
                
    # if "points" not in st.session_state:
    #     st.session_state["points"] = []  
    # Centering the plot in a 12-column grid layout  
    st.write('<div style="display:flex; position:absolute; margin-left:50px; align-items:center;">', unsafe_allow_html=True)
    #st.write('<div style="margin-left:auto; margin-right:auto;">', unsafe_allow_html=True)
                
    # Save the figure as SVG
    svg_buffer = io.BytesIO()
    fig2 = gram_schmidt.fig2
    
    # Adjust font size
    gram_schmidt.ax2.tick_params(axis='both', which='major', labelsize=9)
    fig2.savefig(svg_buffer, format='svg')
    svg_buffer.seek(0)

    # Display the SVG using Streamlit
    st.write(svg_buffer.read().decode("utf-8"), unsafe_allow_html=True)
    
    st.write('</div>', unsafe_allow_html=True)

            
            
if __name__ == '__main__':
    main()
   