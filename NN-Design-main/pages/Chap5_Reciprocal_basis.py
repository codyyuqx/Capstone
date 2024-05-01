import os
import numpy as np
import io
import base64
import streamlit as st 
import matplotlib.pyplot as plt
import matplotlib
from draggable_canvas.src.draggable_canvas import draggable_canvas


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

class ReciprocalBasis:

    def __init__(self):

        self.fig2, self.axes2 = plt.subplots(figsize=(4, 4), dpi=100)
        # Controlling properties of text and its layout on the plot.
        bottom, height = 0.25, 0.5
        left, width = 0.25, 0.5
        right = left + width
        top = bottom + height

        self.label_explanation1 = None
        self.label_explanation2 = None
        
        # Fig object for second plot
        self.axes2_proj = self.axes2.quiver([0], [0], [0], [0],  units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0, linestyle='dashed', color="red")
        self.axes2_proj_line, = self.axes2.plot([], "--")
        self.axes2_proj_line.set_color("red")
        self.text3, self.text4, self.text5 = None, None, None
        self.axes2_v1 = self.axes2.quiver([0], [0], [0], [0], units="xy", scale=1, color="green")
        self.axes2_v2 = self.axes2.quiver([0], [0], [0], [0], units="xy", scale=1, color="green")
        
        self.axes2_l1 = self.axes2.quiver([0], [0], [0], [0], units="xy", scale=0.5, color="black")
        self.axes2_l2 = self.axes2.quiver([0], [0], [0], [0], units="xy", scale=0.5, color="black")
        self.axes2_x = self.axes2.quiver([0], [0], [0], [0], units="xy", scale=0.5, color="red")
        self.axes2.set_title('Expanded vectors', fontsize=13)
        self.axes2.set_xlim(-2, 2)
        self.axes2.set_ylim(-2, 2)
        self.axes2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        self.axes2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        self.axes2.grid(False)

    def expand(self, vector1, vector2, vector3):
         #The vectors sent from the streamlit component renders the vector from left to right, top to bottom according to d3.js renderings
        
        v1 = [int(vector1['x']), int(vector1['y'])]
        v2 = [int(vector2['x']), int(vector2['y'])]
        x1 = [int(vector3['x']), int(vector3['y'])]
        origin = {'x': 150,'y': 150}
        
        #Convert the vectors from d3 client coordiantes to cartesian coordinates
        cartesian_v1 = ((v1[0] - origin['x']) / 100, (origin['y'] - v1[1]) / 100)
        cartesian_v2 = ((v2[0] - origin['x']) / 100, (origin['y'] - v2[1]) / 100)
        cartesian_x =  ((x1[0] - origin['x']) / 100, (origin['y'] - x1[1]) / 100)
          
        b = np.array([[cartesian_v1[0], cartesian_v1[1]],
                        [cartesian_v2[0], cartesian_v2[1]]])
        x = np.array([[cartesian_x[0], cartesian_x[1]]])
        xv = np.dot(np.linalg.inv(b), x.T)
        
        self.label_explanation1 = f"Your vector x is: {round(cartesian_x[0], 2)} * s1 + {round(cartesian_x[1], 2)} * s2"
        self.label_explanation2 = f"The expansion for x in terms of v1 and v2 is: {round(xv[0, 0], 2)} * v1 + {round(xv[1, 0], 2)} * v2" 
    
        
        # self.ax2_proj.set_UVC(proj[0], proj[1])
        # self.ax2_proj_line.set_data([proj[0], v2[0]], [proj[1], v2[1]]
        self.axes2_l1.set_UVC((xv[0]/2) * cartesian_v1[0] , (xv[0]/2) * cartesian_v1[1])
        
        self.axes2_l2 = self.axes2.quiver([xv[0] * cartesian_v1[0]], [xv[0] * cartesian_v1[1]],
                                                [cartesian_x[0] - xv[0] * cartesian_v1[0]], [cartesian_x[1] - xv[0] * cartesian_v1[1]],
                                                units="xy", scale=1, color="black")
        
        self.axes2_v1 = self.axes2.quiver([0], [0], cartesian_v1[0], cartesian_v1[1], units="xy", scale=1, color="green")
        self.axes2_v2 = self.axes2.quiver([0], [0], cartesian_v2[0], cartesian_v2[1], units="xy", scale=1, color="green")
        self.axes2_x =  self.axes2.quiver([0], [0], cartesian_x[0],  cartesian_x[1],  units="xy", scale=1, color="red")
        self.text3 = self.axes2.text(cartesian_v1[0] * 1.3, cartesian_v1[1] * 1.3, "v1")              
        self.text4 = self.axes2.text(cartesian_v2[0] * 1.3, cartesian_v2[1] * 1.3, "v2")
        self.text5 = self.axes2.text(x[0][0] * 1.3, x[0][1] * 1.8, "x")
        self.fig2.canvas.draw()           

    def clear_all(self):
        self.axes2_v1.set_UVC(0, 0)
        self.axes2_v2.set_UVC(0, 0)
        self.axes2_x.set_UVC(0, 0)
        self.axes2_proj_line.set_data([], [])
        self.axes2_l1.set_UVC(0,0)
        self.axes2_l2.set_UVC(0,0)
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
            'Reciprocal Basis, 5, Click on the plot to define the basis {v1, v2}'
            'and the vector x to be expanded in terms of {v1, v2}'
            '<br>'
            '<br>'
            'Click [Expand] to expand a new vector. Click [Start] to define a new basis.'
            '<br>'
            '<br>'
            'Click [Clear] to clear all and restart</p>', unsafe_allow_html=True) 
        
        st.subheader('*Chapter5*')
        st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
    
        
    header_cols = st.columns([4, 2])
    with header_cols[1]:
        st.markdown("""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="font">
                    <span class="right-title-line"><em>RECIPROCAL</em></span>
                    <br>
                    <span class="right-title-line">BASIS</span>
                    </div>
                </div>    
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
    st.markdown('<div class="blue-line"></div>', unsafe_allow_html=True)
    
    #Class instance and second fig object
    reciprocal_basis = ReciprocalBasis()
    value = draggable_canvas(300, 300, "recip", "onRender3Click")
    vectors = []
    
    with st.sidebar:
        if value is not None and len(value) > 0:
            vector1 = value[0]
            vector2 = value[1]
            vector3 = value[2]
            vectors.append(vector1)
            vectors.append(vector2)
            vectors.append(vector3)
        
            if st.button("Expand", use_container_width=True):
                reciprocal_basis.expand(vectors[0], vectors[1], vectors[2])
                st.markdown('<p class="content-font">{}</p>'.format(reciprocal_basis.label_explanation1), unsafe_allow_html=True)
                st.markdown('<p class="content-font">{}</p>'.format(reciprocal_basis.label_explanation2), unsafe_allow_html=True)
        
            if st.button("Start", use_container_width=True):
                reciprocal_basis.clear_all()
                st.rerun()
    
                
    # if "points" not in st.session_state:
    #     st.session_state["points"] = []  
    # Centering the plot in a 12-column grid layout  
    st.write('<div style="display:flex; position:absolute; margin-left:50px; align-items:center;">', unsafe_allow_html=True)
                
    # Save the figure as SVG
    svg_buffer = io.BytesIO()
    fig2 = reciprocal_basis.fig2
    
    # Adjust font size
    reciprocal_basis.axes2.tick_params(axis='both', which='major', labelsize=9)
    fig2.savefig(svg_buffer, format='svg')
    svg_buffer.seek(0)

    # Display the SVG using Streamlit
    st.write(svg_buffer.read().decode("utf-8"), unsafe_allow_html=True)
    
    st.write('</div>', unsafe_allow_html=True)


    
if __name__=="__main__":
    main()