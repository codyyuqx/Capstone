from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from st_pages import Page, show_pages, add_page_title, hide_pages

# Tell streamlit that there is a component called Clickable-graphs,
# and that the code to display that component is in the "frontend" folder
frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
	"clickable_graphs", path=str(frontend_dir)
)


# Create the python function that will be called
def clickable_graphs(
    renderer: str = 'svg', 
    height: int | None = None,
    width: int | None = None,
    key: str | None = None,
    render_function: str = None
    
): 
    """ Return the plot cooridnates that was clicked back to streamlit"""
    component_value = _component_func(
        renderer= renderer,
        height = 400,
        width = 300,
        key=key,
        render_function = render_function
        
    )
    return component_value   

def main():

    # Generate unique keys for each call to clickable_graphs
    unique_key ="grams"
    render_function ="onRender2Click"
    spec = None  
    # Call the clickable_graphs function with preferred set to render2click for two clicks
    click_coords = clickable_graphs(spec, key=unique_key, render_function=render_function)
    
    return click_coords
    
if __name__=="__main__":
    main()