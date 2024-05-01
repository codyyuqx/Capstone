# draggable_canvas

Streamlit component that sends mouse drag coordinates and initial plot to the streamlit app

## Installation instructions 

```sh
pip install draggable_canvas
```

## Usage instructions

```python
import streamlit as st

from draggable_canvas import draggable_canvas

value = draggable_canvas()

st.write(value)
