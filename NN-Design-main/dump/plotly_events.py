import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

from streamlit_plotly_events import plotly_events

# Writes a component similar to st.write()
r = range(100)
x = []
for i in range(100):
    x.extend(r)

y = []
for i in range(100):
    y.extend([i] * 100)

# st.write(x,y)
fig = go.Figure(
    data=[go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'))],
    layout=go.Layout(
        hovermode='closest',
    )
  
)

fig.update_layout(clickmode='event+select',hovermode='closest')
selected_points = plotly_events(fig)
st.write('Selected points:', selected_points)
points = st.plotly_chart(fig)
