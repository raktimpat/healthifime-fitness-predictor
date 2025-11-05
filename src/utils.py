import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


IMAGE_OUTPUT_DIR = '../visuals/'

def save_plotly_fig(fig, filename):
    """Saves a Plotly figure to a static image file."""
    if not os.path.exists(IMAGE_OUTPUT_DIR):
        os.makedirs(IMAGE_OUTPUT_DIR)
    
    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    try:
        fig.write_image(filepath)
        print(f"Saved figure to {filepath}")
    except Exception as e:
        print(f"Could not save figure. Do you have 'kaleido' installed? (pip install kaleido)")
        print(f"Error: {e}")