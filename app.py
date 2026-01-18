"""
HuggingFace Spaces app.py - redirects to gradio_app.py
"""
from gradio_app import create_gradio_interface

# Create and launch the interface
interface = create_gradio_interface()
interface.launch()
