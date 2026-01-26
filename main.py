#!/usr/bin/env python3
"""
Main entry point for Google App Engine deployment.
This file serves as the WSGI application entry point for the Materials Discovery Workshop.
"""

import os
import sys
import gradio as gr
from gradio_app import create_gradio_interface

def create_app():
    """Create and configure the Gradio application."""
    try:
        # Create the Gradio interface
        interface = create_gradio_interface()
        
        # Return the WSGI app
        return interface
        
    except Exception as e:
        print(f"Error creating application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # For local development
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
else:
    # For App Engine deployment
    app = create_app()