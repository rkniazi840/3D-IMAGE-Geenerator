import streamlit as st
import os
import shutil
import tempfile
from werkzeug.utils import secure_filename
from gradio_client import Client
import io
import zipfile
import base64
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configure folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recreate_output_directory():
    """Completely recreate the output directory to ensure it's empty"""
    # Check if the output directory exists and remove it
    if os.path.exists(OUTPUT_FOLDER):
        try:
            # Remove entire directory and all its contents
            shutil.rmtree(OUTPUT_FOLDER)
        except Exception as e:
            st.error(f"Error removing output directory: {e}")
    
    # Create a fresh output directory
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    except Exception as e:
        st.error(f"Error creating output directory: {e}")

def get_binary_file_downloader_html(bin_file, file_label='File', button_text='Download'):
    """Generate a link to download a binary file"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{button_text}</a>'
    return href

def generate_3d_model(uploaded_file, remove_bg, seed, generate_video, refine_details, expansion_weight, mesh_init):
    """Generate 3D model from uploaded image"""
    try:
        # Save uploaded file to disk
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract base filename (without extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Recreate output directory
        recreate_output_directory()
        
        # Initialize Gradio client with Hugging Face token
        with st.spinner("Connecting to Hugging Face Space..."):
            client = Client(
                "https://wuvin-unique3d.hf.space/",
                hf_token=HF_TOKEN
            )
        
        # Make prediction
        try:
            with st.spinner("Generating 3D model, this may take a while..."):
                result = client.predict(
                    filepath,  # image file path
                    remove_bg,  # Remove Background
                    seed,  # Seed
                    generate_video,  # Generate video
                    refine_details,  # Refine Multiview Details
                    expansion_weight,  # Expansion Weight
                    mesh_init,  # Mesh Initialization
                    api_name="/generate3dv2"
                )
        except Exception as e:
            error_message = str(e)
            if "exceeded your GPU quota" in error_message:
                # Extract wait time if available
                wait_time_match = re.search(r"Try again in (\d+:\d+:\d+)", error_message)
                wait_time = wait_time_match.group(1) if wait_time_match else "some time"
                
                st.error(f"⚠️ GPU quota exceeded. Please try again in {wait_time} when your quota resets.")
                st.info("This is a limitation of the free tier on Hugging Face Spaces.")
            else:
                st.error(f"Error processing request: {e}")
            return None, None, None
        
        # The result contains the path to the generated 3D model file
        model_path, video_path = result
        
        if not model_path or not os.path.exists(model_path):
            st.error(f'Failed to generate 3D model or model file not found at {model_path}')
            return None, None, None
        
        # Save the model file to output folder
        output_model_filename = f"{base_filename}_3d_model.glb"
        output_model_path = os.path.join(OUTPUT_FOLDER, output_model_filename)
        
        # Copy the file
        shutil.copyfile(model_path, output_model_path)
        
        # Copy the original image to the output folder
        output_image_filename = f"{base_filename}_original.{filename.split('.')[-1]}"
        output_image_path = os.path.join(OUTPUT_FOLDER, output_image_filename)
        shutil.copyfile(filepath, output_image_path)
        
        # Prepare video path if available
        output_video_path = None
        if video_path and os.path.exists(video_path):
            output_video_filename = f"{base_filename}_video.mp4"
            output_video_path = os.path.join(OUTPUT_FOLDER, output_video_filename)
            shutil.copyfile(video_path, output_video_path)
        
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)
        
        return output_model_path, output_image_path, output_video_path
        
    except Exception as e:
        import traceback
        st.error(f"Error processing request: {e}")
        st.error(traceback.format_exc())
        return None, None, None

def display_3d_model(model_path):
    """Display 3D model using model-viewer component"""
    # Convert the GLB file to Base64
    with open(model_path, "rb") as f:
        glb_bytes = f.read()
    glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
    
    # Create the HTML for the model-viewer component
    model_viewer_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D Model Viewer</title>
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <style>
            model-viewer {{
                width: 100%;
                height: 500px;
                background-color: #f5f5f5;
                --poster-color: transparent;
            }}
        </style>
    </head>
    <body>
        <model-viewer
            id="model"
            camera-controls
            auto-rotate
            shadow-intensity="1"
            src="data:application/octet-stream;base64,{glb_base64}"
            alt="3D Model"
            ar
            ar-modes="webxr scene-viewer quick-look"
            environment-image="neutral">
        </model-viewer>
    </body>
    </html>
    """
    
    # Display using streamlit components
    st.components.v1.html(model_viewer_html, height=520)

# Streamlit UI
st.title("Image to 3D Model Converter")
st.write("Upload an image to generate a 3D model using Unique3D")

# Create a form for file upload with hidden parameters
with st.form("upload_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=list(ALLOWED_EXTENSIONS))
    
    # Hidden parameters with default values
    remove_bg = True
    refine_details = True
    generate_video = False
    seed = 40
    expansion_weight = 0.2
    mesh_init = "thin"
    
    submit_button = st.form_submit_button("Generate 3D Model")


# Process the submission
if submit_button and uploaded_file is not None:
    if allowed_file(uploaded_file.name):
        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width=300)
        
        # Generate the 3D model
        model_path, image_path, video_path = generate_3d_model(
            uploaded_file,
            remove_bg,
            seed,
            generate_video,
            refine_details,
            expansion_weight,
            mesh_init
        )
        
        if model_path and os.path.exists(model_path):
            st.success("3D model generated successfully!")
            
            # Single tab for 3D model output
            st.subheader("3D Model Viewer")
            
            # Display the 3D model
            try:
                display_3d_model(model_path)
            except Exception as e:
                st.error(f"Error displaying 3D model: {e}")
                # Provide download link as fallback
                st.info("3D model viewer couldn't be loaded. You can download the model instead.")
                st.markdown(
                    get_binary_file_downloader_html(model_path, "3D Model", "Download 3D Model (GLB)"),
                    unsafe_allow_html=True
                )
            
            # File validation and error handling
            if uploaded_file:
                if not os.path.splitext(uploaded_file.name)[1].lower() in ALLOWED_EXTENSIONS:
                    st.error(f"Invalid file format. Please upload a file with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}")
            else:
                if submit_button:
                    st.warning("Please upload an image file first.")

