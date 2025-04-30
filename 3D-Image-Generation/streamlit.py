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
import requests
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configure folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure allowed extensions - expanded to include more image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

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

def convert_image_format(input_path, output_path, format='JPEG'):
    """Convert image to specified format to ensure compatibility"""
    try:
        img = Image.open(input_path)
        # Convert to RGB if RGBA to avoid issues with some formats
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output_path, format=format)
        return True
    except Exception as e:
        logger.error(f"Error converting image: {e}")
        return False

def get_binary_file_downloader_html(bin_file, file_label='File', button_text='Download'):
    """Generate a link to download a binary file"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}" class="download-button">{button_text}</a>'
    return href

def test_hf_connection(space_url):
    """Test connection to Hugging Face Space before attempting to use it"""
    try:
        response = requests.get(f"{space_url}/", timeout=10)
        if response.status_code == 200:
            return True
        else:
            logger.warning(f"Hugging Face Space returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Hugging Face Space: {e}")
        return False

def generate_3d_model(uploaded_file, remove_bg, seed, generate_video, refine_details, expansion_weight, mesh_init, retries=3, retry_delay=5):
    """Generate 3D model from uploaded image with retry mechanism"""
    try:
        # Save uploaded file to disk
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Convert image to JPEG for maximum compatibility
        jpeg_filepath = os.path.join(temp_dir, "converted.jpg")
        if not convert_image_format(filepath, jpeg_filepath):
            st.error("Failed to convert uploaded image. Please try a different image.")
            return None, None, None
        
        # Extract base filename (without extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Recreate output directory
        recreate_output_directory()
        
        # Hugging Face Space URL
        hf_space_url = "https://wuvin-unique3d.hf.space"
        
        # Test connection first
        if not test_hf_connection(hf_space_url):
            st.error("⚠️ Cannot connect to the Hugging Face Space. The service might be down or rate-limited.")
            st.info("Try again later or switch to an alternative 3D model generator.")
            return None, None, None
        
        # Initialize Gradio client with Hugging Face token
        attempt = 0
        while attempt < retries:
            try:
                with st.spinner(f"Connecting to Hugging Face Space (Attempt {attempt+1}/{retries})..."):
                    client = Client(
                        hf_space_url,
                        hf_token=HF_TOKEN
                    )
                break
            except ValueError as e:
                error_msg = str(e)
                if "429" in error_msg:  # Rate limit error
                    attempt += 1
                    if attempt < retries:
                        st.warning(f"Rate limited by Hugging Face. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        st.error("⚠️ Maximum retry attempts reached. Hugging Face is rate-limiting requests.")
                        st.info("Please try again later when the rate limit resets.")
                        return None, None, None
                else:
                    st.error(f"Error connecting to Hugging Face: {e}")
                    return None, None, None
        
        # Make prediction
        try:
            with st.spinner("Generating 3D model, this may take a while..."):
                # Display progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 5):
                    time.sleep(1)  # Simulate processing time
                    progress_bar.progress(percent_complete)
                    
                result = client.predict(
                    jpeg_filepath,  # image file path
                    remove_bg,  # Remove Background
                    seed,  # Seed
                    generate_video,  # Generate video
                    refine_details,  # Refine Multiview Details
                    expansion_weight,  # Expansion Weight
                    mesh_init,  # Mesh Initialization
                    api_name="/generate3dv2"
                )
                
                progress_bar.progress(100)
        except Exception as e:
            error_message = str(e)
            if "exceeded your GPU quota" in error_message or "429" in error_message:
                # Extract wait time if available
                wait_time_match = re.search(r"Try again in (\d+:\d+:\d+)", error_message)
                wait_time = wait_time_match.group(1) if wait_time_match else "some time"
                
                st.error(f"⚠️ GPU quota exceeded or rate limit hit. Please try again in {wait_time} when your quota resets.")
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
        output_image_filename = f"{base_filename}_original.jpg"
        output_image_path = os.path.join(OUTPUT_FOLDER, output_image_filename)
        shutil.copyfile(jpeg_filepath, output_image_path)
        
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
            .controls {{
                display: flex;
                justify-content: center;
                margin-top: 10px;
            }}
            .controls button {{
                margin: 0 5px;
                padding: 5px 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
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
        <div class="controls">
            <button onclick="document.getElementById('model').setAttribute('auto-rotate', true)">Start Rotation</button>
            <button onclick="document.getElementById('model').setAttribute('auto-rotate', false)">Stop Rotation</button>
        </div>
    </body>
    </html>
    """
    
    # Display using streamlit components
    st.components.v1.html(model_viewer_html, height=570)

def create_fallback_message():
    """Create a message when the HF service is unavailable"""
    st.warning("⚠️ The Hugging Face Space (Unique3D) seems to be unavailable or rate-limited.")
    st.info("Here are some alternatives for 3D model generation:")
    
    alternatives = [
        {
            "name": "Luma AI",
            "url": "https://lumalabs.ai/",
            "description": "Offers advanced 3D model generation from images with realistic textures"
        },
        {
            "name": "Kaedim",
            "url": "https://www.kaedim3d.com/",
            "description": "Specialized in turning 2D concept art into 3D models"
        },
        {
            "name": "Spline",
            "url": "https://spline.design/",
            "description": "Web-based 3D design tool with AI features"
        }
    ]
    
    for alt in alternatives:
        st.markdown(f"**[{alt['name']}]({alt['url']})** - {alt['description']}")
    
    st.markdown("---")
    st.markdown("Or you can try again later when the rate limit resets.")

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white !important;
        text-decoration: none;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-instructions {
        border-left: 3px solid #1E88E5;
        padding-left: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Image to 3D Model Converter")
st.write("Upload an image to generate a 3D model using Unique3D")

# Instructions
with st.expander("📋 Instructions & Tips", expanded=True):
    st.markdown("""
    <div class="upload-instructions">
    <p><strong>How to use:</strong></p>
    <ol>
        <li>Upload a clear image with a single object</li>
        <li>Wait for the 3D model to generate (may take 1-2 minutes)</li>
        <li>View and download your 3D model</li>
    </ol>
    <p><strong>Best practices:</strong></p>
    <ul>
        <li>Use images with plain backgrounds</li>
        <li>Upload high-quality images</li>
        <li>If you get rate limited, try again after a few minutes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Create tabs for basic and advanced options
tab1, tab2 = st.tabs(["Basic", "Advanced"])

with tab1:
    # Create a form for file upload with simplified parameters
    with st.form("upload_form_basic"):
        uploaded_file = st.file_uploader("Choose an image file", type=list(ALLOWED_EXTENSIONS))
        
        # Hidden parameters with default values
        remove_bg = True
        refine_details = True
        generate_video = False
        seed = 40
        expansion_weight = 0.2
        mesh_init = "thin"
        
        submit_button = st.form_submit_button("Generate 3D Model")

with tab2:
    # Advanced form with all parameters exposed
    with st.form("upload_form_advanced"):
        uploaded_file_adv = st.file_uploader("Choose an image file", type=list(ALLOWED_EXTENSIONS))
        
        # Advanced parameters
        remove_bg_adv = st.checkbox("Remove Background", value=True)
        refine_details_adv = st.checkbox("Refine Multiview Details", value=True)
        generate_video_adv = st.checkbox("Generate Preview Video", value=False)
        seed_adv = st.slider("Random Seed", 0, 100, 40)
        expansion_weight_adv = st.slider("Expansion Weight", 0.0, 1.0, 0.2)
        mesh_init_adv = st.selectbox("Mesh Initialization", ["thin", "medium", "thick"], index=0)
        
        submit_button_adv = st.form_submit_button("Generate 3D Model (Advanced)")

# Process the basic submission
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
            
            # Display the 3D model
            st.subheader("3D Model Viewer")
            try:
                display_3d_model(model_path)
            except Exception as e:
                st.error(f"Error displaying 3D model: {e}")
                # Provide download link as fallback
                st.info("3D model viewer couldn't be loaded. You can download the model instead.")
            
            # Add download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    get_binary_file_downloader_html(model_path, "3D Model", "Download 3D Model (GLB)"),
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    get_binary_file_downloader_html(image_path, "Original Image", "Download Original Image"),
                    unsafe_allow_html=True
                )
            
            if video_path and os.path.exists(video_path):
                st.markdown(
                    get_binary_file_downloader_html(video_path, "Preview Video", "Download Preview Video"),
                    unsafe_allow_html=True
                )
        else:
            # Show fallback message with alternatives
            create_fallback_message()
    else:
        st.error(f"Invalid file format. Please upload a file with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}")

# Process the advanced submission
if submit_button_adv and uploaded_file_adv is not None:
    if allowed_file(uploaded_file_adv.name):
        # Display the uploaded image
        st.subheader("Uploaded Image (Advanced Mode)")
        st.image(uploaded_file_adv, width=300)
        
        # Generate the 3D model with advanced parameters
        model_path, image_path, video_path = generate_3d_model(
            uploaded_file_adv,
            remove_bg_adv,
            seed_adv,
            generate_video_adv,
            refine_details_adv,
            expansion_weight_adv,
            mesh_init_adv
        )
        
        if model_path and os.path.exists(model_path):
            st.success("3D model generated successfully with advanced settings!")
            
            # Display the 3D model
            st.subheader("3D Model Viewer (Advanced Mode)")
            try:
                display_3d_model(model_path)
            except Exception as e:
                st.error(f"Error displaying 3D model: {e}")
                # Provide download link as fallback
                st.info("3D model viewer couldn't be loaded. You can download the model instead.")
            
            # Add download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    get_binary_file_downloader_html(model_path, "3D Model", "Download 3D Model (GLB)"),
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    get_binary_file_downloader_html(image_path, "Original Image", "Download Original Image"),
                    unsafe_allow_html=True
                )
            
            if video_path and os.path.exists(video_path):
                st.markdown(
                    get_binary_file_downloader_html(video_path, "Preview Video", "Download Preview Video"),
                    unsafe_allow_html=True
                )
        else:
            # Show fallback message with alternatives
            create_fallback_message()
    else:
        st.error(f"Invalid file format. Please upload a file with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Unique3D")
