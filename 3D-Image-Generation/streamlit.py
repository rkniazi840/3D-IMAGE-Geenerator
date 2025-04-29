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
import streamlit.components.v1 as components
import json

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

# Create custom component for 3D model viewer using Babylon.js
def babylon_viewer(model_url, height=500):
    """
    Custom component to display 3D model using Babylon.js
    """
    # HTML template for Babylon.js viewer
    babylon_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Babylon.js 3D Model Viewer</title>
        <style>
            #babylon-canvas {{
                width: 100%;
                height: {height}px;
                touch-action: none;
                outline: none;
            }}
            #loading-screen {{
                position: absolute;
                width: 100%;
                height: 100%;
                background-color: #ffffff;
                opacity: 0.8;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 10;
            }}
            .spinner {{
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin-bottom: 10px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        <script src="https://cdn.babylonjs.com/babylon.js"></script>
        <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
    </head>
    <body>
        <div id="loading-screen">
            <div class="spinner"></div>
            <div>Loading 3D Model...</div>
        </div>
        <canvas id="babylon-canvas"></canvas>
        <script>
            const loadingScreen = document.getElementById('loading-screen');
            const canvas = document.getElementById('babylon-canvas');
            const engine = new BABYLON.Engine(canvas, true);
            
            const createScene = async function() {{
                // Create scene
                const scene = new BABYLON.Scene(engine);
                scene.clearColor = new BABYLON.Color4(0.95, 0.95, 0.95, 1.0);
                
                // Add a camera
                const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 3, new BABYLON.Vector3(0, 0, 0), scene);
                camera.attachControl(canvas, true);
                camera.wheelPrecision = 50;
                camera.lowerRadiusLimit = 1;
                camera.upperRadiusLimit = 10;
                
                // Add lights
                const light1 = new BABYLON.HemisphericLight("light1", new BABYLON.Vector3(1, 1, 0), scene);
                const light2 = new BABYLON.PointLight("light2", new BABYLON.Vector3(0, 5, -5), scene);
                
                // Load the model
                try {{
                    const result = await BABYLON.SceneLoader.ImportMeshAsync("", "{model_url}", "");
                    
                    // Center and scale the model
                    const meshes = result.meshes;
                    let minX = Number.MAX_VALUE, maxX = Number.MIN_VALUE;
                    let minY = Number.MAX_VALUE, maxY = Number.MIN_VALUE;
                    let minZ = Number.MAX_VALUE, maxZ = Number.MIN_VALUE;
                    
                    for (let i = 0; i < meshes.length; i++) {{
                        const boundingBox = meshes[i].getBoundingInfo().boundingBox;
                        
                        minX = Math.min(minX, boundingBox.minimumWorld.x);
                        maxX = Math.max(maxX, boundingBox.maximumWorld.x);
                        
                        minY = Math.min(minY, boundingBox.minimumWorld.y);
                        maxY = Math.max(maxY, boundingBox.maximumWorld.y);
                        
                        minZ = Math.min(minZ, boundingBox.minimumWorld.z);
                        maxZ = Math.max(maxZ, boundingBox.maximumWorld.z);
                    }}
                    
                    const centerX = (minX + maxX) / 2;
                    const centerY = (minY + maxY) / 2;
                    const centerZ = (minZ + maxZ) / 2;
                    
                    const root = new BABYLON.TransformNode("root");
                    for (let i = 0; i < meshes.length; i++) {{
                        meshes[i].parent = root;
                    }}
                    
                    root.position.x = -centerX;
                    root.position.y = -centerY;
                    root.position.z = -centerZ;
                    
                    // Scale to fit
                    const rangeX = maxX - minX;
                    const rangeY = maxY - minY;
                    const rangeZ = maxZ - minZ;
                    
                    const maxDimension = Math.max(rangeX, rangeY, rangeZ);
                    if (maxDimension > 2) {{
                        const scale = 2 / maxDimension;
                        root.scaling = new BABYLON.Vector3(scale, scale, scale);
                    }}
                    
                    // Position camera to look at model
                    camera.target = new BABYLON.Vector3(0, 0, 0);
                    camera.radius = 3;
                    
                    loadingScreen.style.display = 'none';
                }} catch (error) {{
                    console.error('Error loading model:', error);
                    loadingScreen.innerHTML = 'Error loading 3D model';
                }}
                
                return scene;
            }};
            
            createScene().then(scene => {{
                engine.runRenderLoop(function() {{
                    scene.render();
                }});
                
                window.addEventListener('resize', function() {{
                    engine.resize();
                }});
            }});
        </script>
    </body>
    </html>
    """
    # Render the HTML using streamlit components
    components.html(babylon_html, height=height+50)

# Streamlit UI
st.title("Image to 3D Model Converter")
st.write("Upload an image to generate a 3D model using Unique3D")

# Create a form for file upload and parameters
with st.form("upload_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=list(ALLOWED_EXTENSIONS))
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_bg = st.checkbox("Remove Background", value=True)
        refine_details = st.checkbox("Refine Multiview Details", value=True)
        generate_video = st.checkbox("Generate Video", value=False)
    
    with col2:
        seed = st.number_input("Seed", value=40)
        expansion_weight = st.number_input("Expansion Weight", value=0.2, min_value=0.0, max_value=1.0, step=0.1)
        mesh_init = st.selectbox("Mesh Initialization", options=["std", "thin", "ellipsoid", "sphere"], index=1)
    
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
            
            # Create tabs for different outputs
            tab1, tab2, tab3 = st.tabs(["3D Model", "Video", "Download All"])
            
            with tab1:
                st.subheader("3D Model Viewer")
                
                # Get the relative URL for the model file
                model_filename = os.path.basename(model_path)
                model_url = f"./{OUTPUT_FOLDER}/{model_filename}"
                
                # Display 3D model viewer
                babylon_viewer(model_url, height=500)
                
                # Also provide download option
                st.markdown(
                    get_binary_file_downloader_html(model_path, "3D Model", "Download 3D Model (GLB)"),
                    unsafe_allow_html=True
                )
            
            with tab2:
                st.subheader("Preview Video")
                if video_path and os.path.exists(video_path):
                    st.video(video_path)
                    st.markdown(
                        get_binary_file_downloader_html(video_path, "Video", "Download Preview Video"),
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No preview video was generated")
            
            with tab3:
                st.subheader("Download All Files")
                
                # Prepare files to include in the zip
                files_to_zip = [model_path, image_path]
                if video_path and os.path.exists(video_path):
                    files_to_zip.append(video_path)
                
                # Create a zip file
                base_filename = os.path.splitext(uploaded_file.name)[0]
                zip_filename = f"{base_filename}_3d_package.zip"
                zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for file_path in files_to_zip:
                        if os.path.exists(file_path):
                            zf.write(file_path, os.path.basename(file_path))
                
                st.markdown(
                    get_binary_file_downloader_html(zip_path, "Zip Package", "Download All Files (ZIP)"),
                    unsafe_allow_html=True
                )
        else:
            st.error("Failed to generate 3D model. Please try again.")
    else:
        st.error(f"Invalid file format. Please upload a file with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}")
else:
    if submit_button:
        st.warning("Please upload an image file first.")
